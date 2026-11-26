"""Self-contained test runner (no pytest needed) — runs every test, prints PASS/FAIL per case."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src importable
# ---------------------------------------------------------------------------
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from analyst_agent.agent import AnalystAgent, _extract_json
from analyst_agent.guardrails import run_execution_guardrails, verify_claims
from analyst_agent.llm import ScriptedLLMClient
from analyst_agent.models import (
    ColumnProfile, DatabaseProfile, PythonResult, Severity,
    SQLResult, TableProfile,
)
from analyst_agent.profiler import DataSourceManager, _infer_sqlite_type, _sanitize_identifier
from analyst_agent.python_sandbox import PythonSandbox, validate_python_ast
from analyst_agent.sql_sandbox import SQLSandbox, validate_sql_readonly


PLAN_RESPONSE_GOOD = json.dumps({
    "sql_queries": [{"id": "q1", "sql": "SELECT category, SUM(revenue) AS total FROM orders GROUP BY category"}],
    "python_scripts": [{"id": "py1", "code": "record_metric('electronics_total', sum(r['total'] for r in DATA['q1'] if r['category'] == 'Electronics'))"}],
})
# Electronics rows in test CSV: revenue 100 + 150 = 250. Apparel: 200.
SYNTHESIS_RESPONSE_GOOD = (
    "Electronics: **250.0** (q1). Apparel: **200.0** (q1). "
    "Python-computed Electronics subtotal (py1): **250.0**."
)


def _make_csv_dir() -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory(prefix="analyst_test_")
    Path(td.name, "orders.csv").write_text(
        "order_id,revenue,category\n1,100.0,Electronics\n2,200.0,Apparel\n3,150.0,Electronics\n",
        encoding="utf-8",
    )
    return td


def _make_profile() -> DatabaseProfile:
    col = ColumnProfile(name="revenue", inferred_type="REAL",
                        null_count=0, null_rate=0.0, distinct_count=3,
                        sample_values=[100.0, 200.0, 150.0],
                        min_value=100.0, max_value=200.0)
    tbl = TableProfile(name="orders", row_count=3, columns=[col])
    return DatabaseProfile(source_path="mem", source_type="csv", tables=[tbl])


def _sr(**kw) -> SQLResult:
    d = dict(query_id="q1", sql="SELECT 1", columns=["x"], rows=[{"x": 1}],
             row_count=1, truncated=False, elapsed_ms=1.0, error=None)
    d.update(kw)
    return SQLResult(**d)


def _pr(**kw) -> PythonResult:
    d = dict(script_id="py1", code="", stdout="", stderr="",
             metrics={}, elapsed_ms=1.0, exit_code=0, error=None)
    d.update(kw)
    return PythonResult(**d)


# ---------------------------------------------------------------------------
# Mini test framework
# ---------------------------------------------------------------------------
_PASS = []
_FAIL = []


def test(name: str):
    """Decorator: runs the test function, records PASS/FAIL."""
    def decorator(fn):
        try:
            fn()
            _PASS.append(name)
            print(f"  PASS  {name}")
        except Exception:
            tb = traceback.format_exc()
            _FAIL.append((name, tb))
            print(f"  FAIL  {name}")
            for line in tb.splitlines()[-6:]:
                print(f"         {line}")
        return fn
    return decorator


def assert_equal(a, b):
    if a != b:
        raise AssertionError(f"{a!r} != {b!r}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or f"Expected True, got {cond!r}")


def assert_in(needle, haystack):
    if needle not in haystack:
        raise AssertionError(f"{needle!r} not in {haystack!r}")


def assert_close(a, b, tol=0.01):
    if abs(a - b) > tol:
        raise AssertionError(f"|{a} - {b}| = {abs(a-b)} > {tol}")


# ---------------------------------------------------------------------------
# 1. Profiler
# ---------------------------------------------------------------------------
print("\n── Profiler ──────────────────────────────────────────────────")

@test("sanitize_identifier: normal name preserved")
def _():
    assert_equal(_sanitize_identifier("revenue"), "revenue")

@test("sanitize_identifier: spaces → underscores")
def _():
    assert_equal(_sanitize_identifier("order id"), "order_id")

@test("sanitize_identifier: leading digit prefixed")
def _():
    assert_equal(_sanitize_identifier("1column"), "col_1column")

@test("infer_type: integers")
def _():
    assert_equal(_infer_sqlite_type(["1", "2", "3"]), "INTEGER")

@test("infer_type: floats")
def _():
    assert_equal(_infer_sqlite_type(["1.0", "2.5"]), "REAL")

@test("infer_type: text")
def _():
    assert_equal(_infer_sqlite_type(["foo", "bar"]), "TEXT")

@test("infer_type: empty → TEXT")
def _():
    assert_equal(_infer_sqlite_type([]), "TEXT")

@test("DataSourceManager: profile has 'orders' table")
def _():
    td = _make_csv_dir()
    try:
        mgr = DataSourceManager(td.name)
        profile = mgr.profile()
        names = [t.name for t in profile.tables]
        assert_in("orders", names)
        mgr.close()
    finally:
        td.cleanup()

@test("DataSourceManager: row count correct")
def _():
    td = _make_csv_dir()
    try:
        mgr = DataSourceManager(td.name)
        profile = mgr.profile()
        orders = next(t for t in profile.tables if t.name == "orders")
        assert_equal(orders.row_count, 3)
        mgr.close()
    finally:
        td.cleanup()

@test("DataSourceManager: missing path raises FileNotFoundError")
def _():
    raised = False
    try:
        DataSourceManager("/tmp/definitely_not_here_xyz.csv")
    except FileNotFoundError:
        raised = True
    assert_true(raised, "Expected FileNotFoundError")

@test("DataSourceManager: unsupported extension raises ValueError")
def _():
    td = tempfile.mkdtemp()
    p = Path(td) / "data.xlsx"
    p.write_bytes(b"fake")
    raised = False
    try:
        DataSourceManager(p)
    except ValueError:
        raised = True
    assert_true(raised, "Expected ValueError for .xlsx")

@test("DataSourceManager: empty CSV raises ValueError")
def _():
    td = tempfile.mkdtemp()
    p = Path(td) / "empty.csv"
    p.write_text("", encoding="utf-8")
    raised = False
    try:
        DataSourceManager(p)
    except ValueError:
        raised = True
    assert_true(raised, "Expected ValueError for empty CSV")

# ---------------------------------------------------------------------------
# 2. SQL Sandbox
# ---------------------------------------------------------------------------
print("\n── SQL Sandbox ───────────────────────────────────────────────")

@test("validate_sql: SELECT passes")
def _():
    assert_true(validate_sql_readonly("SELECT * FROM orders") is None)

@test("validate_sql: WITH CTE passes")
def _():
    assert_true(validate_sql_readonly("WITH c AS (SELECT 1) SELECT * FROM c") is None)

@test("validate_sql: multi-statement rejected")
def _():
    assert_true(validate_sql_readonly("SELECT 1; SELECT 2") is not None)

@test("validate_sql: DROP rejected")
def _():
    assert_true(validate_sql_readonly("DROP TABLE orders") is not None)

@test("validate_sql: DELETE rejected")
def _():
    assert_true(validate_sql_readonly("DELETE FROM orders WHERE 1=1") is not None)

@test("validate_sql: INSERT rejected")
def _():
    assert_true(validate_sql_readonly("INSERT INTO orders VALUES (1)") is not None)

@test("validate_sql: UPDATE rejected")
def _():
    assert_true(validate_sql_readonly("UPDATE orders SET revenue=0") is not None)

@test("validate_sql: empty string rejected")
def _():
    assert_true(validate_sql_readonly("") is not None)

@test("SQLSandbox: SUM(revenue) = 450")
def _():
    td = _make_csv_dir()
    try:
        mgr = DataSourceManager(td.name)
        sb = SQLSandbox(mgr, max_rows=100, timeout_seconds=5.0)
        res = sb.execute("q1", "SELECT SUM(revenue) AS total FROM orders")
        assert_true(res.error is None, f"Unexpected error: {res.error}")
        assert_close(res.rows[0]["total"], 450.0)
        mgr.close()
    finally:
        td.cleanup()

@test("SQLSandbox: GROUP BY returns 2 rows")
def _():
    td = _make_csv_dir()
    try:
        mgr = DataSourceManager(td.name)
        sb = SQLSandbox(mgr, max_rows=100, timeout_seconds=5.0)
        res = sb.execute("q2", "SELECT category, COUNT(*) AS cnt FROM orders GROUP BY category")
        assert_true(res.error is None, f"Unexpected error: {res.error}")
        assert_equal(res.row_count, 2)
        mgr.close()
    finally:
        td.cleanup()

@test("SQLSandbox: nonexistent column returns SQLiteError")
def _():
    td = _make_csv_dir()
    try:
        mgr = DataSourceManager(td.name)
        sb = SQLSandbox(mgr, max_rows=100, timeout_seconds=5.0)
        res = sb.execute("q3", "SELECT fake_col FROM orders")
        assert_true(res.error is not None, "Expected error for fake_col")
        assert_in("SQLiteError", res.error)
        mgr.close()
    finally:
        td.cleanup()

@test("SQLSandbox: forbidden write blocked at validation layer")
def _():
    td = _make_csv_dir()
    try:
        mgr = DataSourceManager(td.name)
        sb = SQLSandbox(mgr, max_rows=100, timeout_seconds=5.0)
        res = sb.execute("q4", "DELETE FROM orders")
        assert_true(res.error is not None)
        assert_in("SQLSandboxError", res.error)
        mgr.close()
    finally:
        td.cleanup()

@test("SQLSandbox: max_rows truncation works")
def _():
    td = _make_csv_dir()
    try:
        mgr = DataSourceManager(td.name)
        sb = SQLSandbox(mgr, max_rows=1, timeout_seconds=5.0)
        res = sb.execute("q5", "SELECT * FROM orders")
        assert_equal(res.row_count, 1)
        assert_true(res.truncated)
        mgr.close()
    finally:
        td.cleanup()

# ---------------------------------------------------------------------------
# 3. Python Sandbox
# ---------------------------------------------------------------------------
print("\n── Python Sandbox ────────────────────────────────────────────")

@test("validate_python_ast: safe code passes")
def _():
    assert_true(validate_python_ast("record_metric('x', 1 + 2)") is None)

@test("validate_python_ast: import os rejected")
def _():
    assert_true(validate_python_ast("import os") is not None)

@test("validate_python_ast: import subprocess rejected")
def _():
    assert_true(validate_python_ast("import subprocess") is not None)

@test("validate_python_ast: eval() rejected")
def _():
    assert_true(validate_python_ast("eval('1+1')") is not None)

@test("validate_python_ast: exec() rejected")
def _():
    assert_true(validate_python_ast("exec('print(1)')") is not None)

@test("validate_python_ast: open() rejected")
def _():
    assert_true(validate_python_ast("open('/etc/passwd')") is not None)

@test("validate_python_ast: import math passes")
def _():
    assert_true(validate_python_ast("import math; record_metric('pi', math.pi)") is None)

@test("PythonSandbox: basic metric computed correctly")
def _():
    sb = PythonSandbox(timeout_seconds=10.0)
    code = "record_metric('total', sum(r['revenue'] for r in DATA['q1']))"
    data = {"q1": [{"revenue": 100.0}, {"revenue": 200.0}]}
    res = sb.execute("py1", code, data)
    assert_true(res.error is None, f"Unexpected error: {res.error}")
    assert_close(res.metrics["total"], 300.0)

@test("PythonSandbox: disallowed import blocked")
def _():
    sb = PythonSandbox(timeout_seconds=10.0)
    res = sb.execute("py1", "import os; record_metric('x', 1)", {})
    assert_true(res.error is not None)
    assert_in("PythonSandboxError", res.error)

@test("PythonSandbox: runtime error gives non-zero exit code")
def _():
    sb = PythonSandbox(timeout_seconds=10.0)
    res = sb.execute("py1", "raise ValueError('deliberate test error')", {})
    assert_true(res.exit_code != 0)
    assert_true(res.error is not None)

# ---------------------------------------------------------------------------
# 4. Guardrails
# ---------------------------------------------------------------------------
print("\n── Guardrails ────────────────────────────────────────────────")

@test("guardrail: SCHEMA_HALLUCINATION on no-such-column error")
def _():
    profile = _make_profile()
    sr = _sr(error="SQLiteError: no such column: fake_col")
    findings = run_execution_guardrails(profile, [sr], [])
    codes = [f.code for f in findings]
    assert_in("SCHEMA_HALLUCINATION", codes)

@test("guardrail: EMPTY_RESULT warning on 0 rows")
def _():
    profile = _make_profile()
    sr = _sr(rows=[], row_count=0)
    findings = run_execution_guardrails(profile, [sr], [])
    codes = [f.code for f in findings]
    assert_in("EMPTY_RESULT", codes)

@test("guardrail: TRUNCATED_RESULT warning")
def _():
    profile = _make_profile()
    sr = _sr(truncated=True)
    findings = run_execution_guardrails(profile, [sr], [])
    codes = [f.code for f in findings]
    assert_in("TRUNCATED_RESULT", codes)

@test("guardrail: HIGH_NULL_RATE warning on 50% nulls")
def _():
    col = ColumnProfile(name="churn_score", inferred_type="REAL",
                        null_count=25, null_rate=0.50, distinct_count=10,
                        sample_values=[0.1, 0.5])
    tbl = TableProfile(name="customers", row_count=50, columns=[col])
    profile = DatabaseProfile(source_path="mem", source_type="csv", tables=[tbl])
    findings = run_execution_guardrails(profile, [], [])
    assert_true(any(f.code == "HIGH_NULL_RATE" for f in findings))

@test("guardrail: PYTHON_EXECUTION_ERROR flagged")
def _():
    profile = _make_profile()
    pr = _pr(error="PythonRuntimeError (exit=1): ZeroDivisionError", exit_code=1)
    findings = run_execution_guardrails(profile, [], [pr])
    assert_true(any(f.code == "PYTHON_EXECUTION_ERROR" for f in findings))

@test("guardrail: NULL_OR_NAN_METRIC warning")
def _():
    profile = _make_profile()
    pr = _pr(metrics={"average_revenue": None})
    findings = run_execution_guardrails(profile, [], [pr])
    assert_true(any(f.code == "NULL_OR_NAN_METRIC" for f in findings))

@test("guardrail: NO_VALID_SQL_EVIDENCE when all SQL failed")
def _():
    profile = _make_profile()
    sr = _sr(error="SQLiteError: broken")
    findings = run_execution_guardrails(profile, [sr], [])
    assert_true(any(f.code == "NO_VALID_SQL_EVIDENCE" for f in findings))

@test("guardrail: clean run → no ERROR-level findings")
def _():
    profile = _make_profile()
    sr = _sr(rows=[{"total": 450.0}], row_count=1)
    pr = _pr(metrics={"average_revenue": 150.0})
    findings = run_execution_guardrails(profile, [sr], [pr])
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert_equal(errors, [])

# ---------------------------------------------------------------------------
# 5. Claim Verification
# ---------------------------------------------------------------------------
print("\n── Claim Verification ────────────────────────────────────────")

@test("claim_verify: grounded number passes")
def _():
    profile = _make_profile()
    sql = [SQLResult(query_id="q1", sql="SELECT 1", columns=["total_revenue"],
                     rows=[{"total_revenue": 450.0}], row_count=1,
                     truncated=False, elapsed_ms=1.0, error=None)]
    _, findings = verify_claims("Total revenue was **450.0**.", profile, sql, [])
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert_equal(errors, [])

@test("claim_verify: hallucinated number flagged")
def _():
    profile = _make_profile()
    sql = [SQLResult(query_id="q1", sql="SELECT 1", columns=["total"],
                     rows=[{"total": 450.0}], row_count=1,
                     truncated=False, elapsed_ms=1.0, error=None)]
    _, findings = verify_claims("Total revenue was **9999.99**.", profile, sql, [])
    assert_true(any(f.code == "UNGROUNDED_NUMERICAL_CLAIM" for f in findings))

@test("claim_verify: year label not flagged as hallucination")
def _():
    profile = _make_profile()
    _, findings = verify_claims("In 2024, revenue grew.", profile, [], [])
    assert_true(not any(f.code == "UNGROUNDED_NUMERICAL_CLAIM" for f in findings))

@test("claim_verify: percentage matched to decimal metric")
def _():
    profile = _make_profile()
    pr = PythonResult(script_id="py1", code="", stdout="", stderr="",
                      metrics={"growth_rate": 0.25}, elapsed_ms=1.0, exit_code=0, error=None)
    _, findings = verify_claims("Revenue grew by 25%.", profile, [], [pr])
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert_true(errors == [], f"25% should match 0.25 metric; got errors: {errors}")

# ---------------------------------------------------------------------------
# 6. JSON extraction
# ---------------------------------------------------------------------------
print("\n── JSON Extraction ───────────────────────────────────────────")

@test("extract_json: plain JSON dict")
def _():
    result = _extract_json('{"sql_queries": [], "python_scripts": []}')
    assert_equal(result["sql_queries"], [])

@test("extract_json: fenced code block")
def _():
    raw = '```json\n{"sql_queries": [{"id": "q1", "sql": "SELECT 1"}], "python_scripts": []}\n```'
    result = _extract_json(raw)
    assert_equal(result["sql_queries"][0]["id"], "q1")

# ---------------------------------------------------------------------------
# 7. End-to-end Agent Loop (offline, ScriptedLLMClient)
# ---------------------------------------------------------------------------
print("\n── Agent End-to-End ──────────────────────────────────────────")

@test("agent: happy path → sql_result has no error")
def _():
    td = _make_csv_dir()
    try:
        llm = ScriptedLLMClient([PLAN_RESPONSE_GOOD, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=1)
        report = agent.analyze(td.name, "What is total revenue by category?")
        assert_true(len(report.sql_results) == 1)
        assert_true(report.sql_results[0].error is None, f"SQL error: {report.sql_results[0].error}")
    finally:
        td.cleanup()

@test("agent: schema hallucination triggers retry on attempt 1")
def _():
    td = _make_csv_dir()
    try:
        bad_plan = json.dumps({"sql_queries": [{"id": "q1", "sql": "SELECT fake_col FROM orders"}], "python_scripts": []})
        llm = ScriptedLLMClient([bad_plan, PLAN_RESPONSE_GOOD, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=2)
        report = agent.analyze(td.name, "Any question")
        assert_true(report.attempts >= 2, f"Expected >=2 attempts, got {report.attempts}")
    finally:
        td.cleanup()

@test("agent: ungrounded number flagged as UNGROUNDED_NUMERICAL_CLAIM")
def _():
    td = _make_csv_dir()
    try:
        hallucinated = "Total revenue was **99999.99** (hallucinated)."
        llm = ScriptedLLMClient([PLAN_RESPONSE_GOOD, hallucinated])
        agent = AnalystAgent(llm=llm, max_retries=0)
        report = agent.analyze(td.name, "What is revenue?")
        ungrounded = [f for f in report.findings if f.code == "UNGROUNDED_NUMERICAL_CLAIM"]
        assert_true(len(ungrounded) > 0, f"No ungrounded findings; findings={report.findings}")
    finally:
        td.cleanup()

@test("agent: report markdown contains all required sections")
def _():
    td = _make_csv_dir()
    try:
        llm = ScriptedLLMClient([PLAN_RESPONSE_GOOD, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=1)
        report = agent.analyze(td.name, "Revenue?")
        md = report.to_markdown()
        assert_in("## Guardrail", md)
        assert_in("## Claim-to-Evidence", md)
        assert_in("## Reproducible Execution Trace", md)
    finally:
        td.cleanup()

@test("agent: forbidden Python import → PYTHON_EXECUTION_ERROR finding")
def _():
    td = _make_csv_dir()
    try:
        bad_py = json.dumps({
            "sql_queries": [{"id": "q1", "sql": "SELECT SUM(revenue) AS total FROM orders"}],
            "python_scripts": [{"id": "py1", "code": "import os; record_metric('x', os.getpid())"}],
        })
        llm = ScriptedLLMClient([bad_py, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=0)
        report = agent.analyze(td.name, "Revenue?")
        py_errors = [f for f in report.findings if f.code == "PYTHON_EXECUTION_ERROR"]
        assert_true(len(py_errors) > 0, f"Expected PYTHON_EXECUTION_ERROR; findings={report.findings}")
    finally:
        td.cleanup()

@test("agent: invalid plan JSON triggers retry instead of crashing")
def _():
    td = _make_csv_dir()
    try:
        bad_json = '{"sql_queries": [{"id": "q1", "sql": "SELECT 1"}], bad_trailing'
        llm = ScriptedLLMClient([bad_json, PLAN_RESPONSE_GOOD, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=2)
        report = agent.analyze(td.name, "Revenue?")
        assert_equal(report.attempts, 2)
        assert_true(report.validation_passed)
    finally:
        td.cleanup()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
total = len(_PASS) + len(_FAIL)
print(f"Results: {len(_PASS)}/{total} passed, {len(_FAIL)} failed")

if _FAIL:
    print("\nFailed tests:")
    for name, tb in _FAIL:
        print(f"\n  ✗ {name}")
        for line in tb.splitlines()[-4:]:
            print(f"    {line}")
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
