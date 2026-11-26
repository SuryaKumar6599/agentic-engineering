"""Tests — schema profiler, SQL sandbox, Python sandbox, guardrails, and agent loop.

All tests run entirely offline using ScriptedLLMClient and in-memory SQLite.
Run with: cd path/to/project && python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from analyst_agent.agent import AnalystAgent, _extract_json
from analyst_agent.guardrails import run_execution_guardrails, verify_claims
from analyst_agent.llm import ScriptedLLMClient
from analyst_agent.models import DatabaseProfile, PythonResult, Severity, SQLResult, TableProfile, ColumnProfile
from analyst_agent.profiler import DataSourceManager, _infer_sqlite_type, _sanitize_identifier
from analyst_agent.python_sandbox import PythonSandbox, validate_python_ast
from analyst_agent.sql_sandbox import SQLSandbox, validate_sql_readonly


# ─────────────────────────── helpers ──────────────────────────────────────────

@pytest.fixture
def sample_csv_dir(tmp_path: Path) -> Path:
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "order_id,revenue,category\n"
        "1,100.0,Electronics\n"
        "2,200.0,Apparel\n"
        "3,150.0,Electronics\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def manager(sample_csv_dir: Path):
    mgr = DataSourceManager(sample_csv_dir)
    yield mgr
    mgr.close()


@pytest.fixture
def sql_sandbox(manager) -> SQLSandbox:
    return SQLSandbox(manager, max_rows=100, timeout_seconds=5.0)


@pytest.fixture
def py_sandbox() -> PythonSandbox:
    return PythonSandbox(timeout_seconds=10.0)


def _make_profile(row_count: int = 3) -> DatabaseProfile:
    col = ColumnProfile(
        name="revenue", inferred_type="REAL",
        null_count=0, null_rate=0.0, distinct_count=3,
        sample_values=[100.0, 200.0, 150.0],
        min_value=100.0, max_value=200.0,
    )
    tbl = TableProfile(name="orders", row_count=row_count, columns=[col])
    return DatabaseProfile(source_path="mem", source_type="csv", tables=[tbl])


# ─────────────────────────── profiler tests ───────────────────────────────────

class TestSanitizeIdentifier:
    def test_normal(self):
        assert _sanitize_identifier("revenue") == "revenue"

    def test_spaces_replaced(self):
        assert _sanitize_identifier("order id") == "order_id"

    def test_leading_digit_prefixed(self):
        assert _sanitize_identifier("1column") == "col_1column"

    def test_special_chars(self):
        assert _sanitize_identifier("col (%)") == "col_____"


class TestInferType:
    def test_integers(self):
        assert _infer_sqlite_type(["1", "2", "3"]) == "INTEGER"

    def test_floats(self):
        assert _infer_sqlite_type(["1.0", "2.5", "3.14"]) == "REAL"

    def test_text(self):
        assert _infer_sqlite_type(["foo", "bar"]) == "TEXT"

    def test_empty(self):
        assert _infer_sqlite_type([]) == "TEXT"

    def test_mixed_int_text(self):
        assert _infer_sqlite_type(["1", "two"]) == "TEXT"


class TestDataSourceManager:
    def test_profile_has_table(self, manager: DataSourceManager):
        profile = manager.profile()
        assert any(t.name == "orders" for t in profile.tables)

    def test_profile_row_count(self, manager: DataSourceManager):
        profile = manager.profile()
        orders = next(t for t in profile.tables if t.name == "orders")
        assert orders.row_count == 3

    def test_profile_column_names(self, manager: DataSourceManager):
        profile = manager.profile()
        orders = next(t for t in profile.tables if t.name == "orders")
        col_names = {c.name for c in orders.columns}
        assert col_names == {"order_id", "revenue", "category"}

    def test_to_prompt_context_contains_table(self, manager: DataSourceManager):
        profile = manager.profile()
        ctx = profile.to_prompt_context()
        assert "orders" in ctx
        assert "revenue" in ctx

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DataSourceManager(tmp_path / "nonexistent.csv")

    def test_unsupported_extension_raises(self, tmp_path: Path):
        p = tmp_path / "data.xlsx"
        p.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            DataSourceManager(p)

    def test_empty_csv_raises(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            DataSourceManager(p)


# ─────────────────────────── SQL sandbox tests ────────────────────────────────

class TestSQLValidation:
    @pytest.mark.parametrize("sql,expect_ok", [
        ("SELECT * FROM orders", True),
        ("WITH cte AS (SELECT 1) SELECT * FROM cte", True),
        ("SELECT 1; SELECT 2", False),
        ("DROP TABLE orders", False),
        ("DELETE FROM orders WHERE 1=1", False),
        ("INSERT INTO orders VALUES (1)", False),
        ("UPDATE orders SET revenue=0", False),
        ("ATTACH DATABASE '/etc/passwd' AS x", False),
        ("", False),
    ])
    def test_validate(self, sql: str, expect_ok: bool):
        err = validate_sql_readonly(sql)
        assert (err is None) == expect_ok, f"Expected ok={expect_ok} for SQL: {sql!r}, got err={err!r}"

    def test_comment_stripping_does_not_bypass(self):
        sql = "SELECT 1; /*DROP TABLE orders*/ SELECT 2"
        assert validate_sql_readonly(sql) is not None, "Multi-statement with comment should be rejected"


class TestSQLSandbox:
    def test_basic_select(self, sql_sandbox: SQLSandbox):
        res = sql_sandbox.execute("q1", "SELECT SUM(revenue) AS total FROM orders")
        assert res.error is None
        assert res.row_count == 1
        total = res.rows[0]["total"]
        assert abs(total - 450.0) < 0.01

    def test_aggregation_by_category(self, sql_sandbox: SQLSandbox):
        res = sql_sandbox.execute("q2", "SELECT category, COUNT(*) as cnt FROM orders GROUP BY category ORDER BY cnt DESC")
        assert res.error is None
        assert res.row_count == 2

    def test_nonexistent_column(self, sql_sandbox: SQLSandbox):
        res = sql_sandbox.execute("q3", "SELECT fake_col FROM orders")
        assert res.error is not None
        assert "SQLiteError" in res.error

    def test_forbidden_write_blocked_by_validation(self, sql_sandbox: SQLSandbox):
        res = sql_sandbox.execute("q4", "DELETE FROM orders")
        assert res.error is not None
        assert "SQLSandboxError" in res.error

    def test_max_rows_truncation(self, manager: DataSourceManager):
        sb = SQLSandbox(manager, max_rows=1, timeout_seconds=5.0)
        res = sb.execute("q5", "SELECT * FROM orders")
        assert res.row_count == 1
        assert res.truncated is True

    def test_elapsed_ms_positive(self, sql_sandbox: SQLSandbox):
        res = sql_sandbox.execute("q6", "SELECT 1")
        assert res.elapsed_ms > 0


# ─────────────────────────── Python sandbox tests ─────────────────────────────

class TestPythonASTValidation:
    @pytest.mark.parametrize("code,expect_ok", [
        ("record_metric('x', 1 + 2)", True),
        ("import math; record_metric('pi', math.pi)", True),
        ("import os", False),
        ("import subprocess", False),
        ("eval('1+1')", False),
        ("exec('print(1)')", False),
        ("open('/etc/passwd')", False),
        ("import statistics; record_metric('m', statistics.mean([1,2,3]))", True),
    ])
    def test_ast_validate(self, code: str, expect_ok: bool):
        err = validate_python_ast(code)
        assert (err is None) == expect_ok, f"Expected ok={expect_ok} for code: {code!r}, got err={err!r}"


class TestPythonSandbox:
    def test_basic_metric(self, py_sandbox: PythonSandbox):
        code = "record_metric('total', sum(r['revenue'] for r in DATA['q1']))"
        data = {"q1": [{"revenue": 100.0}, {"revenue": 200.0}]}
        res = py_sandbox.execute("py1", code, data)
        assert res.error is None
        assert abs(res.metrics["total"] - 300.0) < 0.01

    def test_disallowed_import_blocked(self, py_sandbox: PythonSandbox):
        code = "import os; record_metric('x', 1)"
        res = py_sandbox.execute("py1", code, {})
        assert res.error is not None
        assert "PythonSandboxError" in res.error

    def test_exit_code_nonzero_on_runtime_error(self, py_sandbox: PythonSandbox):
        code = "raise ValueError('deliberate test error')"
        res = py_sandbox.execute("py1", code, {})
        assert res.exit_code != 0
        assert res.error is not None

    def test_elapsed_ms_positive(self, py_sandbox: PythonSandbox):
        res = py_sandbox.execute("py1", "record_metric('x', 42)", {})
        assert res.elapsed_ms > 0

    def test_timeout(self):
        sb = PythonSandbox(timeout_seconds=1.0)
        code = "import math; x = 0\nfor i in range(10**9): x += math.sqrt(i)"
        res = sb.execute("py1", code, {})
        assert res.error is not None
        assert "timed out" in res.error.lower() or res.exit_code != 0


# ─────────────────────────── guardrail tests ──────────────────────────────────

class TestExecutionGuardrails:
    def _sr(self, **kw) -> SQLResult:
        defaults = dict(
            query_id="q1", sql="SELECT 1",
            columns=["x"], rows=[{"x": 1}], row_count=1,
            truncated=False, elapsed_ms=1.0, error=None,
        )
        defaults.update(kw)
        return SQLResult(**defaults)

    def _pr(self, **kw) -> PythonResult:
        defaults = dict(
            script_id="py1", code="", stdout="",
            stderr="", metrics={}, elapsed_ms=1.0, exit_code=0, error=None,
        )
        defaults.update(kw)
        return PythonResult(**defaults)

    def test_schema_hallucination(self):
        profile = _make_profile()
        sr = self._sr(error="SQLiteError: no such column: fake_col")
        findings = run_execution_guardrails(profile, [sr], [])
        codes = [f.code for f in findings]
        assert "SCHEMA_HALLUCINATION" in codes

    def test_empty_result_warning(self):
        profile = _make_profile()
        sr = self._sr(rows=[], row_count=0)
        findings = run_execution_guardrails(profile, [sr], [])
        codes = [f.code for f in findings]
        assert "EMPTY_RESULT" in codes

    def test_truncated_result_warning(self):
        profile = _make_profile()
        sr = self._sr(truncated=True)
        findings = run_execution_guardrails(profile, [sr], [])
        codes = [f.code for f in findings]
        assert "TRUNCATED_RESULT" in codes

    def test_high_null_rate_warning(self):
        col = ColumnProfile(
            name="churn_score", inferred_type="REAL",
            null_count=25, null_rate=0.50, distinct_count=10,
            sample_values=[0.1, 0.5],
        )
        tbl = TableProfile(name="customers", row_count=50, columns=[col])
        profile = DatabaseProfile(source_path="mem", source_type="csv", tables=[tbl])
        findings = run_execution_guardrails(profile, [], [])
        assert any(f.code == "HIGH_NULL_RATE" for f in findings)

    def test_python_execution_error(self):
        profile = _make_profile()
        pr = self._pr(error="PythonRuntimeError (exit=1): ZeroDivisionError", exit_code=1)
        findings = run_execution_guardrails(profile, [], [pr])
        assert any(f.code == "PYTHON_EXECUTION_ERROR" for f in findings)

    def test_null_metric_warning(self):
        profile = _make_profile()
        pr = self._pr(metrics={"average_revenue": None})
        findings = run_execution_guardrails(profile, [], [pr])
        assert any(f.code == "NULL_OR_NAN_METRIC" for f in findings)

    def test_no_valid_sql_evidence(self):
        profile = _make_profile()
        sr = self._sr(error="SQLiteError: broken")
        findings = run_execution_guardrails(profile, [sr], [])
        assert any(f.code == "NO_VALID_SQL_EVIDENCE" for f in findings)

    def test_all_valid_no_errors(self):
        profile = _make_profile()
        sr = self._sr(rows=[{"total": 450.0}], row_count=1)
        pr = self._pr(metrics={"average_revenue": 150.0})
        findings = run_execution_guardrails(profile, [sr], [pr])
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert errors == []


class TestClaimVerification:
    def _make_sql_result(self, rows: list[dict], query_id: str = "q1") -> SQLResult:
        return SQLResult(
            query_id=query_id, sql="SELECT 1",
            columns=list(rows[0].keys()) if rows else [],
            rows=rows, row_count=len(rows),
            truncated=False, elapsed_ms=1.0, error=None,
        )

    def test_grounded_claim_passes(self):
        profile = _make_profile()
        sql = [self._make_sql_result([{"total_revenue": 450.0}])]
        _, findings = verify_claims(
            "Total revenue was **450.0**.", profile, sql, []
        )
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert errors == [], f"Should pass but got: {errors}"

    def test_ungrounded_claim_fails(self):
        profile = _make_profile()
        sql = [self._make_sql_result([{"total_revenue": 450.0}])]
        _, findings = verify_claims(
            "Total revenue was **9999.99**.", profile, sql, []
        )
        assert any(f.code == "UNGROUNDED_NUMERICAL_CLAIM" for f in findings)

    def test_year_as_label_ignored(self):
        profile = _make_profile()
        _, findings = verify_claims(
            "In 2024, revenue grew.", profile, [], []
        )
        # Year 2024 should not be treated as an ungrounded metric
        assert not any(f.code == "UNGROUNDED_NUMERICAL_CLAIM" for f in findings)

    def test_percentage_grounding(self):
        profile = _make_profile()
        pr = PythonResult(
            script_id="py1", code="", stdout="",
            stderr="", metrics={"growth_rate": 0.25},
            elapsed_ms=1.0, exit_code=0, error=None,
        )
        _, findings = verify_claims(
            "Revenue grew by 25%.", profile, [], [pr]
        )
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert errors == [], f"25% should match 0.25; got: {errors}"


# ─────────────────────────── JSON extraction tests ───────────────────────────

class TestExtractJSON:
    def test_plain_json(self):
        raw = '{"sql_queries": [], "python_scripts": []}'
        result = _extract_json(raw)
        assert result["sql_queries"] == []

    def test_fenced_json_block(self):
        raw = '```json\n{"sql_queries": [{"id": "q1", "sql": "SELECT 1"}], "python_scripts": []}\n```'
        result = _extract_json(raw)
        assert result["sql_queries"][0]["id"] == "q1"

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _extract_json("not valid json at all !!!")


# ─────────────────────────── end-to-end agent loop tests ─────────────────────

PLAN_RESPONSE_GOOD = json.dumps({
    "sql_queries": [{"id": "q1", "sql": "SELECT category, SUM(revenue) AS total FROM orders GROUP BY category"}],
    "python_scripts": [{"id": "py1", "code": "record_metric('electronics_total', sum(r['total'] for r in DATA['q1'] if r['category'] == 'Electronics'))"}],
})

SYNTHESIS_RESPONSE_GOOD = (
    "Electronics generated **450.0** in total revenue from q1. "
    "Electronics-only subtotal (py1): **450.0**."
)


class TestAgentEndToEnd:
    def test_happy_path(self, sample_csv_dir: Path):
        llm = ScriptedLLMClient([PLAN_RESPONSE_GOOD, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=1)
        report = agent.analyze(sample_csv_dir, "What is total revenue by category?")
        assert report.attempts >= 1
        assert len(report.sql_results) == 1
        assert report.sql_results[0].error is None

    def test_schema_hallucination_triggers_retry(self, sample_csv_dir: Path):
        bad_plan = json.dumps({
            "sql_queries": [{"id": "q1", "sql": "SELECT fake_col FROM orders"}],
            "python_scripts": [],
        })
        # Attempt 1: hallucinated SQL → retry
        # Attempt 2: good SQL + good synthesis
        llm = ScriptedLLMClient([bad_plan, PLAN_RESPONSE_GOOD, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=2)
        report = agent.analyze(sample_csv_dir, "Any question")
        assert report.attempts >= 2

    def test_ungrounded_claim_flagged(self, sample_csv_dir: Path):
        hallucinated_synthesis = "Total revenue was **99999.99** (hallucinated)."
        llm = ScriptedLLMClient([PLAN_RESPONSE_GOOD, hallucinated_synthesis])
        agent = AnalystAgent(llm=llm, max_retries=0)
        report = agent.analyze(sample_csv_dir, "What is revenue?")
        ungrounded = [f for f in report.findings if f.code == "UNGROUNDED_NUMERICAL_CLAIM"]
        assert len(ungrounded) > 0

    def test_report_markdown_contains_sections(self, sample_csv_dir: Path):
        llm = ScriptedLLMClient([PLAN_RESPONSE_GOOD, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=1)
        report = agent.analyze(sample_csv_dir, "Revenue?")
        md = report.to_markdown()
        assert "## Guardrail" in md
        assert "## Claim-to-Evidence" in md
        assert "## Reproducible Execution Trace" in md

    def test_forbidden_python_blocked_no_crash(self, sample_csv_dir: Path):
        bad_py_plan = json.dumps({
            "sql_queries": [{"id": "q1", "sql": "SELECT SUM(revenue) AS total FROM orders"}],
            "python_scripts": [{"id": "py1", "code": "import os; record_metric('x', os.getpid())"}],
        })
        llm = ScriptedLLMClient([bad_py_plan, SYNTHESIS_RESPONSE_GOOD])
        agent = AnalystAgent(llm=llm, max_retries=0)
        report = agent.analyze(sample_csv_dir, "Revenue?")
        py_errors = [f for f in report.findings if f.code == "PYTHON_EXECUTION_ERROR"]
        assert len(py_errors) > 0
