"""Data-science guardrails and claim-to-evidence verification."""

from __future__ import annotations

import math
import re
from typing import Any

from analyst_agent.models import (
    ClaimVerification,
    DatabaseProfile,
    GuardrailFinding,
    PythonResult,
    Severity,
    SQLResult,
)

# Matches numbers like $1,234.56, 42.5%, -12.3, 1000
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([+-]?(?:\$)?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|[+-]?(?:\$)?\d+(?:\.\d+)?%?)"
    r"(?![A-Za-z0-9_])"
)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+\.|[-*•])\s+")
_QUERY_REF_RE = re.compile(r"\b(?:q|py)\d+\b", re.IGNORECASE)
_JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+[\"`\[]?([A-Za-z_][A-Za-z0-9_]*)[\"`\]]?", re.IGNORECASE)


def collect_evidence_numbers(
    profile: DatabaseProfile,
    sql_results: list[SQLResult],
    python_results: list[PythonResult],
) -> set[float]:
    """Gather all numeric values observed in profile, SQL outputs, and Python metrics."""
    nums: set[float] = set()

    def _add_num(val: Any) -> None:
        if isinstance(val, bool) or val is None:
            return
        if isinstance(val, (int, float)):
            if math.isfinite(val):
                f = float(val)
                nums.add(f)
                # Also allow percentage representation of ratios in [0, 1] and vice versa
                if -1.0 <= f <= 1.0:
                    nums.add(f * 100.0)
                if 0.0 <= abs(f) <= 100.0:
                    nums.add(f / 100.0)

    for t in profile.tables:
        _add_num(t.row_count)
        for c in t.columns:
            _add_num(c.null_count)
            _add_num(c.null_rate)
            _add_num(c.distinct_count)
            _add_num(c.min_value)
            _add_num(c.max_value)

    for sr in sql_results:
        _add_num(sr.row_count)
        for row in sr.rows:
            for val in row.values():
                _add_num(val)

    for pr in python_results:
        for val in pr.metrics.values():
            if isinstance(val, dict):
                for sub_v in val.values():
                    _add_num(sub_v)
            elif isinstance(val, list):
                for sub_v in val:
                    _add_num(sub_v)
            else:
                _add_num(val)

    return nums


def _matches_evidence(val: float, evidence: set[float], rel_tol: float = 0.015, abs_tol: float = 0.05) -> bool:
    for ev in evidence:
        if math.isclose(val, ev, rel_tol=rel_tol, abs_tol=abs_tol):
            return True
    return False


def verify_claims(
    answer_markdown: str,
    profile: DatabaseProfile,
    sql_results: list[SQLResult],
    python_results: list[PythonResult],
) -> tuple[list[ClaimVerification], list[GuardrailFinding]]:
    """Verify every quantitative claim in the narrative against execution evidence."""
    evidence = collect_evidence_numbers(profile, sql_results, python_results)
    verifications: list[ClaimVerification] = []
    findings: list[GuardrailFinding] = []

    for raw_line in answer_markdown.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue

        # Strip markdown list numbering ("1. ") and query/script IDs ("q1", "py1")
        cleaned = _LIST_PREFIX_RE.sub("", line)
        cleaned = _QUERY_REF_RE.sub("", cleaned)

        matches = _NUMBER_RE.findall(cleaned)
        if not matches:
            continue

        extracted: list[float] = []
        matched: list[float] = []
        ungrounded: list[float] = []

        for token in matches:
            raw_num = token.replace("$", "").replace(",", "").rstrip("%")
            try:
                val = float(raw_num)
            except ValueError:
                continue
            # Ignore calendar years 1990-2035 if they appear as dates/labels
            if val.is_integer() and 1990 <= int(val) <= 2035 and "%" not in token and "$" not in token:
                continue
            extracted.append(val)
            if _matches_evidence(val, evidence):
                matched.append(val)
            else:
                ungrounded.append(val)

        if not extracted:
            continue

        verified = len(ungrounded) == 0
        verifications.append(
            ClaimVerification(
                claim_text=line,
                extracted_numbers=extracted,
                matched_values=matched,
                ungrounded_numbers=ungrounded,
                verified=verified,
            )
        )
        if not verified:
            findings.append(
                GuardrailFinding(
                    code="UNGROUNDED_NUMERICAL_CLAIM",
                    severity=Severity.ERROR,
                    message=(
                        f"Narrative contains number(s) {ungrounded} not found in SQL/Python "
                        f"execution outputs: '{line[:120]}'"
                    ),
                    remediation=(
                        "Only report exact numbers computed by SQL or Python artifacts, "
                        "or compute the derived figure explicitly in Python via record_metric()."
                    ),
                )
            )

    return verifications, findings


def run_execution_guardrails(
    profile: DatabaseProfile,
    sql_results: list[SQLResult],
    python_results: list[PythonResult],
) -> list[GuardrailFinding]:
    """Run data-science and execution sanity checks on SQL and Python results."""
    findings: list[GuardrailFinding] = []
    table_row_counts = {t.name.lower(): t.row_count for t in profile.tables}

    # Check high null-rate columns in profile
    for t in profile.tables:
        for c in t.columns:
            if c.null_rate >= 0.20:
                findings.append(
                    GuardrailFinding(
                        code="HIGH_NULL_RATE",
                        severity=Severity.WARNING,
                        message=(
                            f"Column `{t.name}.{c.name}` has {c.null_rate:.1%} missing values "
                            f"({c.null_count}/{t.row_count})."
                        ),
                        remediation="Explicitly filter or document NULL handling in aggregations and denominators.",
                    )
                )

    successful_sql = 0
    for sr in sql_results:
        if sr.error:
            err_lower = sr.error.lower()
            if "no such column" in err_lower or "no such table" in err_lower:
                findings.append(
                    GuardrailFinding(
                        code="SCHEMA_HALLUCINATION",
                        severity=Severity.ERROR,
                        artifact_id=sr.query_id,
                        message=f"SQL referenced a nonexistent table or column: {sr.error}",
                        remediation="Use only exact table and column names from the DatabaseProfile.",
                    )
                )
            else:
                remediation = "Fix SQL syntax or sandbox violation and re-run."
                if "ambiguous column name" in err_lower:
                    remediation = (
                        "Qualify every column in SELECT/WHERE/GROUP BY with its table or CTE alias "
                        "(e.g., `c.customer_id` instead of `customer_id`) when using JOINs."
                    )
                findings.append(
                    GuardrailFinding(
                        code="SQL_EXECUTION_ERROR",
                        severity=Severity.ERROR,
                        artifact_id=sr.query_id,
                        message=sr.error,
                        remediation=remediation,
                    )
                )
            continue

        successful_sql += 1
        if sr.row_count == 0:
            findings.append(
                GuardrailFinding(
                    code="EMPTY_RESULT",
                    severity=Severity.WARNING,
                    artifact_id=sr.query_id,
                    message=f"Query `{sr.query_id}` returned 0 rows.",
                    remediation="Verify filter predicates, case sensitivity, and date formats.",
                )
            )

        if sr.truncated:
            findings.append(
                GuardrailFinding(
                    code="TRUNCATED_RESULT",
                    severity=Severity.WARNING,
                    artifact_id=sr.query_id,
                    message=f"Query `{sr.query_id}` was truncated at {sr.row_count} rows.",
                    remediation="Aggregate in SQL (GROUP BY / SUM / COUNT) instead of fetching raw rows.",
                )
            )

        # Join fanout detection: if JOIN is used and returned rows > max referenced table row count
        if _JOIN_RE.search(sr.sql):
            ref_tables = [m.lower() for m in _TABLE_REF_RE.findall(sr.sql)]
            max_base_rows = max((table_row_counts.get(t, 0) for t in ref_tables), default=0)
            if max_base_rows > 0 and sr.row_count > max_base_rows:
                findings.append(
                    GuardrailFinding(
                        code="JOIN_FANOUT_EXPLOSION",
                        severity=Severity.ERROR,
                        artifact_id=sr.query_id,
                        message=(
                            f"JOIN query `{sr.query_id}` produced {sr.row_count} rows, exceeding "
                            f"the largest referenced table ({max_base_rows} rows). Likely many-to-many fanout."
                        ),
                        remediation="Pre-aggregate child tables in a CTE before joining, or use DISTINCT keys.",
                    )
                )

        # Check for None/NaN in numeric-looking result columns
        for row in sr.rows:
            for col, val in row.items():
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    findings.append(
                        GuardrailFinding(
                            code="NON_FINITE_VALUE",
                            severity=Severity.ERROR,
                            artifact_id=sr.query_id,
                            message=f"Non-finite value ({val}) in column `{col}`.",
                            remediation="Guard divisions with NULLIF(denom, 0).",
                        )
                    )
                    break

    if sql_results and successful_sql == 0:
        findings.append(
            GuardrailFinding(
                code="NO_VALID_SQL_EVIDENCE",
                severity=Severity.ERROR,
                message="All SQL queries failed; report cannot be grounded.",
            )
        )

    for pr in python_results:
        if pr.error:
            findings.append(
                GuardrailFinding(
                    code="PYTHON_EXECUTION_ERROR",
                    severity=Severity.ERROR,
                    artifact_id=pr.script_id,
                    message=pr.error,
                    remediation="Fix Python syntax/runtime error or remove disallowed imports.",
                )
            )
            continue

        for k, v in pr.metrics.items():
            if v is None:
                findings.append(
                    GuardrailFinding(
                        code="NULL_OR_NAN_METRIC",
                        severity=Severity.WARNING,
                        artifact_id=pr.script_id,
                        message=f"Python metric `{k}` evaluated to None/NaN.",
                        remediation="Check for empty slices or zero denominators before computing metric.",
                    )
                )
            if ("sample_size" in k.lower() or k.lower() in {"n", "count"}) and isinstance(v, (int, float)):
                if 0 < v < 30:
                    findings.append(
                        GuardrailFinding(
                            code="SMALL_SAMPLE_SIZE",
                            severity=Severity.WARNING,
                            artifact_id=pr.script_id,
                            message=f"Small sample size detected (`{k}={v}` < 30); interpret rates/effects cautiously.",
                        )
                    )

    return findings
