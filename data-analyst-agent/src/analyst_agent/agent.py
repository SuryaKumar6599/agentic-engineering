"""Two-phase Sandboxed Data-Analyst Agent with Guardrail Self-Correction Loop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from analyst_agent.guardrails import run_execution_guardrails, verify_claims
from analyst_agent.llm import LLMClient
from analyst_agent.models import (
    AnalysisReport,
    DatabaseProfile,
    GuardrailFinding,
    PythonResult,
    Severity,
    SQLResult,
)
from analyst_agent.profiler import DataSourceManager
from analyst_agent.python_sandbox import PythonSandbox
from analyst_agent.sql_sandbox import SQLSandbox

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

PLAN_SYSTEM_PROMPT = """\
You are a rigorous Data Analyst Agent working with a SQLite database.
Given a database schema profile and a business question, generate:
1. Read-only SQLite SELECT queries (`sql_queries`) to extract the exact aggregations needed.
2. Optional Python analysis scripts (`python_scripts`) to compute derived statistics.

CRITICAL RULES — violating any of these will cause the query to fail:
- Use ONLY exact table and column names listed in the DatabaseProfile below. Do NOT invent tables.
- NEVER JOIN to a table unless it appears in the DatabaseProfile.
- NEVER import sqlite3, os, subprocess, sys, socket, urllib, requests in Python scripts.
- In Python scripts, SQL result rows are pre-loaded for you in DATA["<query_id>"] as a list of dicts.
  Example: `for row in DATA["q1"]: record_metric("total", row["revenue"])`
- Do NOT call sqlite3.connect() or open any file in Python. The data is already in DATA.
- Call `record_metric("name", value)` for every derived number you plan to cite in the report. When looping over rows, include the group/category name in the key (e.g., `record_metric(f"{row['region']}_revenue", row['total'])`) so values don't overwrite each other.
- Guard every division with NULLIF(denominator, 0) in SQL.
- Always pre-aggregate child tables in a CTE before JOIN to avoid row fanout, and ALWAYS qualify every column with its table or CTE alias (e.g., `c.customer_id`, `o.total_revenue`) whenever using JOINs.
- To count orders after pre-aggregating in a CTE, use `SUM(o.order_count)` rather than `COUNT(...)`.

Respond ONLY with valid JSON (no markdown fences, no extra text):
{"sql_queries": [{"id": "q1", "sql": "SELECT ..."}], "python_scripts": [{"id": "py1", "code": "..."}]}
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a rigorous Data Analyst Agent writing an executive summary from verified execution outputs.
Rules:
- Cite ONLY exact numbers present in the SQL results or Python `metrics` below.
- Reference artifact IDs (e.g. `q1`, `py1`) for every key figure.
- Explicitly mention any data quality caveats (high null rates, small sample sizes, empty slices).
- Do NOT invent, extrapolate, or round numbers inconsistently.
Respond with concise, clear Markdown.
"""


def _extract_json(text: str) -> dict[str, Any]:
    match = _JSON_BLOCK_RE.search(text)
    raw = match.group(1) if match else text.strip()
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        # Fallback: extract outermost {...} object if surrounded by extra text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1], strict=False)
        raise


def _format_execution_context(
    sql_results: list[SQLResult],
    python_results: list[PythonResult],
) -> str:
    parts: list[str] = []
    for sr in sql_results:
        parts.append(f"SQL `{sr.query_id}`:")
        if sr.error:
            parts.append(f"  ERROR: {sr.error}")
        else:
            parts.append(f"  rows ({sr.row_count}, truncated={sr.truncated}): {json.dumps(sr.rows[:50], default=str)}")
    for pr in python_results:
        parts.append(f"Python `{pr.script_id}`:")
        if pr.error:
            parts.append(f"  ERROR: {pr.error}")
        else:
            parts.append(f"  metrics: {json.dumps(pr.metrics, default=str)}")
            if pr.stdout:
                parts.append(f"  stdout: {pr.stdout[:500]}")
    return "\n".join(parts)


class AnalystAgent:
    """Orchestrates schema profiling, sandboxed SQL/Python execution, and validated reporting."""

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 2,
        max_sql_rows: int = 500,
        sql_timeout: float = 5.0,
        python_timeout: float = 10.0,
    ) -> None:
        self.llm = llm
        self.max_retries = max_retries
        self.max_sql_rows = max_sql_rows
        self.sql_timeout = sql_timeout
        self.python_timeout = python_timeout

    def analyze(self, source_path: str | Path, question: str) -> AnalysisReport:
        manager = DataSourceManager(source_path)
        try:
            profile = manager.profile()
            sql_sandbox = SQLSandbox(
                manager,
                max_rows=self.max_sql_rows,
                timeout_seconds=self.sql_timeout,
            )
            py_sandbox = PythonSandbox(timeout_seconds=self.python_timeout)
            return self._run_loop(profile, sql_sandbox, py_sandbox, question)
        finally:
            manager.close()

    def _run_loop(
        self,
        profile: DatabaseProfile,
        sql_sandbox: SQLSandbox,
        py_sandbox: PythonSandbox,
        question: str,
    ) -> AnalysisReport:
        feedback = ""
        sql_results: list[SQLResult] = []
        python_results: list[PythonResult] = []
        findings: list[GuardrailFinding] = []
        claim_verifications = []
        answer_markdown = ""

        attempts = 0
        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            plan_prompt = (
                f"Business Question: {question}\n\n"
                f"{profile.to_prompt_context()}"
            )
            if feedback:
                plan_prompt += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION. FIX THESE ISSUES:\n{feedback}"

            plan_raw = self.llm.generate(PLAN_SYSTEM_PROMPT, plan_prompt)
            try:
                plan = _extract_json(plan_raw)
                if not isinstance(plan, dict):
                    raise ValueError("Plan JSON must be an object with 'sql_queries' and 'python_scripts'.")
            except (json.JSONDecodeError, ValueError) as exc:
                findings = [
                    GuardrailFinding(
                        code="INVALID_PLAN_JSON",
                        severity=Severity.ERROR,
                        message=f"Failed to parse LLM plan as valid JSON: {exc}",
                        remediation="Return ONLY a single valid JSON object; escape all double quotes inside Python code strings.",
                    )
                ]
                if attempt <= self.max_retries:
                    feedback = (
                        f"- [INVALID_PLAN_JSON] {exc} "
                        "(Remediation: Return ONLY valid JSON; escape quotes and newlines inside string values.)"
                    )
                    continue
                return AnalysisReport(
                    question=question,
                    answer_markdown="Agent failed to produce a valid JSON execution plan.",
                    sql_results=[],
                    python_results=[],
                    findings=findings,
                    claim_verifications=[],
                    validation_passed=False,
                    attempts=attempts,
                    profile=profile,
                )

            sql_results = [
                sql_sandbox.execute(q["id"], q["sql"])
                for q in plan.get("sql_queries", [])
            ]

            data_payload = {
                sr.query_id: sr.rows
                for sr in sql_results
                if sr.error is None
            }
            # Skip Python execution if all SQL queries failed (prevents cascading KeyError noise)
            if sql_results and not data_payload:
                python_results = []
            else:
                python_results = [
                    py_sandbox.execute(p["id"], p["code"], data_payload)
                    for p in plan.get("python_scripts", [])
                ]

            exec_findings = run_execution_guardrails(profile, sql_results, python_results)
            exec_errors = [f for f in exec_findings if f.severity == Severity.ERROR]

            if exec_errors and attempt <= self.max_retries:
                col_hints = "\n".join(
                    f"  {sr.query_id} produced columns: {sr.columns} — sample row: {sr.rows[0] if sr.rows else 'no rows'}"
                    for sr in sql_results if sr.error is None and sr.columns
                )
                feedback = "\n".join(
                    f"- [{f.code}] {f.message} (Remediation: {f.remediation or 'fix'})"
                    for f in exec_errors
                )
                if col_hints:
                    feedback += f"\n\nActual SQL output columns (use THESE exact names in Python DATA access):\n{col_hints}"
                continue

            synth_prompt = (
                f"Business Question: {question}\n\n"
                f"Execution Results:\n{_format_execution_context(sql_results, python_results)}\n\n"
                f"Available column names per query:\n"
                + "\n".join(
                    f"  {sr.query_id}: {sr.columns}"
                    for sr in sql_results if sr.error is None
                )
            )
            if exec_findings:
                notes = "\n".join(f"- [{f.severity.value}] {f.code}: {f.message}" for f in exec_findings)
                synth_prompt += f"\n\nData Quality Warnings:\n{notes}"

            answer_markdown = self.llm.generate(SYNTHESIS_SYSTEM_PROMPT, synth_prompt)
            claim_verifications, claim_findings = verify_claims(
                answer_markdown, profile, sql_results, python_results
            )
            findings = exec_findings + claim_findings
            all_errors = [f for f in findings if f.severity == Severity.ERROR]

            if not all_errors:
                return AnalysisReport(
                    question=question,
                    answer_markdown=answer_markdown,
                    sql_results=sql_results,
                    python_results=python_results,
                    findings=findings,
                    claim_verifications=claim_verifications,
                    validation_passed=True,
                    attempts=attempts,
                    profile=profile,
                )

            if attempt <= self.max_retries:
                col_hints = "\n".join(
                    f"  {sr.query_id} columns: {sr.columns}, sample: {sr.rows[0] if sr.rows else 'no rows'}"
                    for sr in sql_results if sr.error is None and sr.columns
                )
                feedback = "\n".join(
                    f"- [{f.code}] {f.message} (Remediation: {f.remediation or 'fix'})"
                    for f in all_errors
                )
                if col_hints:
                    feedback += f"\n\nActual SQL columns (use these exact names in Python):\n{col_hints}"

        has_errors = any(f.severity == Severity.ERROR for f in findings)
        return AnalysisReport(
            question=question,
            answer_markdown=answer_markdown,
            sql_results=sql_results,
            python_results=python_results,
            findings=findings,
            claim_verifications=claim_verifications,
            validation_passed=not has_errors,
            attempts=attempts,
            profile=profile,
        )
