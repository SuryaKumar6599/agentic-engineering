"""Core data models for the data-analyst agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class ColumnProfile:
    name: str
    inferred_type: str
    null_count: int
    null_rate: float
    distinct_count: int
    sample_values: list[Any]
    min_value: Any = None
    max_value: Any = None


@dataclass(slots=True)
class TableProfile:
    name: str
    row_count: int
    columns: list[ColumnProfile]
    primary_key_candidates: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatabaseProfile:
    source_path: str
    source_type: str  # "csv" | "sqlite"
    tables: list[TableProfile]

    def to_prompt_context(self) -> str:
        lines = [f"Database source: {self.source_path} ({self.source_type})"]
        for t in self.tables:
            lines.append(f"\nTable `{t.name}` ({t.row_count:,} rows):")
            if t.primary_key_candidates:
                lines.append(f"  Likely unique key(s): {', '.join(t.primary_key_candidates)}")
            for c in t.columns:
                range_str = ""
                if c.min_value is not None and c.max_value is not None:
                    range_str = f", range=[{c.min_value} .. {c.max_value}]"
                samples = ", ".join(repr(v) for v in c.sample_values[:4])
                lines.append(
                    f"  - {c.name} ({c.inferred_type}): "
                    f"nulls={c.null_count} ({c.null_rate:.1%}), "
                    f"distinct={c.distinct_count}{range_str}, samples=[{samples}]"
                )
        return "\n".join(lines)

    def all_table_names(self) -> set[str]:
        return {t.name for t in self.tables}

    def all_column_names(self) -> dict[str, set[str]]:
        return {t.name: {c.name for c in t.columns} for t in self.tables}


@dataclass(slots=True)
class SQLResult:
    query_id: str
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float
    error: str | None = None


@dataclass(slots=True)
class PythonResult:
    script_id: str
    code: str
    stdout: str
    stderr: str
    metrics: dict[str, Any]
    elapsed_ms: float
    exit_code: int
    error: str | None = None


@dataclass(slots=True)
class ExecutionArtifact:
    artifact_id: str
    kind: str  # "sql" | "python"
    sql_result: SQLResult | None = None
    python_result: PythonResult | None = None


@dataclass(slots=True)
class GuardrailFinding:
    code: str
    severity: Severity
    message: str
    artifact_id: str | None = None
    remediation: str | None = None


@dataclass(slots=True)
class ClaimVerification:
    claim_text: str
    extracted_numbers: list[float]
    matched_values: list[float]
    ungrounded_numbers: list[float]
    verified: bool


@dataclass(slots=True)
class AnalysisReport:
    question: str
    answer_markdown: str
    sql_results: list[SQLResult]
    python_results: list[PythonResult]
    findings: list[GuardrailFinding]
    claim_verifications: list[ClaimVerification]
    validation_passed: bool
    attempts: int
    profile: DatabaseProfile

    def to_markdown(self) -> str:
        status = "PASSED" if self.validation_passed else "FAILED"
        parts = [
            f"# Validated Data Analysis Report",
            f"",
            f"**Question:** {self.question}",
            f"**Validation Status:** `{status}` (attempts: {self.attempts})",
            f"",
            f"## Executive Summary & Findings",
            f"",
            self.answer_markdown.strip(),
            f"",
            f"## Guardrail & Statistical Sanity Checks",
            f"",
        ]
        if not self.findings:
            parts.append("- No guardrail warnings or errors triggered.")
        else:
            for f in self.findings:
                badge = f.severity.value.upper()
                loc = f" (`{f.artifact_id}`)" if f.artifact_id else ""
                parts.append(f"- **[{badge}] {f.code}**{loc}: {f.message}")
                if f.remediation:
                    parts.append(f"  - *Remediation:* {f.remediation}")

        parts.extend([
            "",
            "## Claim-to-Evidence Verification",
            "",
        ])
        if not self.claim_verifications:
            parts.append("- No quantitative claims extracted.")
        else:
            for cv in self.claim_verifications:
                mark = "PASS" if cv.verified else "FAIL"
                parts.append(
                    f"- **[{mark}]** {cv.claim_text} "
                    f"(matched={cv.matched_values}, ungrounded={cv.ungrounded_numbers})"
                )

        parts.extend(["", "## Reproducible Execution Trace", ""])
        for sr in self.sql_results:
            parts.append(f"### `{sr.query_id}` ({sr.row_count} rows, {sr.elapsed_ms:.1f} ms)")
            parts.append("```sql")
            parts.append(sr.sql.strip())
            parts.append("```")
            if sr.error:
                parts.append(f"- **Error:** `{sr.error}`")

        for pr in self.python_results:
            parts.append(f"### `{pr.script_id}` (exit={pr.exit_code}, {pr.elapsed_ms:.1f} ms)")
            parts.append("```python")
            parts.append(pr.code.strip())
            parts.append("```")
            if pr.metrics:
                parts.append(f"- **Computed Metrics:** `{pr.metrics}`")
            if pr.error:
                parts.append(f"- **Error:** `{pr.error}`")

        return "\n".join(parts)
