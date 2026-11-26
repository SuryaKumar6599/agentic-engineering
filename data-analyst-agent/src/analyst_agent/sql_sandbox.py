"""Sandboxed read-only SQL executor with static + runtime authorizer guards."""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

from analyst_agent.models import SQLResult
from analyst_agent.profiler import DataSourceManager

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"VACUUM|REINDEX|LOAD_EXTENSION|TRIGGER|INDEX)\b",
    re.IGNORECASE,
)


def _strip_sql_comments_and_strings(sql: str) -> str:
    # Remove single-line comments, block comments, and string literals for keyword scanning.
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    no_strings = re.sub(r"'(?:''|[^'])*'", "''", no_line)
    return no_strings.strip()


def validate_sql_readonly(sql: str) -> str | None:
    """Return an error message if SQL is unsafe or non-SELECT; otherwise None."""
    cleaned = _strip_sql_comments_and_strings(sql)
    if not cleaned:
        return "Empty SQL query."

    # Allow optional trailing semicolon, reject multiple statements.
    without_trailing = cleaned.rstrip(";").strip()
    if ";" in without_trailing:
        return "Multiple SQL statements are not allowed in a single query."

    first_token = without_trailing.split(None, 1)[0].upper()
    if first_token not in {"SELECT", "WITH"}:
        return f"Only SELECT or WITH (CTE) queries are permitted; got '{first_token}'."

    match = _FORBIDDEN_KEYWORDS.search(without_trailing)
    if match:
        return f"Forbidden SQL keyword detected: '{match.group(1).upper()}'."

    return None


def _readonly_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    trigger_name: str | None,
) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        allowed.add(sqlite3.SQLITE_RECURSIVE)
    if action in allowed:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLSandbox:
    """Executes validated read-only SQL against a DataSourceManager."""

    def __init__(
        self,
        data_source: DataSourceManager,
        max_rows: int = 500,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.data_source = data_source
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def execute(self, query_id: str, sql: str) -> SQLResult:
        start = time.perf_counter()
        validation_err = validate_sql_readonly(sql)
        if validation_err:
            return SQLResult(
                query_id=query_id,
                sql=sql,
                columns=[],
                rows=[],
                row_count=0,
                truncated=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"SQLSandboxError: {validation_err}",
            )

        conn = self.data_source.open_readonly_connection()
        deadline = time.monotonic() + self.timeout_seconds

        def _progress() -> int:
            if time.monotonic() > deadline:
                return 1  # abort query
            return 0

        try:
            conn.set_progress_handler(_progress, 10_000)
            conn.set_authorizer(_readonly_authorizer)
            cursor = conn.execute(sql.strip().rstrip(";"))
            columns = [desc[0] for desc in (cursor.description or [])]
            fetched = cursor.fetchmany(self.max_rows + 1)
            truncated = len(fetched) > self.max_rows
            if truncated:
                fetched = fetched[: self.max_rows]
            rows: list[dict[str, Any]] = [
                {col: row[i] for i, col in enumerate(columns)}
                for row in fetched
            ]
            return SQLResult(
                query_id=query_id,
                sql=sql,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=None,
            )
        except sqlite3.OperationalError as exc:
            msg = str(exc)
            if "interrupted" in msg.lower():
                msg = f"Query exceeded {self.timeout_seconds}s timeout."
            return SQLResult(
                query_id=query_id,
                sql=sql,
                columns=[],
                rows=[],
                row_count=0,
                truncated=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"SQLiteError: {msg}",
            )
        except Exception as exc:
            return SQLResult(
                query_id=query_id,
                sql=sql,
                columns=[],
                rows=[],
                row_count=0,
                truncated=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"SQLiteError: {exc}",
            )
        finally:
            conn.set_authorizer(None)
            conn.set_progress_handler(None, 0)
            conn.close()
