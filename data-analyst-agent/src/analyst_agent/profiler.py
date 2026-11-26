"""Data ingestion (CSV / SQLite) and schema profiling."""

from __future__ import annotations

import csv
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from analyst_agent.models import ColumnProfile, DatabaseProfile, TableProfile

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def _infer_sqlite_type(values: list[str]) -> str:
    non_empty = [v.strip() for v in values if v is not None and v.strip() != ""]
    if not non_empty:
        return "TEXT"
    if all(_INT_RE.match(v) for v in non_empty):
        return "INTEGER"
    if all(_FLOAT_RE.match(v) for v in non_empty):
        return "REAL"
    return "TEXT"


def _coerce_value(val: str | None, col_type: str) -> Any:
    if val is None or val.strip() == "":
        return None
    s = val.strip()
    if col_type == "INTEGER":
        try:
            return int(s)
        except ValueError:
            return s
    if col_type == "REAL":
        try:
            return float(s)
        except ValueError:
            return s
    return s


class DataSourceManager:
    """Manages CSV-to-SQLite ingestion and read-only profiling."""

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path).resolve()
        if not self.source_path.exists():
            raise FileNotFoundError(f"Data source not found: {self.source_path}")
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.db_path: Path
        self.source_type: str

        if self.source_path.is_dir():
            self.source_type = "csv"
            self.db_path = self._ingest_csv_directory(self.source_path)
        elif self.source_path.suffix.lower() == ".csv":
            self.source_type = "csv"
            self.db_path = self._ingest_csv_files([self.source_path])
        elif self.source_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            self.source_type = "sqlite"
            self.db_path = self.source_path
        else:
            raise ValueError(
                f"Unsupported data source '{self.source_path.suffix}'. "
                "Provide a .csv file, directory of .csv files, or .db/.sqlite file."
            )

    def _ingest_csv_directory(self, directory: Path) -> Path:
        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No .csv files found in {directory}")
        return self._ingest_csv_files(csv_files)

    def _ingest_csv_files(self, csv_files: list[Path]) -> Path:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="analyst_agent_db_")
        db_path = Path(self._tmpdir.name) / "ingested.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            for csv_file in csv_files:
                table_name = _sanitize_identifier(csv_file.stem)
                with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.reader(f)
                    try:
                        raw_headers = next(reader)
                    except StopIteration:
                        raise ValueError(f"CSV file is empty: {csv_file}")
                    headers = [_sanitize_identifier(h) for h in raw_headers]
                    rows = list(reader)

                col_types: list[str] = []
                for idx in range(len(headers)):
                    sample_col = [r[idx] if idx < len(r) else "" for r in rows[:1000]]
                    col_types.append(_infer_sqlite_type(sample_col))

                col_defs = ", ".join(
                    f'"{h}" {t}' for h, t in zip(headers, col_types, strict=True)
                )
                conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

                placeholders = ", ".join("?" for _ in headers)
                coerced_rows = [
                    tuple(
                        _coerce_value(r[i] if i < len(r) else None, col_types[i])
                        for i in range(len(headers))
                    )
                    for r in rows
                ]
                if coerced_rows:
                    conn.executemany(
                        f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                        coerced_rows,
                    )
            conn.commit()
        finally:
            conn.close()
        return db_path

    def open_readonly_connection(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn

    def profile(self) -> DatabaseProfile:
        conn = self.open_readonly_connection()
        try:
            tables_raw = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            table_profiles: list[TableProfile] = []
            for trow in tables_raw:
                tname = trow["name"]
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
                cols_info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                col_profiles: list[ColumnProfile] = []
                pk_candidates: list[str] = []

                for cinfo in cols_info:
                    cname = cinfo["name"]
                    ctype = cinfo["type"] or "TEXT"
                    stats = conn.execute(
                        f'SELECT '
                        f'SUM(CASE WHEN "{cname}" IS NULL OR "{cname}" = \'\' THEN 1 ELSE 0 END) AS nulls, '
                        f'COUNT(DISTINCT "{cname}") AS distinct_cnt, '
                        f'MIN("{cname}") AS min_v, '
                        f'MAX("{cname}") AS max_v '
                        f'FROM "{tname}"'
                    ).fetchone()
                    null_count = stats["nulls"] or 0
                    distinct_count = stats["distinct_cnt"] or 0
                    null_rate = (null_count / row_count) if row_count > 0 else 0.0

                    samples_rows = conn.execute(
                        f'SELECT DISTINCT "{cname}" FROM "{tname}" '
                        f'WHERE "{cname}" IS NOT NULL AND "{cname}" != \'\' LIMIT 5'
                    ).fetchall()
                    sample_values = [r[0] for r in samples_rows]

                    if row_count > 0 and null_count == 0 and distinct_count == row_count:
                        pk_candidates.append(cname)

                    min_val = stats["min_v"] if ctype.upper() in {"INTEGER", "REAL", "NUMERIC"} else None
                    max_val = stats["max_v"] if ctype.upper() in {"INTEGER", "REAL", "NUMERIC"} else None

                    col_profiles.append(
                        ColumnProfile(
                            name=cname,
                            inferred_type=ctype,
                            null_count=null_count,
                            null_rate=null_rate,
                            distinct_count=distinct_count,
                            sample_values=sample_values,
                            min_value=min_val,
                            max_value=max_val,
                        )
                    )
                table_profiles.append(
                    TableProfile(
                        name=tname,
                        row_count=row_count,
                        columns=col_profiles,
                        primary_key_candidates=pk_candidates,
                    )
                )
            return DatabaseProfile(
                source_path=str(self.source_path),
                source_type=self.source_type,
                tables=table_profiles,
            )
        finally:
            conn.close()

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
