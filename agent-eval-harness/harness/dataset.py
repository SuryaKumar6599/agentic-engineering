"""
harness/dataset.py
──────────────────
Load / save golden datasets (JSONL or JSON) and generate synthetic
test cases via LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from harness.models import GoldenCase


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    """
    Load golden cases from a .jsonl or .json file.

    JSONL: one GoldenCase JSON object per line.
    JSON:  a top-level list of GoldenCase objects.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden dataset not found: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)

    return [GoldenCase.model_validate(r) for r in raw]


def save_golden_cases(cases: list[GoldenCase], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")


def iter_golden_cases(path: str | Path) -> Iterator[GoldenCase]:
    """Stream cases one at a time (useful for very large datasets)."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield GoldenCase.model_validate_json(line)


def filter_by_tag(cases: list[GoldenCase], tag: str) -> list[GoldenCase]:
    return [c for c in cases if tag in c.tags]
