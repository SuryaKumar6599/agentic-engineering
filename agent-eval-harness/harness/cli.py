"""
harness/cli.py
──────────────
CLI entry-point:  `eval run`, `eval report`, `eval budget`.

  eval run --dataset evals/fixtures/golden/qa.jsonl
  eval report --results results/latest.jsonl
  eval budget --results results/latest.jsonl
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from harness.config import settings
from harness.dataset import load_golden_cases
from harness.models import EvalResult

app = typer.Typer(help="Agent eval harness CLI")
console = Console()


@app.command("run")
def run_cmd(
    dataset: Path = typer.Option(..., help="Path to golden .jsonl dataset"),
    agent: str = typer.Option("example", help="Agent module name in agents/"),
    out: Path = typer.Option(Path("results/latest.jsonl"), help="Output results file"),
):
    """Run the full eval suite against a dataset."""
    import importlib

    agent_mod = importlib.import_module(f"agents.{agent}.agent")
    agent_fn = agent_mod.run

    from harness.metrics.task_success import answer_relevancy, task_completion_geval
    from harness.runner import run_eval_suite

    cases = load_golden_cases(dataset)
    endpoint = settings.local_model_base_url if settings.is_local else "OpenAI cloud"
    console.print(
        f"[bold cyan]Running {len(cases)} cases with agent '{agent}' "
        f"(model: {settings.resolved_model_name} @ {endpoint})…[/bold cyan]"
    )

    results = asyncio.run(
        run_eval_suite(
            agent_fn=agent_fn,
            cases=cases,
            metric_factories=[answer_relevancy, task_completion_geval],
        )
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    console.print(f"[green]Results written to {out}[/green]")

    # Exit non-zero if any case failed (for CI)
    failed = [r for r in results if not r.passed]
    if failed:
        raise typer.Exit(code=1)


@app.command("budget")
def budget_cmd(
    results: Path = typer.Option(Path("results/latest.jsonl"), help="Results file"),
):
    """Check if total cost stays within budget."""
    total = sum(
        EvalResult.model_validate_json(line).total_cost_usd
        for line in results.read_text().splitlines()
        if line.strip()
    )
    budget = settings.cost_budget_usd
    console.print(f"Total cost: ${total:.4f}  Budget: ${budget:.2f}")
    if total > budget:
        console.print("[bold red]BUDGET EXCEEDED[/bold red]")
        raise typer.Exit(code=1)
    console.print("[green]Within budget ✓[/green]")


@app.command("report")
def report_cmd(
    results: Path = typer.Option(Path("results/latest.jsonl"), help="Results file"),
    fmt: str = typer.Option("table", help="Output format: table | json | markdown"),
):
    """Pretty-print an eval results file."""
    rows = [
        EvalResult.model_validate_json(line)
        for line in results.read_text().splitlines()
        if line.strip()
    ]

    if fmt == "json":
        console.print(json.dumps([r.summary for r in rows], indent=2))
        return

    from rich.table import Table
    table = Table(title="Eval Results")
    table.add_column("Task")
    table.add_column("Model", style="magenta")
    table.add_column("Pass")
    table.add_column("Cost")
    table.add_column("Latency ms")
    table.add_column("Failures")

    for r in rows:
        fails = [m.name for m in r.metrics if m.status in ("fail", "error")]
        table.add_row(
            r.task_id,
            r.model or "—",
            "✅" if r.passed else "❌",
            f"${r.total_cost_usd:.4f}",
            f"{r.latency_ms:.0f}",
            ", ".join(fails) or "—",
        )
    console.print(table)


if __name__ == "__main__":
    app()
