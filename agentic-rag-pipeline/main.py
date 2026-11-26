#!/usr/bin/env python3
"""Agentic RAG Pipeline — CLI entrypoint.

Usage:
    # Ingest documents
    python main.py --ingest data/sample_docs/

    # One-shot query
    python main.py --query "What tasks are involved in building a REST API?"

    # Interactive REPL
    python main.py
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from pipeline.agent import AgenticRAGAgent
from pipeline.ingestion import DocumentIngester
from schemas.models import AgentResponse

console = Console()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_response(resp: AgentResponse) -> None:
    """Pretty-print an AgentResponse to the terminal."""
    mode_badge = "[bold green]RAG[/]" if resp.mode == "rag" else "[bold yellow]ZERO-SHOT[/]"
    console.print()

    # Answer panel
    console.print(Panel(
        Markdown(resp.answer),
        title=f"Answer  {mode_badge}",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Task classification
    cat_color = {
        "RESEARCH": "blue",
        "CODE_GENERATION": "green",
        "DATA_ANALYSIS": "magenta",
        "PLANNING": "yellow",
        "GENERAL_QA": "white",
    }.get(resp.task.category.value, "white")

    console.print(Panel(
        f"[bold {cat_color}]{resp.task.category.value}[/]  "
        f"[dim]confidence {resp.task.confidence:.0%}[/]\n"
        f"[italic]{resp.task.rationale}[/]",
        title="Task Classification",
        border_style=cat_color,
        padding=(0, 2),
    ))

    # Workflow steps
    step_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    step_table.add_column("Step", style="bold cyan", width=6)
    step_table.add_column("Description")
    step_table.add_column("Complexity", width=10)
    step_table.add_column("Depends On", width=14)

    dep_map = resp.workflow.dependencies
    for step in resp.workflow.steps:
        deps = dep_map.get(str(step.id), [])
        step_table.add_row(
            str(step.id),
            step.description,
            step.estimated_complexity,
            ", ".join(deps) if deps else "-",
        )

    console.print(Panel(step_table, title="Workflow Plan", border_style="magenta", padding=(0, 1)))

    # Sources
    if resp.sources:
        console.print(Panel(
            "\n".join(f"[dim]• {s[:120]}...[/]" for s in resp.sources[:3]),
            title=f"Sources ({len(resp.sources)} chunks)",
            border_style="dim",
            padding=(0, 2),
        ))

    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_agent() -> AgenticRAGAgent:
    console.print("[dim]Initialising agent...[/]", end=" ")
    agent = AgenticRAGAgent()
    console.print("[green]ready[/]")
    return agent


def cmd_ingest(path: str) -> None:
    console.rule("[bold]Document Ingestion[/]")
    ingester = DocumentIngester()
    try:
        count = ingester.ingest_path(path)
        console.print(f"[green]✓[/] Ingested [bold]{count}[/] chunks from [cyan]{path}[/]")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)


def cmd_query(query: str, agent: AgenticRAGAgent) -> None:
    console.rule("[bold]Query[/]")
    if not agent._retriever.is_loaded:
        console.print(
            "[yellow]Warning:[/] No index found. Running in zero-shot mode. "
            "Run [cyan]--ingest <path>[/] first for RAG mode."
        )
    console.print(f"[dim]Query:[/] {query}\n")
    resp = agent.run(query)
    _render_response(resp)


def cmd_repl(agent: AgenticRAGAgent) -> None:
    console.rule("[bold]Interactive REPL[/] [dim](Ctrl+C or type 'exit' to quit)[/]")
    if not agent._retriever.is_loaded:
        console.print(
            "[yellow]Warning:[/] No index loaded. Zero-shot mode active.\n"
        )
    while True:
        try:
            query = console.input("[bold cyan]> [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/]")
            break
        if not query or query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Bye.[/]")
            break
        resp = agent.run(query)
        _render_response(resp)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Agentic RAG Pipeline for Context-Aware Task Detection",
    )
    parser.add_argument("--ingest", metavar="PATH", help="Ingest documents from file or directory")
    parser.add_argument("--query", metavar="TEXT", help="Run a single query and exit")
    args = parser.parse_args()

    if args.ingest:
        cmd_ingest(args.ingest)
        if not args.query:
            return

    agent = _build_agent()

    if args.query:
        cmd_query(args.query, agent)
    else:
        cmd_repl(agent)


if __name__ == "__main__":
    main()
