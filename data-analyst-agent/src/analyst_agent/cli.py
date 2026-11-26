"""Command-line interface for the sandboxed data-analyst agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyst_agent.agent import AnalystAgent
from analyst_agent.llm import auto_select_client


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="analyst-agent",
        description="Run a sandboxed SQL & Python data-analyst agent with guardrails.",
    )
    parser.add_argument("--source", required=True, help="Path to CSV file, CSV directory, or SQLite DB.")
    parser.add_argument("--question", required=True, help="Business question to answer.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "ollama"], default=None)
    parser.add_argument("--model", default=None, help="Override model name (e.g. qwen2.5:7b, qwen3.6:27b).")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="Optional output markdown report path.")
    args = parser.parse_args()

    client = auto_select_client(args.provider, args.model)
    agent = AnalystAgent(llm=client, max_retries=args.max_retries)
    report = agent.analyze(args.source, args.question)
    md = report.to_markdown()

    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"Report written to {args.out} (validation_passed={report.validation_passed})")
    else:
        print(md)
    return 0 if report.validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
