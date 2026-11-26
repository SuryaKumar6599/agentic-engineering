"""
scripts/generate_report.py
───────────────────────────
Generate a standalone HTML eval report from results/latest.jsonl.
Embeds a Chart.js dashboard with:
  - pass-rate donut chart
  - per-metric bar chart
  - latency histogram
  - cost time-series

Usage:
    python scripts/generate_report.py --results results/latest.jsonl --out results/report.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from jinja2 import Template

app = typer.Typer()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Agent Eval Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8fafc; color: #1e293b; }
  h1 { color: #0f172a; }
  .summary { display: flex; gap: 2rem; margin: 2rem 0; }
  .card { background: white; border-radius: 12px; padding: 1.5rem; flex: 1; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .metric { font-size: 2rem; font-weight: 700; }
  .label  { color: #64748b; font-size: .875rem; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 2rem 0; }
  .chart-box { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  th { background: #0f172a; color: white; padding: .75rem 1rem; text-align: left; }
  td { padding: .75rem 1rem; border-bottom: 1px solid #e2e8f0; }
  tr:last-child td { border-bottom: none; }
  .pass { color: #16a34a; font-weight: 600; }
  .fail { color: #dc2626; font-weight: 600; }
</style>
</head>
<body>
<h1>🤖 Agent Eval Report</h1>
<p style="color:#64748b">Generated: {{ generated_at }} &nbsp;|&nbsp; Model: <strong>{{ model_label }}</strong></p>

<div class="summary">
  <div class="card">
    <div class="metric">{{ pass_count }}/{{ total }}</div>
    <div class="label">Cases Passed</div>
  </div>
  <div class="card">
    <div class="metric">{{ "%.1f"|format(pass_rate * 100) }}%</div>
    <div class="label">Pass Rate</div>
  </div>
  <div class="card">
    <div class="metric">${{ "%.4f"|format(total_cost) }}</div>
    <div class="label">Total Cost</div>
  </div>
  <div class="card">
    <div class="metric">{{ "%.0f"|format(p95_latency_ms) }} ms</div>
    <div class="label">P95 Latency</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box">
    <h3>Pass / Fail</h3>
    <canvas id="donut"></canvas>
  </div>
  <div class="chart-box">
    <h3>Average Score per Metric</h3>
    <canvas id="metrics"></canvas>
  </div>
  <div class="chart-box">
    <h3>Latency Distribution (ms)</h3>
    <canvas id="latency"></canvas>
  </div>
  <div class="chart-box">
    <h3>Cost per Case (USD)</h3>
    <canvas id="cost"></canvas>
  </div>
</div>

<h2>Case Details</h2>
<table>
  <thead><tr><th>Task ID</th><th>Model</th><th>Pass?</th><th>Cost</th><th>Latency ms</th><th>Failing Metrics</th></tr></thead>
  <tbody>
  {% for r in results %}
  <tr>
    <td>{{ r.task_id }}</td>
    <td>{{ r.get("model") or "—" }}</td>
    <td class="{{ 'pass' if r.passed else 'fail' }}">{{ '✅' if r.passed else '❌' }}</td>
    <td>${{ "%.4f"|format(r.total_cost_usd) }}</td>
    <td>{{ "%.0f"|format(r.latency_ms) }}</td>
    <td>{{ r.failing_metrics | join(', ') or '—' }}</td>
  </tr>
  {% endfor %}
  </tbody>
</table>

<script>
const passCount = {{ pass_count }};
const failCount = {{ total - pass_count }};
new Chart(document.getElementById('donut'), {
  type: 'doughnut',
  data: {
    labels: ['Pass', 'Fail'],
    datasets: [{ data: [passCount, failCount], backgroundColor: ['#16a34a', '#dc2626'] }]
  }
});

const metricNames = {{ metric_names | tojson }};
const metricScores = {{ metric_scores | tojson }};
new Chart(document.getElementById('metrics'), {
  type: 'bar',
  data: {
    labels: metricNames,
    datasets: [{ label: 'Avg Score', data: metricScores, backgroundColor: '#3b82f6' }]
  },
  options: { scales: { y: { min: 0, max: 1 } } }
});

const latencies = {{ latencies | tojson }};
new Chart(document.getElementById('latency'), {
  type: 'bar',
  data: {
    labels: {{ task_ids | tojson }},
    datasets: [{ label: 'Latency ms', data: latencies, backgroundColor: '#8b5cf6' }]
  }
});

const costs = {{ costs | tojson }};
new Chart(document.getElementById('cost'), {
  type: 'bar',
  data: {
    labels: {{ task_ids | tojson }},
    datasets: [{ label: 'Cost USD', data: costs, backgroundColor: '#f59e0b' }]
  }
});
</script>
</body>
</html>
"""


@app.command()
def generate(
    results: Path = typer.Option(Path("results/latest.jsonl")),
    out: Path = typer.Option(Path("results/report.html")),
):
    from datetime import datetime
    import statistics

    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]

    pass_count = sum(1 for r in rows if all(m["status"] == "pass" for m in r["metrics"]))
    total = len(rows)
    total_cost = sum(r["total_cost_usd"] for r in rows)
    latencies = [r["latency_ms"] for r in rows]
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[0]

    # Per-metric avg scores
    metric_agg: dict[str, list[float]] = {}
    for r in rows:
        for m in r["metrics"]:
            if m["score"] is not None:
                metric_agg.setdefault(m["name"], []).append(m["score"])
    metric_names = list(metric_agg.keys())
    metric_scores = [
        round(sum(v) / len(v), 3) for v in metric_agg.values()
    ]

    enriched = []
    for r in rows:
        r["passed"] = all(m["status"] == "pass" for m in r["metrics"])
        r["failing_metrics"] = [m["name"] for m in r["metrics"] if m["status"] == "fail"]
        enriched.append(r)

    models_used = sorted({r.get("model") for r in rows if r.get("model")})
    model_label = ", ".join(models_used) if models_used else "—"

    html = Template(HTML_TEMPLATE).render(
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        model_label=model_label,
        pass_count=pass_count,
        total=total,
        pass_rate=pass_count / total if total else 0,
        total_cost=total_cost,
        p95_latency_ms=p95,
        results=enriched,
        metric_names=metric_names,
        metric_scores=metric_scores,
        latencies=latencies,
        costs=[r["total_cost_usd"] for r in rows],
        task_ids=[r["task_id"] for r in rows],
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    typer.echo(f"Report written to {out}")


if __name__ == "__main__":
    app()
