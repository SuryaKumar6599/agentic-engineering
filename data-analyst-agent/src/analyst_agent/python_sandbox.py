"""Sandboxed Python executor using AST validation + isolated subprocess."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from analyst_agent.models import PythonResult

ALLOWED_MODULES = frozenset({
    "math",
    "statistics",
    "json",
    "csv",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "decimal",
    "re",
    "random",
    "numpy",
    "pandas",
    "scipy",
})

FORBIDDEN_BUILTINS = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
})

FORBIDDEN_ATTRS = frozenset({
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__builtins__",
    "__import__",
    "system",
    "popen",
})


def validate_python_ast(code: str) -> str | None:
    """Return an error string if the Python AST violates sandbox policy."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return f"Disallowed import: '{alias.name}'."
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                return "Relative imports are not allowed."
            root = node.module.split(".")[0]
            if root not in ALLOWED_MODULES:
                return f"Disallowed import from: '{node.module}'."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                return f"Disallowed builtin call: '{node.func.id}()'."
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return f"Disallowed attribute access: '.{node.attr}'."
    return None


_RUNNER_TEMPLATE = """\
import json
import math
import statistics
import sys

try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
except Exception:
    pass

with open({data_path_repr}, "r", encoding="utf-8") as _f:
    DATA = json.load(_f)

_METRICS = {{}}

def record_metric(name, value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        _METRICS[str(name)] = None
    else:
        _METRICS[str(name)] = value

# --- User analysis script ---
{user_code}
# --- End user analysis script ---

with open({metrics_path_repr}, "w", encoding="utf-8") as _mf:
    json.dump(_METRICS, _mf, default=str)
"""


class PythonSandbox:
    """Executes AST-validated Python analysis code in an isolated subprocess."""

    def __init__(self, timeout_seconds: float = 10.0, cpu_limit: int = 8) -> None:
        self.timeout_seconds = timeout_seconds
        self.cpu_limit = cpu_limit

    def execute(
        self,
        script_id: str,
        code: str,
        data_payload: dict[str, Any],
    ) -> PythonResult:
        start = time.perf_counter()
        ast_err = validate_python_ast(code)
        if ast_err:
            return PythonResult(
                script_id=script_id,
                code=code,
                stdout="",
                stderr="",
                metrics={},
                elapsed_ms=(time.perf_counter() - start) * 1000,
                exit_code=-1,
                error=f"PythonSandboxError: {ast_err}",
            )

        with tempfile.TemporaryDirectory(prefix="analyst_py_") as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "data.json"
            metrics_path = tmp / "metrics.json"
            script_path = tmp / "analysis.py"

            data_path.write_text(json.dumps(data_payload, default=str), encoding="utf-8")
            indented_code = code.strip()
            runner_src = _RUNNER_TEMPLATE.format(
                cpu_limit=self.cpu_limit,
                data_path_repr=repr(str(data_path)),
                metrics_path_repr=repr(str(metrics_path)),
                user_code=indented_code,
            )
            script_path.write_text(runner_src, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    cwd=tmpdir,
                    env={"PATH": ""},
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return PythonResult(
                    script_id=script_id,
                    code=code,
                    stdout="",
                    stderr="",
                    metrics={},
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    exit_code=-1,
                    error=f"PythonSandboxError: Execution timed out after {self.timeout_seconds}s.",
                )

            metrics: dict[str, Any] = {}
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metrics = {}

            err_msg = None
            if proc.returncode != 0:
                err_msg = f"PythonRuntimeError (exit={proc.returncode}): {proc.stderr.strip()}"

            return PythonResult(
                script_id=script_id,
                code=code,
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                metrics=metrics,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                exit_code=proc.returncode,
                error=err_msg,
            )
