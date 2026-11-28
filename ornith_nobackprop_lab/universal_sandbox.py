"""Isolated 2-second Python Execution Sandbox for Executable Pass@1 & Verifiable RL."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SandboxResult:
    passed: bool
    error_type: str  # "ok", "empty", "syntax", "timeout", "assertion", "runtime"
    message: str
    extracted_code: str


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning prefix when present."""
    if "</think>" in text:
        tail = text.split("</think>", 1)[1].strip()
        if tail:
            return tail
    return text.strip()


def _try_parse(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_python_code(raw_output: str, expected_fn: Optional[str] = None) -> str:
    """Extract clean Python code from a model generation without brittle domain regexes."""
    cleaned = strip_thinking(raw_output)
    if not cleaned:
        return ""

    # 1. Prefer fenced markdown code blocks
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        if expected_fn:
            for blk in fenced:
                if f"def {expected_fn}" in blk:
                    return blk.strip()
        # Concatenate or pick longest valid Python block
        valid = [b.strip() for b in fenced if b.strip()]
        if valid:
            for blk in valid:
                if _try_parse(blk):
                    return blk
            return valid[0]

    # 2. Unfenced fallback: start from first import/def/class if preceded by prose
    lines = cleaned.splitlines()
    start_idx = 0
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if (
            (expected_fn and s.startswith(f"def {expected_fn}"))
            or s.startswith("import ")
            or s.startswith("from ")
            or s.startswith("def ")
        ):
            start_idx = i
            break
    candidate_lines = lines[start_idx:]

    # Trim trailing natural language prose via progressive AST validation
    for end_idx in range(len(candidate_lines), 0, -1):
        sub = "\n".join(candidate_lines[:end_idx]).strip()
        if sub and _try_parse(sub):
            return sub
    return "\n".join(candidate_lines).strip()


def run_code_in_sandbox(
    raw_output: str,
    test_code: str,
    expected_fn: Optional[str] = None,
    mock_module_name: Optional[str] = None,
    mock_module_source: Optional[str] = None,
    timeout_sec: float = 2.0,
) -> SandboxResult:
    """Execute extracted Python code against assert unit tests in an isolated subprocess."""
    code = extract_python_code(raw_output, expected_fn=expected_fn)
    if not code:
        return SandboxResult(False, "empty", "No Python code found in output", "")
    if not _try_parse(code):
        return SandboxResult(False, "syntax", "Extracted code failed ast.parse", code)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        if mock_module_name and mock_module_source:
            (tmp_path / f"{mock_module_name}.py").write_text(mock_module_source, encoding="utf-8")

        runner_src = (
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            f"{code}\n\n"
            "# --- Unit Tests ---\n"
            f"{test_code}\n"
        )
        runner_path = tmp_path / "run_test.py"
        runner_path.write_text(runner_src, encoding="utf-8")

        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-s", str(runner_path)],
                cwd=tmpdir,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, "timeout", f"Execution exceeded {timeout_sec}s timeout", code)

        if proc.returncode == 0:
            return SandboxResult(True, "ok", proc.stdout.strip(), code)

        err = (proc.stderr or proc.stdout or "").strip()
        err_type = "assertion" if "AssertionError" in err else "runtime"
        return SandboxResult(False, err_type, err[-500:], code)
