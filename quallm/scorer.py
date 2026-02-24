"""Auto-scorer: run extracted output.py and check stdout against expected strings."""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def score_output(
    code_path: Path,
    expected_output: list[str],
    test_harness: str = "",
    timeout: float = 30.0,
) -> tuple[int, str]:
    """Run *code_path* (optionally with *test_harness* appended), capture stdout,
    and check that every string in *expected_output* appears in it.

    Returns (score 0-100, verdict string).

    Scoring:
    - If the process crashes  → 0.
    - Otherwise score = round(100 * hits / len(expected_output)).
    - If expected_output is empty, a clean exit gives 100.
    """
    if not code_path.exists():
        return 0, "no output.py"

    # Build the script to run: original code + optional test harness
    source = code_path.read_text()
    if test_harness:
        source = source + "\n\n# --- auto-scorer test harness ---\n" + test_harness

    # Write to a temp file so we never mutate the saved output.py
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=code_path.parent
    ) as tmp:
        tmp.write(source)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        return 0, f"timeout (>{timeout:.0f}s)"
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return 0, f"runner error: {exc}"
    finally:
        tmp_path.unlink(missing_ok=True)

    stdout = result.stdout
    stderr = result.stderr

    # Save scorer stdout/stderr for inspection
    (code_path.parent / "scorer_stdout.txt").write_text(stdout)
    if stderr.strip():
        (code_path.parent / "scorer_stderr.txt").write_text(stderr)

    if result.returncode != 0:
        last_err = stderr.strip().split("\n")[-1] if stderr.strip() else "unknown"
        return 0, f"crashed (exit {result.returncode}): {last_err}"

    # Empty expected_output → clean run = 100
    if not expected_output:
        return 100, "ran clean (no assertions)"

    hits = [e for e in expected_output if e in stdout]
    misses = [e for e in expected_output if e not in stdout]
    score = round(100 * len(hits) / len(expected_output))

    if score == 100:
        verdict = "PASS"
    elif score == 0:
        verdict = f"FAIL — expected: {misses}"
    else:
        verdict = f"PARTIAL ({score}%) — missing: {misses}"

    return score, verdict
