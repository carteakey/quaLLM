"""Save and load structured evaluation results."""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .runner import RunResult

log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def _make_run_id(prompt_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = prompt_name.lower().replace(" ", "_")
    return f"{safe}_{ts}"


def save_run(
    results: list[RunResult],
    prompt_name: str,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Persist a list of RunResults to disk.

    Directory layout:
        results/<run_id>/
            metadata.json
            <model_safe_name>/
                output.txt      — raw model output
                output.py       — extracted code (if available)
                perf.json       — performance metrics

    Returns the run directory path.
    """
    if run_id is None:
        run_id = _make_run_id(prompt_name)

    base = (output_dir or RESULTS_DIR) / run_id
    base.mkdir(parents=True, exist_ok=True)

    # Metadata
    meta = {
        "run_id": run_id,
        "prompt_name": prompt_name,
        "models": [r.model_name for r in results],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (base / "metadata.json").write_text(json.dumps(meta, indent=2))

    for result in results:
        safe_name = result.model_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        model_dir = base / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Raw output
        (model_dir / "output.txt").write_text(result.raw_output)

        # Extracted code
        if result.code_output:
            (model_dir / "output.py").write_text(result.code_output)

        # Perf metrics
        perf = {
            "model_name": result.model_name,
            "prompt_name": result.prompt_name,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "prompt_tk_s": result.prompt_tk_s,
            "gen_tk_s": result.gen_tk_s,
            "total_time_s": result.total_time_s,
            "timestamp": result.timestamp,
        }
        (model_dir / "perf.json").write_text(json.dumps(perf, indent=2))

    log.info("Results saved to %s", base)
    return base


def load_run(run_dir: str | Path) -> dict:
    """Load a saved run from disk, returning metadata + per-model perf."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())

    models = {}
    for subdir in run_dir.iterdir():
        if subdir.is_dir():
            perf_file = subdir / "perf.json"
            if perf_file.exists():
                models[subdir.name] = json.loads(perf_file.read_text())

    return {"metadata": meta, "models": models}
