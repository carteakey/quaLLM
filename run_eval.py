#!/usr/bin/env python3
"""quaLLM — run an evaluation across one or more models.

Usage examples:

  # Run bouncing balls on both models (framework manages servers):
  python run_eval.py \\
      --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \\
      --prompt prompts/bouncing_balls.txt

  # Models already running — skip server orchestration:
  python run_eval.py \\
      --models model_configs/qwen3_coder_next.yaml \\
      --prompt prompts/bouncing_balls.txt \\
      --no-start-server

  # Run outputs and capture screenshots:
  python run_eval.py \\
      --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \\
      --prompt prompts/bouncing_balls.txt \\
      --run-outputs

  # Blind evaluation (anonymized model names):
  python run_eval.py \\
      --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \\
      --prompt prompts/bouncing_balls.txt \\
      --blind --run-outputs
"""

import argparse
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from quallm.config import load_model_config, load_prompt, ModelConfig
from quallm.orchestrator import start_server, stop_server, is_server_ready
from quallm.runner import run_prompt, RunResult
from quallm.results import save_run, RESULTS_DIR
from quallm.plotting import plot_comparison, plot_summary_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quallm")


# ── Blind evaluation helpers ────────────────────────────────────
BLIND_NAMES = [
    "Model Alpha", "Model Beta", "Model Gamma", "Model Delta",
    "Model Epsilon", "Model Zeta", "Model Eta", "Model Theta",
]


def _blindify(results: list[RunResult]) -> tuple[list[RunResult], dict]:
    """Shuffle results and replace model names with anonymous labels.

    Returns (anonymized_results, blind_key) where blind_key maps
    anonymous names back to real model names.
    """
    indices = list(range(len(results)))
    random.shuffle(indices)

    blind_key = {}
    anon_results = []

    for new_idx, orig_idx in enumerate(indices):
        r = results[orig_idx]
        anon_name = BLIND_NAMES[new_idx] if new_idx < len(BLIND_NAMES) else f"Model {new_idx + 1}"
        blind_key[anon_name] = r.model_name

        anon_results.append(RunResult(
            model_name=anon_name,
            prompt_name=r.prompt_name,
            raw_output=r.raw_output,
            code_output=r.code_output,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            prompt_tk_s=r.prompt_tk_s,
            gen_tk_s=r.gen_tk_s,
            total_time_s=r.total_time_s,
            timestamp=r.timestamp,
        ))

    return anon_results, blind_key


# ── Run output.py with screenshot capture ───────────────────────
def _capture_output_screenshot(
    code_path: Path,
    screenshot_path: Path,
    display_seconds: float = 5.0,
) -> bool:
    """Run an output.py file, wait, then capture a screenshot of the window.

    Uses xdotool + import (ImageMagick) on Linux to grab the window.
    Falls back to just running the script if no screenshot tools available.
    """
    log.info("Running %s for %.0fs to capture screenshot…", code_path, display_seconds)

    proc = subprocess.Popen(
        [sys.executable, str(code_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(display_seconds)

    # Try to capture screenshot via ImageMagick import + xdotool
    try:
        # Find the window by PID
        result = subprocess.run(
            ["xdotool", "search", "--pid", str(proc.pid)],
            capture_output=True, text=True, timeout=3,
        )
        window_ids = result.stdout.strip().split("\n")

        if window_ids and window_ids[0]:
            # Capture the window
            subprocess.run(
                ["import", "-window", window_ids[-1], str(screenshot_path)],
                timeout=5,
            )
            log.info("Screenshot saved: %s", screenshot_path)
        else:
            # Fallback: capture entire screen region
            subprocess.run(
                ["import", "-window", "root", str(screenshot_path)],
                timeout=5,
            )
            log.info("Full-screen screenshot saved: %s", screenshot_path)

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("Screenshot capture failed (%s). Install xdotool + imagemagick for auto-screenshots.", e)

    # Kill the tkinter window
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    return screenshot_path.exists()


def _run_outputs(run_dir: Path, display_seconds: float = 5.0) -> None:
    """Find and run all output.py files in a run directory, capturing screenshots."""
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        code_path = model_dir / "output.py"
        if code_path.exists():
            screenshot_path = model_dir / "screenshot.png"
            _capture_output_screenshot(code_path, screenshot_path, display_seconds)


# ── Summary printing ────────────────────────────────────────────
def _print_summary(results: list[RunResult]) -> None:
    """Print a quick ascii summary table."""
    hdr = f"{'Model':<25} {'Gen tk/s':>10} {'Prompt tk/s':>12} {'Tokens':>8} {'Time (s)':>10}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.model_name:<25} {r.gen_tk_s:>10.1f} {r.prompt_tk_s:>12.1f} "
              f"{r.completion_tokens:>8} {r.total_time_s:>10.1f}")
    print("=" * len(hdr) + "\n")


def _dummy_results(models: list[ModelConfig], prompt_name: str) -> list[RunResult]:
    """Generate fake results for --dry-run mode."""
    return [
        RunResult(
            model_name=m.name,
            prompt_name=prompt_name,
            raw_output="# Dry-run: no actual output generated.",
            code_output="# Dry-run placeholder",
            prompt_tokens=random.randint(200, 500),
            completion_tokens=random.randint(800, 3000),
            prompt_tk_s=round(random.uniform(50, 200), 1),
            gen_tk_s=round(random.uniform(5, 25), 1),
            total_time_s=round(random.uniform(30, 300), 1),
        )
        for m in models
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="quaLLM — evaluate prompts across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Paths to model config YAML files",
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Path to prompt file (.txt or .yaml)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Skip server start/stop (assume servers already running)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and generate dummy results without running models",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Anonymize model names for unbiased human evaluation",
    )
    parser.add_argument(
        "--run-outputs",
        action="store_true",
        help="Run generated output.py files and capture screenshots",
    )
    parser.add_argument(
        "--display-seconds",
        type=float,
        default=5.0,
        help="Seconds to display output window before capturing screenshot (default: 5)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=RESULTS_DIR,
        help=f"Base output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens for generation (default: 16384)",
    )

    args = parser.parse_args()

    # ── Load configs ────────────────────────────────────────────
    models = [load_model_config(p) for p in args.models]
    prompt = load_prompt(args.prompt)

    log.info("Prompt: %s (%s eval)", prompt.name, prompt.eval_type)
    for m in models:
        log.info("Model:  %s  (port %d)", m.name, m.port)

    if args.dry_run:
        log.info("*** DRY RUN — generating dummy results ***")
        results = _dummy_results(models, prompt.name)
    else:
        # ── Run each model sequentially ─────────────────────────
        results: list[RunResult] = []

        for model in models:
            proc = None
            try:
                if not args.no_start_server:
                    log.info("Starting server for %s …", model.name)
                    proc = start_server(model, log_file=str(args.output_dir / f"{model.safe_name}_server.log"))
                else:
                    # Verify server is reachable
                    if not is_server_ready(model.port):
                        log.error("Server for %s not reachable on port %d", model.name, model.port)
                        sys.exit(1)

                result = run_prompt(model, prompt, max_tokens=args.max_tokens)
                results.append(result)

            finally:
                if proc is not None:
                    stop_server(proc)

    # ── Blind mode ──────────────────────────────────────────────
    blind_key = None
    if args.blind:
        results, blind_key = _blindify(results)
        log.info("🔒 Blind mode: model identities anonymized")

    # ── Save results ────────────────────────────────────────────
    run_dir = save_run(results, prompt.name, output_dir=args.output_dir)

    # Save blind key separately (sealed — don't peek until eval is done!)
    if blind_key:
        blind_path = run_dir / "blind_key.json"
        blind_path.write_text(json.dumps(blind_key, indent=2))
        log.info("🔑 Blind key saved to %s (open AFTER scoring!)", blind_path)

    # ── Generate plots ──────────────────────────────────────────
    plot_comparison(results, run_dir / "comparison.png")
    plot_summary_table(results, run_dir / "summary_table.png")

    # ── Run outputs + capture screenshots ───────────────────────
    if args.run_outputs and prompt.extract_code and prompt.eval_type == "human":
        _run_outputs(run_dir, display_seconds=args.display_seconds)

    # ── Print summary ───────────────────────────────────────────
    _print_summary(results)
    log.info("Results saved to: %s", run_dir)

    # Hint about running code outputs
    if prompt.extract_code and prompt.eval_type == "human" and not args.run_outputs:
        for r in results:
            safe = r.model_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            code_path = run_dir / safe / "output.py"
            if code_path.exists():
                log.info("Run visual eval: python %s", code_path)

    if args.blind:
        print("🔒 Models are anonymized. Score them first, then check blind_key.json!")


if __name__ == "__main__":
    main()
