#!/usr/bin/env python3
"""quaLLM — run an evaluation across one or more models.

Usage examples:

  # Run a single prompt:
  python run_eval.py \\
      --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \\
      --prompt prompts/bouncing_balls.txt

  # Run an entire prompt set:
  python run_eval.py \\
      --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \\
      --prompt-set prompt_sets/algorithms.yaml

  # Blind evaluation with screenshot capture:
  python run_eval.py \\
      --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \\
      --prompt-set prompt_sets/vibe_coding.yaml \\
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

import yaml

from quallm.config import load_model_config, load_prompt, ModelConfig
from quallm.orchestrator import start_server, stop_server, is_server_ready
from quallm.runner import run_prompt, RunResult
from quallm.results import save_run, RESULTS_DIR
from quallm.plotting import plot_comparison, plot_summary_table
from quallm.scorer import score_output

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
def _screenshot_pil(screenshot_path: Path) -> bool:
    """Capture screenshot using PIL ImageGrab (no system deps needed)."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        img.save(str(screenshot_path))
        return True
    except Exception:
        return False


def _screenshot_xdotool(pid: int, screenshot_path: Path) -> bool:
    """Capture a specific window screenshot using xdotool + ImageMagick import."""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--pid", str(pid)],
            capture_output=True, text=True, timeout=3,
        )
        window_ids = result.stdout.strip().split("\n")
        if window_ids and window_ids[0]:
            subprocess.run(
                ["import", "-window", window_ids[-1], str(screenshot_path)],
                timeout=5,
            )
            return screenshot_path.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def _screenshot_scrot(screenshot_path: Path) -> bool:
    """Capture focused window using scrot."""
    try:
        subprocess.run(
            ["scrot", "-u", str(screenshot_path)],
            timeout=5,
        )
        return screenshot_path.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _capture_output_screenshot(
    code_path: Path,
    screenshot_path: Path,
    display_seconds: float = 5.0,
) -> bool:
    """Run an output.py file, wait, then capture a screenshot.

    Tries multiple capture strategies in order:
      1. xdotool + ImageMagick import (window-specific)
      2. scrot (focused window)
      3. PIL ImageGrab (Python-only, no system deps)

    Also checks for runtime errors and saves them to runtime_error.txt.
    """
    log.info("Running %s for %.0fs to capture screenshot…", code_path, display_seconds)

    proc = subprocess.Popen(
        [sys.executable, str(code_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Check if process crashes quickly (before display timeout)
    try:
        proc.wait(timeout=min(2.0, display_seconds))
        # Process exited already — likely a crash
        stderr_output = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.returncode != 0:
            error_path = code_path.parent / "runtime_error.txt"
            error_path.write_text(f"Exit code: {proc.returncode}\n\n{stderr_output}")
            log.error("❌ %s crashed (exit %d): %s", code_path.name, proc.returncode,
                      stderr_output.strip().split("\n")[-1] if stderr_output.strip() else "unknown error")
            return False
        # Exited cleanly (maybe a non-GUI script)
        log.info("Process exited cleanly (exit 0) before display timeout.")
        return False
    except subprocess.TimeoutExpired:
        pass  # Still running — good, it's a GUI app

    # Wait remaining display time
    remaining = display_seconds - 2.0
    if remaining > 0:
        time.sleep(remaining)

    # Try capture strategies in order of reliability
    captured = False

    if not captured:
        captured = _screenshot_xdotool(proc.pid, screenshot_path)
        if captured:
            log.info("Screenshot saved (xdotool): %s", screenshot_path)

    if not captured:
        captured = _screenshot_scrot(screenshot_path)
        if captured:
            log.info("Screenshot saved (scrot): %s", screenshot_path)

    if not captured:
        captured = _screenshot_pil(screenshot_path)
        if captured:
            log.info("Screenshot saved (PIL): %s", screenshot_path)

    if not captured:
        log.warning("Screenshot capture failed. Install one of: Pillow, xdotool+imagemagick, or scrot.")

    # Kill the tkinter window
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Capture any stderr even if the process was running
    stderr_output = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if stderr_output.strip():
        error_path = code_path.parent / "runtime_error.txt"
        error_path.write_text(f"Exit code: {proc.returncode}\n\n{stderr_output}")
        if proc.returncode != 0:
            log.warning("⚠️  %s had errors (exit %d)", code_path.name, proc.returncode)
        else:
            log.info("Process had stderr output (saved to runtime_error.txt)")

    return captured


def _run_outputs(run_dir: Path, display_seconds: float = 5.0) -> None:
    """Find and run all output.py files in a run directory, capturing screenshots."""
    crashed = []
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        code_path = model_dir / "output.py"
        if code_path.exists():
            screenshot_path = model_dir / "screenshot.png"
            success = _capture_output_screenshot(code_path, screenshot_path, display_seconds)
            error_path = model_dir / "runtime_error.txt"
            if error_path.exists():
                crashed.append(model_dir.name)

    if crashed:
        log.warning("⚠️  Runtime errors in: %s (see runtime_error.txt in each dir)", ", ".join(crashed))


# ── Summary printing ────────────────────────────────────────────
def _print_summary(results: list[RunResult], scores: dict | None = None) -> None:
    """Print a quick ascii summary table."""
    show_score = bool(scores)
    hdr = (
        f"{'Model':<25} {'Gen tk/s':>10} {'Prompt tk/s':>12} {'Tokens':>8} {'Time (s)':>10}"
        + (f" {'Score':>7}" if show_score else "")
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        score_col = ""
        if show_score and r.model_name in scores:
            sc = scores[r.model_name]["score"]
            score_col = f" {sc:>6d}%"
        print(f"{r.model_name:<25} {r.gen_tk_s:>10.1f} {r.prompt_tk_s:>12.1f} "
              f"{r.completion_tokens:>8} {r.total_time_s:>10.1f}{score_col}")
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


def _load_prompt_set(path: str) -> tuple[list[str], dict]:
    """Load a prompt set YAML and return (list of prompt paths, prompt set metadata)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("prompts", []), data


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
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt", "-p",
        help="Path to a single prompt file (.txt or .yaml)",
    )
    prompt_group.add_argument(
        "--prompt-set",
        help="Path to a prompt set YAML (runs all prompts in the set)",
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

    # Resolve prompt list (single prompt or prompt set)
    if args.prompt_set:
        prompt_paths, prompt_set_meta = _load_prompt_set(args.prompt_set)
        set_eval_type = prompt_set_meta.get("eval_type")
        log.info("Prompt set: %s (%d prompts)", args.prompt_set, len(prompt_paths))
    else:
        prompt_paths = [args.prompt]
        set_eval_type = None

    prompts = [load_prompt(p) for p in prompt_paths]

    # Propagate prompt-set-level eval_type to individual prompts that were
    # loaded from plain .txt files (which default to "human").
    if set_eval_type:
        for p in prompts:
            if p.eval_type == "human":
                p.eval_type = set_eval_type

    for m in models:
        log.info("Model:  %s  (port %d)", m.name, m.port)

    # ── Run each prompt ─────────────────────────────────────────
    all_run_dirs = []

    for prompt in prompts:
        log.info("─" * 60)
        log.info("Prompt: %s (%s eval)", prompt.name, prompt.eval_type)

        if args.dry_run:
            log.info("*** DRY RUN — generating dummy results ***")
            results = _dummy_results(models, prompt.name)
        else:
            results: list[RunResult] = []

            for model in models:
                proc = None
                try:
                    if not args.no_start_server:
                        log.info("Starting server for %s …", model.name)
                        proc = start_server(model, log_file=str(args.output_dir / f"{model.safe_name}_server.log"))
                    else:
                        if not is_server_ready(model.port):
                            log.error("Server for %s not reachable on port %d", model.name, model.port)
                            sys.exit(1)

                    result = run_prompt(model, prompt, max_tokens=args.max_tokens)
                    results.append(result)

                finally:
                    if proc is not None:
                        stop_server(proc)

        # ── Blind mode ──────────────────────────────────────────
        blind_key = None
        if args.blind:
            results, blind_key = _blindify(results)
            log.info("🔒 Blind mode: model identities anonymized")

        # ── Save results ────────────────────────────────────────
        run_dir = save_run(results, prompt.name, output_dir=args.output_dir)
        all_run_dirs.append(run_dir)

        if blind_key:
            blind_path = run_dir / "blind_key.json"
            blind_path.write_text(json.dumps(blind_key, indent=2))
            log.info("🔑 Blind key saved to %s (open AFTER scoring!)", blind_path)

        # ── Generate plots ──────────────────────────────────────
        plot_comparison(results, run_dir / "comparison.png")
        plot_summary_table(results, run_dir / "summary_table.png")

        # ── Auto-score (eval_type == "auto") ────────────────────
        scores: dict | None = None
        if prompt.eval_type == "auto":
            log.info("Auto-scoring %s …", prompt.name)
            scores = {}
            for r in results:
                safe = r.model_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
                code_path = run_dir / safe / "output.py"
                sc, verdict = score_output(
                    code_path,
                    prompt.expected_output,
                    prompt.test_harness,
                )
                scores[r.model_name] = {"score": sc, "verdict": verdict}
                log.info("  %-25s → %3d/100  %s", r.model_name, sc, verdict)

            (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))

        # ── Run outputs + capture screenshots ───────────────────
        if args.run_outputs and prompt.extract_code and prompt.eval_type == "human":
            _run_outputs(run_dir, display_seconds=args.display_seconds)

        # ── Print summary ───────────────────────────────────────
        _print_summary(results, scores)
        log.info("Results saved to: %s", run_dir)

        if prompt.extract_code and prompt.eval_type == "human" and not args.run_outputs:
            for r in results:
                safe = r.model_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
                code_path = run_dir / safe / "output.py"
                if code_path.exists():
                    log.info("Run visual eval: python %s", code_path)

    # ── Final summary ───────────────────────────────────────────
    if len(all_run_dirs) > 1:
        log.info("═" * 60)
        log.info("Prompt set complete: %d prompts evaluated", len(all_run_dirs))
        for d in all_run_dirs:
            log.info("  📁 %s", d)

    if args.blind:
        print("🔒 Models are anonymized. Score them first, then check blind_key.json!")


if __name__ == "__main__":
    main()
