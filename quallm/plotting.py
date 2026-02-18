"""Generate comparison plots for model evaluation results."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .runner import RunResult

log = logging.getLogger(__name__)

# ── Visual style ─────────────────────────────────────────────────
COLORS = ["#f39800", "#4a90d9", "#50c878", "#e45e32", "#9b59b6", "#1abc9c"]
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "font.family": "sans-serif",
    "font.size": 12,
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
})


def plot_comparison(
    results: list[RunResult],
    output_path: str | Path,
    title: Optional[str] = None,
) -> Path:
    """Generate a grouped bar chart comparing generation speed and prompt speed.

    Saves to *output_path* and returns the Path.
    """
    output_path = Path(output_path)

    names = [r.model_name for r in results]
    gen_speeds = [r.gen_tk_s for r in results]
    prompt_speeds = [r.prompt_tk_s for r in results]
    total_tokens = [r.completion_tokens for r in results]
    total_times = [r.total_time_s for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        title or f"Model Comparison — {results[0].prompt_name}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # ── 1. Generation speed (tk/s) ──────────────────────────────
    ax = axes[0]
    bars = ax.bar(names, gen_speeds, color=COLORS[: len(names)], edgecolor="white", linewidth=0.5)
    ax.set_title("Generation Speed", fontweight="bold")
    ax.set_ylabel("tokens / second")
    ax.grid(axis="y", linestyle="--")
    for bar, val in zip(bars, gen_speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ── 2. Prompt processing speed (tk/s) ───────────────────────
    ax = axes[1]
    bars = ax.bar(names, prompt_speeds, color=COLORS[: len(names)], edgecolor="white", linewidth=0.5)
    ax.set_title("Prompt Processing Speed", fontweight="bold")
    ax.set_ylabel("tokens / second")
    ax.grid(axis="y", linestyle="--")
    for bar, val in zip(bars, prompt_speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ── 3. Total output tokens ──────────────────────────────────
    ax = axes[2]
    bars = ax.bar(names, total_tokens, color=COLORS[: len(names)], edgecolor="white", linewidth=0.5)
    ax.set_title("Completion Tokens", fontweight="bold")
    ax.set_ylabel("tokens")
    ax.grid(axis="y", linestyle="--")
    for bar, val, t in zip(bars, total_tokens, total_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val}\n({t:.0f}s)", ha="center", va="bottom", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Comparison plot saved to %s", output_path)
    return output_path


def plot_summary_table(
    results: list[RunResult],
    output_path: str | Path,
) -> Path:
    """Render a summary table as an image."""
    output_path = Path(output_path)

    headers = ["Model", "Gen tk/s", "Prompt tk/s", "Completion\nTokens", "Total\nTime (s)"]
    rows = [
        [r.model_name, f"{r.gen_tk_s:.1f}", f"{r.prompt_tk_s:.1f}",
         str(r.completion_tokens), f"{r.total_time_s:.1f}"]
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j, _ in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor("#f39800")
        cell.set_text_props(fontweight="bold", color="white")

    # Alternate row colors
    for i, row in enumerate(rows):
        color = "#1e2d4a" if i % 2 == 0 else "#16213e"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Summary table saved to %s", output_path)
    return output_path
