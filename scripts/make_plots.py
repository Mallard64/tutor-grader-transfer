#!/usr/bin/env python
"""Learning-curve plot: (d) math-pretrained vs (e) programming-only.

    uv run python scripts/make_plots.py

Writes results/figures/learning_curve.png and learning_curve_per_dimension.png.

The plot answers the second half of the research question: how many labeled
programming examples does (e) need before it catches (d)? The horizontal
reference lines are the two things "closing the gap" could mean -- the zero-shot
transfer score (c) and the majority baseline (a).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs on servers and in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schema import DIMENSIONS  # noqa: E402
from utils import load_json  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
FIG_DIR = REPO_ROOT / "results" / "figures"

SERIES = {
    "d": ("math-pretrained + k programming", "tab:blue", "o"),
    "e": ("programming-only, k examples", "tab:orange", "s"),
}


def collect(dimension: str | None = None) -> dict[str, dict[int, list[float]]]:
    """{experiment: {k: [score per seed]}} on the programming test set."""
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(RAW_DIR.glob("*.json")):
        r = load_json(path)
        if r.get("eval_domain") != "programming" or "metrics" not in r:
            continue
        key = dimension or "mean"
        out[r["experiment"]][r["k"]].append(r["metrics"][key]["macro_f1"])
    return out


def mean_std(points: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks = np.array(sorted(points))
    means = np.array([np.mean(points[k]) for k in ks])
    stds = np.array([np.std(points[k], ddof=1) if len(points[k]) > 1 else 0.0 for k in ks])
    return ks, means, stds


def plot_curve(ax, data: dict[str, dict[int, list[float]]], title: str) -> None:
    for exp, (label, color, marker) in SERIES.items():
        if exp not in data or not data[exp]:
            continue
        ks, means, stds = mean_std(data[exp])
        ax.plot(ks, means, marker=marker, color=color, label=label)
        ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.15)

    # Reference lines: zero-shot transfer and the majority floor.
    if data.get("c"):
        _, c_means, _ = mean_std(data["c"])
        ax.axhline(
            float(np.mean(c_means)),
            ls="--",
            color="tab:green",
            lw=1.2,
            label="(c) math-trained, zero-shot",
        )
    if data.get("a"):
        _, a_means, _ = mean_std(data["a"])
        ax.axhline(
            float(np.mean(a_means)), ls=":", color="gray", lw=1.2, label="(a) majority baseline"
        )

    ax.set_xscale("symlog", linthresh=8)
    ax.set_xlabel("labeled programming examples (k)")
    ax.set_ylabel("macro-F1")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main() -> int:
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.json")):
        print(f"No results in {RAW_DIR}. Run scripts/run_experiments.py first.", file=sys.stderr)
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = collect()
    if not data.get("d") and not data.get("e"):
        print("No (d) or (e) results yet; nothing to plot.", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_curve(ax, data, "Programming test set: mean macro-F1 across 4 dimensions")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curve.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, dim in zip(axes.flat, DIMENSIONS, strict=True):
        plot_curve(ax, collect(dim), dim.replace("_", " "))
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curve_per_dimension.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {FIG_DIR}/learning_curve.png and learning_curve_per_dimension.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
