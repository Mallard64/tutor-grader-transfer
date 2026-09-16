#!/usr/bin/env python
"""Aggregate results/raw/*.json into a results table.

    uv run python scripts/make_table.py

Writes results/tables/results.md and results/tables/results.csv.

Across-seed aggregation reports mean +- standard deviation of macro-F1. The
bootstrap CI stored in each result file is the within-run uncertainty for one
seed; the std across seeds is the seed-to-seed variation. They answer different
questions, so both are kept: the per-seed CSV carries the CIs, the aggregate
table carries the std.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schema import DIMENSION_CODES, DIMENSIONS  # noqa: E402
from utils import load_json  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw"
OUT_DIR = REPO_ROOT / "results" / "tables"


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.json")):
        result = load_json(path)
        if "metrics" not in result:
            continue
        row = {
            "experiment": result["experiment"],
            "system": result["system"],
            "eval_domain": result["eval_domain"],
            "k": result["k"],
            "seed": result["seed"],
            "n_test": result["n_test"],
            "mean_macro_f1": result["metrics"]["mean"]["macro_f1"],
        }
        for dim in DIMENSIONS:
            m = result["metrics"][dim]
            row[f"f1_{DIMENSION_CODES[dim]}"] = m["macro_f1"]
            row[f"ci_low_{DIMENSION_CODES[dim]}"] = m["ci_low"]
            row[f"ci_high_{DIMENSION_CODES[dim]}"] = m["ci_high"]
        cal = result.get("calibration")
        row["ece_mean"] = cal["mean"]["ece"] if cal else None

        # Worst shortcut gap across dimensions: how much more the predictions
        # track a surface feature than the human labels do.
        # A constant predictor has no variance, so its correlations are NaN;
        # those carry no information about shortcuts and are dropped.
        gaps = [
            feat["gap"]
            for dim_report in result.get("shortcuts", {}).values()
            for feat in dim_report.values()
            if feat.get("gap") is not None and not pd.isna(feat["gap"])
        ]
        row["max_shortcut_gap"] = max(gaps, key=abs) if gaps else None
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    f1_cols = [f"f1_{DIMENSION_CODES[d]}" for d in DIMENSIONS]
    grouped = df.groupby(["experiment", "system", "eval_domain", "k"], as_index=False)

    agg = grouped.agg(
        n_seeds=("seed", "nunique"),
        n_test=("n_test", "first"),
        mean_f1=("mean_macro_f1", "mean"),
        std_f1=("mean_macro_f1", "std"),
        ece=("ece_mean", "mean"),
        shortcut_gap=("max_shortcut_gap", "mean"),
        **{f"{c}_mean": (c, "mean") for c in f1_cols},
    )
    return agg.sort_values(["experiment", "eval_domain", "k"]).reset_index(drop=True)


def to_markdown(agg: pd.DataFrame) -> str:
    lines = [
        "# Results",
        "",
        "macro-F1, mean +- std across seeds. Higher is better; ECE lower is better.",
        "`shortcut gap` = Spearman(prediction, feature) - Spearman(human label, feature),",
        "worst feature per run. Near zero means the model leans on surface features no",
        "more than the human labels do.",
        "",
        "| exp | system | eval domain | k | seeds | mean macro-F1 | "
        + " | ".join(DIMENSION_CODES[d] for d in DIMENSIONS)
        + " | ECE | shortcut gap |",
        "|---|---|---|---|---|---|" + "---|" * (len(DIMENSIONS) + 2),
    ]
    for _, r in agg.iterrows():
        std = "" if pd.isna(r["std_f1"]) else f" ± {r['std_f1']:.3f}"
        per_dim = " | ".join(f"{r[f'f1_{DIMENSION_CODES[d]}_mean']:.3f}" for d in DIMENSIONS)
        ece = "-" if pd.isna(r["ece"]) else f"{r['ece']:.3f}"
        gap = "-" if pd.isna(r["shortcut_gap"]) else f"{r['shortcut_gap']:+.3f}"
        lines.append(
            f"| {r['experiment']} | {r['system']} | {r['eval_domain']} | {int(r['k'])} | "
            f"{int(r['n_seeds'])} | {r['mean_f1']:.3f}{std} | {per_dim} | {ece} | {gap} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.json")):
        print(f"No results in {RAW_DIR}. Run scripts/run_experiments.py first.", file=sys.stderr)
        return 1

    df = load_rows()
    agg = aggregate(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "results.md").write_text(to_markdown(agg), encoding="utf-8")
    agg.to_csv(OUT_DIR / "results.csv", index=False)
    df.to_csv(OUT_DIR / "results_per_seed.csv", index=False)

    print(to_markdown(agg))
    print(f"Wrote {OUT_DIR}/results.md, results.csv, results_per_seed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
