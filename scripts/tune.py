#!/usr/bin/env python
"""Compare training recipes on the VALIDATION split only.

    uv run python scripts/tune.py --configs configs/tune_*.yaml --seeds 0 1 2

Writes one JSON per (arm, seed) to results/tuning/ and prints a summary.

Two rules this script exists to enforce:

1. Selection happens on validation, never on test. Nothing here touches the
   test split or writes to results/raw/. The test set is spent once, on the
   frozen recipe, by scripts/run_experiments.py.
2. Whatever wins must then be used identically for (c), (d) and (e). Tuning on
   math and then reporting a transfer gap against an untuned programming arm
   would inflate the gap.

Validation here is 334 responses over 41 dialogues, which is small. Three or
four arms is about the most it can honestly adjudicate -- a broad grid would
just fit the noise.
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.prepare import build_splits  # noqa: E402
from data.schema import DIMENSIONS  # noqa: E402
from utils import RunConfig, load_json, save_json  # noqa: E402

# Honors TGT_RESULTS_ROOT so a sweep survives a recycled Colab VM.
_ROOT = Path(os.environ.get("TGT_RESULTS_ROOT", REPO_ROOT / "results"))
OUT_DIR = _ROOT / "tuning"
TABLE_PATH = REPO_ROOT / "results" / "tables" / "tuning.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--domain", default="math")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="keep run dirs; by default they are deleted once the val score is recorded",
    )
    return p.parse_args()


def result_path(arm: str, seed: int) -> Path:
    return OUT_DIR / f"{arm}__s{seed}.json"


def main() -> int:
    args = parse_args()
    splits = build_splits(args.domain)
    train, val = splits["train"], splits["val"]
    print(f"train {len(train)} / val {len(val)} responses ({args.domain})\n")

    from models.train import train_grader

    for cfg_path in args.configs:
        base = RunConfig.from_yaml(cfg_path)
        for seed in args.seeds:
            path = result_path(base.name, seed)
            if path.exists() and not args.force:
                print(f"skip {base.name} seed {seed} (already done)")
                continue

            cfg = RunConfig(
                **{k: v for k, v in vars(base).items() if k != "extra"},
                extra=dict(base.extra),
            )
            cfg.name = f"{base.name}_s{seed}"
            cfg.seed = seed

            print(f"\n=== {base.name} seed {seed} ===", flush=True)
            _, val_metrics = train_grader(cfg, train, val)

            record = {
                "arm": base.name,
                "seed": seed,
                "config": cfg_path,
                "num_epochs": cfg.num_epochs,
                "class_weights": cfg.class_weights,
                "val": {k: v for k, v in val_metrics.items() if "f1" in k or "loss" in k},
            }
            save_json(record, path)
            score = val_metrics.get("eval_mean_macro_f1", float("nan"))
            print(f"RESULT {base.name} s{seed}: {score:.4f}", flush=True)

            if not args.keep_checkpoints:
                import shutil

                shutil.rmtree(cfg.run_dir, ignore_errors=True)

    summarize()
    return 0


def summarize() -> None:
    """Print and save mean +- std of validation macro-F1 per arm."""
    if not OUT_DIR.exists():
        return
    by_arm: dict[str, list[dict]] = {}
    for path in sorted(OUT_DIR.glob("*.json")):
        r = load_json(path)
        by_arm.setdefault(r["arm"], []).append(r)
    if not by_arm:
        return

    cols = " | ".join(d[:2].upper() for d in DIMENSIONS)
    header = f"| arm | seeds | val macro-F1 | {cols} |"
    lines = [
        "## Validation comparison (selection metric -- test set not touched)",
        "",
        header,
        "|---|---|---|" + "---|" * len(DIMENSIONS),
    ]
    ranked = []
    for arm, rows in sorted(by_arm.items()):
        vals = [r["val"]["eval_mean_macro_f1"] for r in rows]
        mean = st.mean(vals)
        std = st.stdev(vals) if len(vals) > 1 else 0.0
        per_dim = []
        for d in DIMENSIONS:
            xs = [r["val"][f"eval_f1_{d}"] for r in rows if f"eval_f1_{d}" in r["val"]]
            per_dim.append(f"{st.mean(xs):.3f}" if xs else "-")
        cells = " | ".join(per_dim)
        lines.append(f"| {arm} | {len(vals)} | {mean:.4f} ± {std:.4f} | {cells} |")
        ranked.append((mean, arm, std, len(vals)))

    ranked.sort(reverse=True)
    body = "\n".join(lines)
    print("\n" + body)

    best = ranked[0]
    print(f"\nBest on validation: {best[1]} ({best[0]:.4f} ± {best[2]:.4f}, {best[3]} seeds)")
    if len(ranked) > 1:
        gap = best[0] - ranked[1][0]
        pooled = max(best[2], ranked[1][2], 1e-9)
        print(
            f"Margin over {ranked[1][1]}: {gap:+.4f} ({gap / pooled:.2f} sd). "
            "On 41 validation dialogues, treat anything under ~1 sd as a tie."
        )

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(body + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
