"""(b) Zero-shot LLM judge with the rubric, on both domains.

Costs real API calls, so predictions are cached to results/raw/judge_cache/ and
reused across reruns. Delete the cache file to re-judge.

The judge is scored exactly like every other system: same test split, same
macro-F1, same shortcut probes. It has no calibrated probabilities, so its
calibration entry is null.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.prepare import build_splits
from data.schema import DIMENSIONS, LABEL2ID, gold_matrix
from experiments.common import build_result, save_result, summarize
from llm.judge import JUDGE_MODEL, judge_records
from utils import load_json, save_json

CACHE_DIR = Path("results/raw/judge_cache")


def run(domain: str, model: str = JUDGE_MODEL, seed: int = 0, limit: int | None = None) -> dict:
    """Judge the test split of `domain` and score the result."""
    splits = build_splits(domain)
    test = splits["test"]
    if limit:
        test = test[:limit]

    cache_path = CACHE_DIR / f"{domain}__{model}.json"
    cached = load_json(cache_path) if cache_path.exists() else {}

    todo = [rec for rec in test if rec.response_id not in cached]
    if todo:
        print(f"Judging {len(todo)} responses with {model} ({len(cached)} cached)...")
        _, raw = judge_records(todo, model=model)
        for rec, labels in zip(todo, raw, strict=True):
            cached[rec.response_id] = labels
        save_json(cached, cache_path)
    else:
        print(f"All {len(test)} judgments served from cache.")

    y_pred = np.array(
        [[LABEL2ID[cached[rec.response_id][dim]] for dim in DIMENSIONS] for rec in test],
        dtype=int,
    )
    y_true = gold_matrix(test)

    result = build_result(
        experiment="b",
        system=f"judge_{model}",
        eval_domain=domain,
        records=test,
        y_true=y_true,
        y_pred=y_pred,
        probs=None,
        k=0,
        seed=seed,
        extra={"judge_model": model},
    )
    save_result(result)
    print(summarize(result))
    return result


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "math")
