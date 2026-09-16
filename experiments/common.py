"""Shared plumbing for every experiment: one result schema, one writer.

All five experiments emit the same JSON shape so that scripts/make_table.py and
scripts/make_plots.py never need to know which experiment produced a row.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import DIMENSION_CODES, DIMENSIONS, Record
from eval.calibration import ece_per_dimension
from eval.metrics import evaluate_predictions
from eval.shortcuts import shortcut_report
from utils import save_json

RESULTS_DIR = Path("results/raw")


def build_result(
    experiment: str,
    system: str,
    eval_domain: str,
    records: Sequence[Record],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
    k: int = 0,
    seed: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one system on one test set and package the result."""
    groups = [rec.dialogue_id for rec in records]

    result: dict[str, Any] = {
        "experiment": experiment,
        "system": system,
        "eval_domain": eval_domain,
        "k": k,
        "seed": seed,
        "n_test": len(records),
        "n_dialogues": len(set(groups)),
        "metrics": evaluate_predictions(y_true, y_pred, groups=groups, seed=seed),
        "shortcuts": shortcut_report(records, y_pred, y_true, seed=seed),
    }
    # Majority and judge baselines have no calibrated probabilities to report.
    result["calibration"] = ece_per_dimension(probs, y_true) if probs is not None else None
    if extra:
        result["extra"] = extra
    return result


def result_path(experiment: str, system: str, eval_domain: str, k: int, seed: int) -> Path:
    return RESULTS_DIR / f"{experiment}__{system}__{eval_domain}__k{k}__s{seed}.json"


def save_result(result: dict[str, Any]) -> Path:
    path = result_path(
        result["experiment"],
        result["system"],
        result["eval_domain"],
        result["k"],
        result["seed"],
    )
    save_json(result, path)
    return path


def summarize(result: dict[str, Any]) -> str:
    """One-line console summary."""
    m = result["metrics"]
    per_dim = "  ".join(f"{DIMENSION_CODES[d]}={m[d]['macro_f1']:.3f}" for d in DIMENSIONS)
    return (
        f"[{result['experiment']}] {result['system']} -> {result['eval_domain']} "
        f"(k={result['k']}, seed={result['seed']}): mean={m['mean']['macro_f1']:.3f}  {per_dim}"
    )
