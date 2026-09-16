"""Macro-F1 per dimension with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score

from data.schema import DIMENSIONS, LABELS, UNLABELED

N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-F1 over the 3 classes, with all classes always in the average.

    `labels=range(len(LABELS))` matters: if a model never predicts 'To some
    extent' and the bootstrap sample happens to contain none either, sklearn
    would otherwise silently drop the class and inflate the score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
    )


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str] | None = None,
    n_boot: int = N_BOOTSTRAP,
    level: float = CI_LEVEL,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for macro-F1.

    If `groups` is given (dialogue ids), resamples whole dialogues rather than
    individual responses. Responses to the same dialogue are correlated, so
    resampling them independently gives intervals that are too narrow.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    point = macro_f1(y_true, y_pred)
    if len(y_true) == 0 or n_boot <= 0:
        return {
            "macro_f1": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": int(len(y_true)),
        }

    rng = np.random.default_rng(seed)

    if groups is None:
        index_pool = [np.arange(len(y_true))]
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        index_pool = [np.flatnonzero(groups == g) for g in uniq]

    scores = np.empty(n_boot, dtype=float)
    n_units = len(index_pool)
    for b in range(n_boot):
        picks = rng.integers(0, n_units, size=n_units)
        idx = np.concatenate([index_pool[p] for p in picks])
        scores[b] = macro_f1(y_true[idx], y_pred[idx])

    scores = scores[~np.isnan(scores)]
    alpha = (1.0 - level) / 2.0
    return {
        "macro_f1": point,
        "ci_low": float(np.quantile(scores, alpha)),
        "ci_high": float(np.quantile(scores, 1.0 - alpha)),
        "n": int(len(y_true)),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence[str] | None = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per-dimension macro-F1 + CI, plus the across-dimension mean.

    `y_true` / `y_pred` are [n_examples, n_dimensions]. Entries equal to
    UNLABELED in y_true are skipped for that dimension.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.shape[1] != len(DIMENSIONS):
        raise ValueError(f"expected {len(DIMENSIONS)} dimension columns, got {y_true.shape[1]}")

    out: dict[str, dict[str, float]] = {}
    for j, dim in enumerate(DIMENSIONS):
        mask = y_true[:, j] != UNLABELED
        g = None if groups is None else np.asarray(groups)[mask]
        out[dim] = bootstrap_ci(
            y_true[mask, j], y_pred[mask, j], groups=g, n_boot=n_boot, seed=seed
        )

    finite = [out[d]["macro_f1"] for d in DIMENSIONS if not np.isnan(out[d]["macro_f1"])]
    out["mean"] = {"macro_f1": float(np.mean(finite)) if finite else float("nan")}
    return out


def paired_bootstrap_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    groups: Sequence[str] | None = None,
    n_boot: int = N_BOOTSTRAP,
    level: float = CI_LEVEL,
    seed: int = 0,
) -> dict[str, float]:
    """CI on macro-F1(b) - macro-F1(a) using the *same* resamples for both.

    Use this to compare two systems on one test set: the unpaired CIs above
    overlap far more often than the paired difference is uncertain.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    rng = np.random.default_rng(seed)

    if groups is None:
        index_pool = [np.arange(len(y_true))]
    else:
        groups = np.asarray(groups)
        index_pool = [np.flatnonzero(groups == g) for g in np.unique(groups)]

    n_units = len(index_pool)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picks = rng.integers(0, n_units, size=n_units)
        idx = np.concatenate([index_pool[p] for p in picks])
        deltas[i] = macro_f1(y_true[idx], b[idx]) - macro_f1(y_true[idx], a[idx])

    alpha = (1.0 - level) / 2.0
    return {
        "delta": macro_f1(y_true, b) - macro_f1(y_true, a),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
    }
