"""Shortcut probes.

A grader can score well by reading surface features instead of pedagogy:
long responses look helpful, code blocks look like guidance, and a response
that just reveals the answer looks confident. These probes ask how much of the
model's output is recoverable from three such features.

The headline number is not the raw correlation but the *gap*: how much more
strongly the feature correlates with the model's predictions than with the
human labels. Human labels correlate with length too -- longer responses really
do tend to contain more guidance -- so a nonzero correlation is not by itself
evidence of a shortcut. A correlation meaningfully larger than the human one is.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from data.schema import DIMENSIONS, UNLABELED, Record

_CODE_BLOCK = re.compile(r"```|\n {4}\S|\bdef \w+\(|\bfor \w+ in \b|;\s*$", re.MULTILINE)

# Cue phrases that mark handing over the answer rather than eliciting it.
_REVEAL_CUES = (
    "the answer is",
    "the correct answer",
    "the solution is",
    "it should be",
    "you should have",
    "the right answer",
    "correct code is",
    "here is the fix",
    "here's the fix",
    "here is the corrected",
    "here's the corrected",
    "replace it with",
    "change it to",
)
# "= 42" or "is 42" at the end of a clause: an asserted numeric result.
_ASSERTED_NUMBER = re.compile(r"(?:=|\bis\b|\bequals\b)\s*\$?-?\d[\d,]*(?:\.\d+)?")


def has_code_block(text: str) -> int:
    """1 if the response contains something that looks like code."""
    return int(bool(_CODE_BLOCK.search(text)))


def reveals_answer(text: str) -> int:
    """1 if the response appears to hand the student the answer.

    Heuristic, and deliberately a blunt one: an explicit reveal cue, or an
    asserted numeric/code result. It over-fires on responses that restate a
    student's correct intermediate value. Treat it as a coarse probe feature,
    not as a label.
    """
    lowered = text.lower()
    if any(cue in lowered for cue in _REVEAL_CUES):
        return 1
    if _ASSERTED_NUMBER.search(lowered):
        return 1
    if "```" in text and text.count("\n") >= 3:
        return 1
    return 0


def response_length(text: str) -> float:
    """Length in whitespace tokens."""
    return float(len(text.split()))


FEATURES = {
    "length": response_length,
    "code_block": has_code_block,
    "reveals_answer": reveals_answer,
}


def extract_features(records: Sequence[Record]) -> dict[str, np.ndarray]:
    """Compute every probe feature for a list of records."""
    return {
        name: np.array([fn(rec.tutor_response) for rec in records], dtype=float)
        for name, fn in FEATURES.items()
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, so ties (code_block is binary) are handled correctly."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank(np.asarray(x, dtype=float)), _rank(np.asarray(y, dtype=float)))


def _stump_agreement(feature: np.ndarray, target: np.ndarray, seed: int = 0) -> float:
    """Accuracy of a depth-2 tree that predicts `target` from the feature alone.

    If a single surface feature reproduces most of the model's predictions,
    the model is mostly reading that feature.
    """
    if len(np.unique(target)) < 2:
        return float("nan")
    clf = DecisionTreeClassifier(max_depth=2, random_state=seed)
    x = feature.reshape(-1, 1)
    clf.fit(x, target)
    return float((clf.predict(x) == target).mean())


def shortcut_report(
    records: Sequence[Record],
    y_pred: np.ndarray,
    y_true: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-dimension, per-feature shortcut diagnostics.

    Args:
        records: the evaluated records, in the same order as the predictions.
        y_pred: [n, n_dimensions] predicted class indices.
        y_true: optional [n, n_dimensions] gold labels, for the human baseline.

    Returns {dimension: {feature: {corr_pred, corr_gold, gap, stump_agreement}}}.
    """
    y_pred = np.asarray(y_pred)
    if len(records) != len(y_pred):
        raise ValueError(f"{len(records)} records but {len(y_pred)} predictions")
    features = extract_features(records)

    report: dict[str, dict[str, dict[str, float]]] = {}
    for j, dim in enumerate(DIMENSIONS):
        per_feature: dict[str, dict[str, float]] = {}
        for name, values in features.items():
            pred_col = y_pred[:, j]
            entry = {
                "corr_pred": spearman(values, pred_col),
                "stump_agreement": _stump_agreement(values, pred_col, seed=seed),
                "feature_mean": float(values.mean()),
            }
            if y_true is not None:
                gold = np.asarray(y_true)[:, j]
                mask = gold != UNLABELED
                entry["corr_gold"] = spearman(values[mask], gold[mask])
                entry["gap"] = entry["corr_pred"] - entry["corr_gold"]
            per_feature[name] = entry
        report[dim] = per_feature
    return report
