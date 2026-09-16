"""Expected calibration error.

Transfer usually degrades confidence before it degrades accuracy, so we report
ECE alongside macro-F1: a math-trained grader can stay reasonably accurate on
programming while becoming badly overconfident, and that matters if the grader
is ever used to auto-filter responses.
"""

from __future__ import annotations

import numpy as np

from data.schema import DIMENSIONS, UNLABELED

N_BINS = 10


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = N_BINS,
) -> dict[str, float]:
    """Top-label ECE with equal-width confidence bins.

    Args:
        probs: [n, n_classes] predicted probabilities.
        labels: [n] true class indices.

    Returns ECE, max calibration error, mean confidence and mean accuracy.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "avg_confidence": float("nan"),
            "accuracy": float("nan"),
            "n": 0,
        }

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Include the left edge, and the right edge only in the last bin, so
        # every point falls in exactly one bin.
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        ece += (count / len(labels)) * gap
        mce = max(mce, gap)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "avg_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "n": int(len(labels)),
    }


def ece_per_dimension(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = N_BINS,
) -> dict[str, dict[str, float]]:
    """ECE for each dimension.

    Args:
        probs: [n, n_dimensions, n_classes]
        labels: [n, n_dimensions] with UNLABELED for missing entries.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    if probs.ndim != 3:
        raise ValueError(f"expected probs with shape [n, dims, classes], got {probs.shape}")

    out: dict[str, dict[str, float]] = {}
    for j, dim in enumerate(DIMENSIONS):
        mask = labels[:, j] != UNLABELED
        out[dim] = expected_calibration_error(probs[mask, j, :], labels[mask, j], n_bins=n_bins)

    finite = [out[d]["ece"] for d in DIMENSIONS if not np.isnan(out[d]["ece"])]
    out["mean"] = {"ece": float(np.mean(finite)) if finite else float("nan")}
    return out


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> list[dict[str, float]]:
    """Per-bin confidence/accuracy, for plotting a reliability diagram."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        count = int(in_bin.sum())
        bins.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": count,
                "confidence": float(confidence[in_bin].mean()) if count else float("nan"),
                "accuracy": float(correct[in_bin].mean()) if count else float("nan"),
            }
        )
    return bins
