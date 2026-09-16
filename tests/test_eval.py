"""Metrics, calibration, and shortcut probes."""

from __future__ import annotations

import numpy as np
import pytest

from data.schema import DIMENSIONS, UNLABELED, Record
from eval.calibration import ece_per_dimension, expected_calibration_error
from eval.metrics import bootstrap_ci, evaluate_predictions, macro_f1, paired_bootstrap_delta
from eval.shortcuts import has_code_block, reveals_answer, shortcut_report, spearman


def test_macro_f1_perfect_and_empty():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert np.isnan(macro_f1([], []))


def test_macro_f1_counts_all_three_classes_even_if_unpredicted():
    # Predicting only the majority class must not score 1.0 just because the
    # other classes vanished from the average.
    score = macro_f1([2, 2, 2, 0], [2, 2, 2, 2])
    assert score < 0.5


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = y_true.copy()
    y_pred[:60] = (y_pred[:60] + 1) % 3

    out = bootstrap_ci(y_true, y_pred, n_boot=300, seed=0)
    assert out["ci_low"] <= out["macro_f1"] <= out["ci_high"]
    assert out["n"] == 300


def test_grouped_bootstrap_is_wider_than_ungrouped():
    # Responses to one dialogue are correlated; ignoring that understates the CI.
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 8)
    y_true = rng.integers(0, 3, size=320)
    y_pred = y_true.copy()
    y_pred[::4] = (y_pred[::4] + 1) % 3

    flat = bootstrap_ci(y_true, y_pred, n_boot=400, seed=0)
    grouped = bootstrap_ci(y_true, y_pred, groups=groups, n_boot=400, seed=0)
    assert (grouped["ci_high"] - grouped["ci_low"]) > (flat["ci_high"] - flat["ci_low"])


def test_evaluate_predictions_skips_unlabeled_entries():
    y_true = np.array([[2, 2, 2, 2], [UNLABELED, 0, 0, 0]])
    y_pred = np.array([[2, 2, 2, 2], [0, 0, 0, 0]])
    out = evaluate_predictions(y_true, y_pred, n_boot=50, seed=0)
    assert out["mistake_identification"]["n"] == 1
    assert out["mistake_location"]["n"] == 2
    assert set(DIMENSIONS).issubset(out)


def test_evaluate_predictions_rejects_wrong_shape():
    with pytest.raises(ValueError, match="dimension columns"):
        evaluate_predictions(np.zeros((3, 2), int), np.zeros((3, 2), int), n_boot=10)


def test_paired_delta_is_positive_when_b_is_better():
    y_true = np.array([0, 1, 2] * 40)
    worse = np.zeros_like(y_true)
    better = y_true.copy()
    out = paired_bootstrap_delta(y_true, worse, better, n_boot=200, seed=0)
    assert out["delta"] > 0
    assert out["ci_low"] > 0


def test_ece_zero_for_perfectly_calibrated_confident_predictions():
    probs = np.array([[0.0, 0.0, 1.0]] * 50)
    labels = np.full(50, 2)
    assert expected_calibration_error(probs, labels)["ece"] == pytest.approx(0.0, abs=1e-9)


def test_ece_large_for_confidently_wrong_predictions():
    probs = np.array([[0.0, 0.0, 1.0]] * 50)
    labels = np.zeros(50, dtype=int)
    out = expected_calibration_error(probs, labels)
    assert out["ece"] == pytest.approx(1.0, abs=1e-9)
    assert out["accuracy"] == 0.0


def test_every_point_falls_in_exactly_one_bin():
    # Confidence exactly 0.5 and exactly 1.0 must both be counted once.
    probs = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([0, 2])
    assert expected_calibration_error(probs, labels)["n"] == 2


def test_ece_per_dimension_shape():
    probs = np.full((10, len(DIMENSIONS), 3), 1 / 3)
    labels = np.zeros((10, len(DIMENSIONS)), dtype=int)
    out = ece_per_dimension(probs, labels)
    assert set(DIMENSIONS).issubset(out)
    assert "mean" in out


def test_spearman_handles_ties():
    binary = np.array([0, 0, 1, 1], dtype=float)
    target = np.array([1, 2, 3, 4], dtype=float)
    assert spearman(binary, target) == pytest.approx(0.894, abs=0.01)


def test_has_code_block_and_reveals_answer():
    assert has_code_block("try ```def f(): pass```")
    assert not has_code_block("What do you think the next step is?")
    assert reveals_answer("The answer is 42.")
    assert reveals_answer("Change it to x = 3")
    assert not reveals_answer("What happens if the list is empty?")


def test_shortcut_report_detects_a_length_shortcut():
    # A model whose prediction is purely a function of response length.
    records, preds = [], []
    for i in range(40):
        long = i % 2 == 0
        records.append(
            Record(
                dialogue_id=f"d{i}",
                tutor_response=("word " * 60) if long else "short",
                dialogue_context="ctx",
                labels=dict.fromkeys(DIMENSIONS, "Yes"),
            )
        )
        preds.append([2 if long else 0] * len(DIMENSIONS))

    report = shortcut_report(records, np.array(preds))
    length = report["actionability"]["length"]
    assert length["corr_pred"] > 0.9
    assert length["stump_agreement"] == pytest.approx(1.0)


def test_shortcut_report_rejects_length_mismatch(many_records):
    with pytest.raises(ValueError, match="predictions"):
        shortcut_report(many_records, np.zeros((3, len(DIMENSIONS)), dtype=int))
