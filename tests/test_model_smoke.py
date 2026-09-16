"""End-to-end smoke test for the multi-head grader.

Marked `slow` and skipped when torch/transformers are absent, so the fast data
tests still run without the training stack:

    uv run pytest -m slow        # just this
    uv run pytest -m "not slow"  # everything else

It uses a tiny random DeBERTa-v2 from the Hub rather than deberta-v3-base, so it
checks the wiring (Trainer integration, label plumbing, checkpoint round-trip)
in seconds without a 370MB download. It says nothing about accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.schema import DIMENSIONS, LABELS, Record

pytest.importorskip("torch", reason="training stack not installed")
pytest.importorskip("transformers", reason="training stack not installed")

pytestmark = pytest.mark.slow

TINY_MODEL = "hf-internal-testing/tiny-random-DebertaV2Model"


def _records(n: int, start: int = 0) -> list[Record]:
    return [
        Record(
            dialogue_id=f"d{(i + start) // 2}",
            tutor_response=f"tutor response number {i}",
            dialogue_context=f"Tutor: question {i}\nStudent: answer {i}",
            labels={dim: LABELS[(i + j) % 3] for j, dim in enumerate(DIMENSIONS)},
            domain="programming",
            response_id=f"r{i + start}",
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train once; several assertions share the checkpoint."""
    from models.train import train_grader
    from utils import RunConfig

    out = tmp_path_factory.mktemp("runs")
    cfg = RunConfig(
        name="smoke",
        model_name=TINY_MODEL,
        num_epochs=1.0,
        batch_size=4,
        output_dir=str(out),
        seed=0,
    )
    model, metrics = train_grader(cfg, _records(24), _records(8, start=100))
    return cfg, model, metrics


def test_compute_metrics_receives_labels(trained):
    """Guards TrainingArguments(label_names=...).

    Without it, Trainer cannot find the label column on a plain nn.Module and
    compute_metrics silently scores nothing.
    """
    _, _, metrics = trained
    assert "eval_mean_macro_f1" in metrics
    for dim in DIMENSIONS:
        assert f"eval_f1_{dim}" in metrics


def test_checkpoint_artifacts_written(trained):
    cfg, _, _ = trained
    for name in ("pytorch_model.bin", "grader_config.json", "config.yaml", "val_metrics.json"):
        assert (cfg.run_dir / name).exists(), f"missing {name}"


def test_round_trip_predict_shapes_and_normalization(trained):
    from models.train import load_grader, predict

    cfg, _, _ = trained
    val = _records(8, start=100)
    model, tokenizer = load_grader(cfg.run_dir)
    probs, preds = predict(model, val, tokenizer=tokenizer, batch_size=4)

    assert probs.shape == (len(val), len(DIMENSIONS), len(LABELS))
    assert preds.shape == (len(val), len(DIMENSIONS))
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)
    assert preds.min() >= 0
    assert preds.max() < len(LABELS)


def test_warm_start_from_checkpoint(trained, tmp_path):
    """The path experiment (d) depends on: resume from the math checkpoint."""
    from models.train import train_grader
    from utils import RunConfig

    cfg, _, _ = trained
    warm = RunConfig(
        name="smoke_warm",
        model_name=TINY_MODEL,
        num_epochs=1.0,
        batch_size=4,
        output_dir=str(tmp_path),
        seed=0,
        init_from=str(cfg.run_dir),
    )
    _, metrics = train_grader(warm, _records(8), _records(8, start=100))
    assert "eval_mean_macro_f1" in metrics


def test_unlabeled_dimensions_do_not_break_the_loss():
    """UNLABELED entries must be ignored, not treated as a class index."""
    import torch

    from models.modeling import MultiHeadGrader

    model = MultiHeadGrader(model_name=TINY_MODEL)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.tensor([[2, 1, 0, 2], [-100, -100, -100, -100]]),
    }
    out = model(**batch)
    assert out["logits"].shape == (2, len(DIMENSIONS), len(LABELS))
    assert torch.isfinite(out["loss"]), "loss went non-finite with unlabeled rows"
