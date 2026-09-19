"""Record schema, label normalization, and JSONL round-tripping."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.schema import (
    DIMENSIONS,
    LABEL2ID,
    UNLABELED,
    Record,
    normalize_label,
    read_jsonl,
    write_jsonl,
)


def test_dimensions_and_labels_are_stable():
    # Guards the wire format: changing either invalidates every saved result.
    assert DIMENSIONS == (
        "mistake_identification",
        "mistake_location",
        "providing_guidance",
        "actionability",
    )
    assert LABEL2ID == {"No": 0, "To some extent": 1, "Yes": 2}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Yes", "Yes"),
        ("yes", "Yes"),
        ("  YES  ", "Yes"),
        ("To some extent", "To some extent"),
        ("to  some   extent", "To some extent"),
        ("No", "No"),
        (None, None),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_label_rejects_unknown():
    with pytest.raises(ValueError, match="Unrecognized label"):
        normalize_label("Maybe")


def test_record_fills_missing_dimensions_with_none():
    rec = Record(
        dialogue_id="d1",
        tutor_response="hi",
        dialogue_context="ctx",
        labels={"mistake_identification": "Yes"},
    )
    assert rec.labels["actionability"] is None
    assert not rec.is_labeled
    assert rec.label_ids() == [2, UNLABELED, UNLABELED, UNLABELED]


def test_record_is_labeled_and_label_ids():
    rec = Record(
        dialogue_id="d1",
        tutor_response="hi",
        dialogue_context="ctx",
        labels=dict.fromkeys(DIMENSIONS, "No"),
    )
    assert rec.is_labeled
    assert rec.label_ids() == [0, 0, 0, 0]


def test_response_id_defaults_to_dialogue_and_tutor():
    rec = Record("d1", "hi", "ctx", {}, tutor_name="GPT4")
    assert rec.response_id == "d1::GPT4"


def test_jsonl_round_trip(tmp_path, many_records):
    path = tmp_path / "recs.jsonl"
    assert write_jsonl(many_records, path) == len(many_records)

    loaded = read_jsonl(path)
    assert len(loaded) == len(many_records)
    assert loaded[0].labels == many_records[0].labels
    assert loaded[0].response_id == many_records[0].response_id
    assert loaded[0].domain == "programming"


def test_model_root_env_var_overrides_the_config(monkeypatch, tmp_path):
    """TGT_MODEL_ROOT must beat output_dir, not the other way round.

    Every shipped config sets output_dir, so if the YAML won, the env var would
    be ignored precisely when a --config is passed. That is how Colab runs are
    launched, and it silently wrote checkpoints to the ephemeral VM disk
    instead of the mounted Drive.
    """
    from utils import RunConfig

    cfg = RunConfig(name="run1", output_dir="models/runs")
    assert cfg.run_dir == Path("models/runs/run1")

    monkeypatch.setenv("TGT_MODEL_ROOT", str(tmp_path / "drive"))
    assert cfg.run_dir == tmp_path / "drive" / "run1"

    monkeypatch.delenv("TGT_MODEL_ROOT")
    assert cfg.run_dir == Path("models/runs/run1")
