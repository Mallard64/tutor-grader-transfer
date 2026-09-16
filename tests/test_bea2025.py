"""BEA 2025 dev-set loader, tested against a miniature of the real file."""

from __future__ import annotations

import pytest

from data.bea2025 import last_student_turn, load_bea2025, normalize_context, parse_devset
from data.schema import DIMENSIONS


def test_parse_devset_flattens_to_one_record_per_response(raw_devset):
    records = parse_devset(raw_devset)
    # 2 tutors on the first dialogue, 1 on the second.
    assert len(records) == 3
    assert {r.tutor_name for r in records} == {"Sonnet", "Phi3", "GPT4"}
    assert all(r.domain == "math" for r in records)


def test_parse_devset_maps_annotation_keys(raw_devset):
    records = parse_devset(raw_devset)
    sonnet = next(r for r in records if r.tutor_name == "Sonnet")
    assert sonnet.labels == dict.fromkeys(DIMENSIONS, "Yes")

    phi3 = next(r for r in records if r.tutor_name == "Phi3")
    assert phi3.labels["providing_guidance"] == "To some extent"
    assert phi3.labels["actionability"] == "No"


def test_all_parsed_records_are_fully_labeled(raw_devset):
    assert all(r.is_labeled for r in parse_devset(raw_devset))


def test_require_labels_drops_unannotated_responses(raw_devset):
    raw_devset[1]["tutor_responses"]["GPT4"].pop("annotation")

    assert len(parse_devset(raw_devset, require_labels=True)) == 2
    kept = parse_devset(raw_devset, require_labels=False)
    assert len(kept) == 3
    assert next(r for r in kept if r.tutor_name == "GPT4").labels["actionability"] is None


def test_normalize_context_strips_nbsp_and_blank_lines():
    ctx = normalize_context("Tutor: hi \n\n Student:  there  \n")
    assert ctx == "Tutor: hi\nStudent: there"
    assert " " not in ctx


def test_normalize_context_keeps_one_turn_per_line(raw_devset):
    ctx = normalize_context(raw_devset[0]["conversation_history"])
    assert ctx.count("\n") == 3
    assert ctx.startswith("Tutor:")


def test_last_student_turn(raw_devset):
    ctx = normalize_context(raw_devset[0]["conversation_history"])
    assert last_student_turn(ctx) == "One pound."


def test_last_student_turn_missing_returns_empty():
    assert last_student_turn("Tutor: only me") == ""


def test_load_from_file(devset_file):
    records = load_bea2025(path=devset_file, download=False)
    assert len(records) == 3
    assert records[0].dialogue_id == "221-362eb11a-f190-42a6"


def test_load_missing_file_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_bea2025(path=tmp_path / "nope.json", download=False)


def test_load_rejects_non_list_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"conversation_id": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON list"):
        load_bea2025(path=bad, download=False)
