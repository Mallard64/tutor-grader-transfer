"""Dialogue-grouped splitting: the property that matters is no leakage."""

from __future__ import annotations

import pytest

from data.splits import (
    assert_no_group_leakage,
    group_key,
    sample_k_examples,
    split_groups,
    split_records,
)


def test_group_key_strips_uuid_suffix():
    assert group_key("221-362eb11a-f190") == "221"


def test_group_key_collapses_variant_dialogues():
    # '292861665' and '292861665_1' are two dialogues over one source problem.
    assert group_key("292861665-abc") == group_key("292861665_1-def") == "292861665"


def test_group_key_can_keep_variants_separate():
    assert group_key("292861665_1-def", collapse_variants=False) == "292861665_1"


def test_split_records_puts_every_response_somewhere(many_records):
    splits = split_records(many_records)
    assert sum(len(v) for v in splits.values()) == len(many_records)
    assert set(splits) == {"train", "val", "test"}


def test_split_records_has_no_dialogue_leakage(many_records):
    splits = split_records(many_records)
    assert_no_group_leakage(splits)

    seen = {}
    for name, recs in splits.items():
        for rec in recs:
            seen.setdefault(rec.dialogue_id, name)
            assert seen[rec.dialogue_id] == name


def test_responses_to_one_dialogue_never_split(many_records):
    splits = split_records(many_records)
    for recs in splits.values():
        ids = [r.dialogue_id for r in recs]
        # Every dialogue contributes all 4 of its responses or none.
        for did in set(ids):
            assert ids.count(did) == 4


def test_split_is_deterministic(many_records):
    a = split_records(many_records, seed=7)
    b = split_records(many_records, seed=7)
    assert [r.response_id for r in a["test"]] == [r.response_id for r in b["test"]]


def test_different_seed_changes_assignment(many_records):
    a = split_records(many_records, seed=1)
    b = split_records(many_records, seed=2)
    assert [r.response_id for r in a["test"]] != [r.response_id for r in b["test"]]


def test_split_groups_rejects_bad_fractions():
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_groups(["a", "b"], fractions=(0.5, 0.3, 0.3))


def test_split_groups_roughly_respects_fractions():
    assignment = split_groups([f"g{i}" for i in range(2000)], fractions=(0.7, 0.15, 0.15))
    counts = {
        name: sum(1 for v in assignment.values() if v == name) for name in assignment.values()
    }
    assert 0.65 < counts["train"] / 2000 < 0.75


def test_assert_no_group_leakage_catches_a_leak(many_records):
    splits = split_records(many_records)
    # Force a leak: copy one train record into test.
    splits["test"].append(splits["train"][0])
    with pytest.raises(AssertionError, match="appears in both"):
        assert_no_group_leakage(splits)


def test_sample_k_examples_returns_exactly_k(many_records):
    assert len(sample_k_examples(many_records, 8, seed=0)) == 8
    assert sample_k_examples(many_records, 0, seed=0) == []


def test_sample_k_examples_is_seed_stable(many_records):
    a = [r.response_id for r in sample_k_examples(many_records, 12, seed=3)]
    b = [r.response_id for r in sample_k_examples(many_records, 12, seed=3)]
    c = [r.response_id for r in sample_k_examples(many_records, 12, seed=4)]
    assert a == b
    assert a != c


def test_sample_k_examples_draws_whole_dialogues_first(many_records):
    picked = sample_k_examples(many_records, 8, seed=0, by_dialogue=True)
    # 8 responses at 4 per dialogue should come from exactly 2 dialogues.
    assert len({r.dialogue_id for r in picked}) == 2


def test_sample_k_more_than_available_raises(many_records):
    with pytest.raises(ValueError, match="only"):
        sample_k_examples(many_records, 999, seed=0)
