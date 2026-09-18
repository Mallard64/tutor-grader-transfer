"""Socratic Debugging Benchmark parsing and the annotation join."""

from __future__ import annotations

import pytest

from data.schema import DIMENSIONS
from data.socratic import build_record, parse_sections, parse_turns, render_context

SAMPLE = """<problem>
Write `count_words(sentence)` that returns the number of words.
</problem>
<bug_code>
1. def count_words(sentence):
2.    return 0
</bug_code>
<bug_desc>
Off-by-one: the last word is not counted.
</bug_desc>
<stu_desc>
My counts are off by one.
</stu_desc>
<dialogue>
User: Hi! my code doesn't work
Assistant: What seems to be the issue?
    <alt>Let's walk through your code line by line.
    <alt>When you run your code, what do you observe?
User: My output counts are off by one
Assistant: Does this happen with all inputs?
	<alt>Can you walk me through your code?
User: It happens with all of them so far
Assistant: Can you try perturbing the inputs?
    <alt>What happens for a trailing space?
User: Still the same
</dialogue>
"""


def _annotation(turn_id: int = 2) -> dict:
    return {
        "source_file": "19_50_word_counter_conversational_thread_1.txt",
        "problem_id": "19",
        "bug_id": "50",
        "turn_id": turn_id,
        "mistake_identification": {"label": "No", "evidence": "does not identify it"},
        "mistake_location": {"label": "No", "evidence": "points elsewhere"},
        "providing_guidance": {"label": "Yes", "evidence": "suggests an experiment"},
        "actionability": {"label": "Yes", "evidence": "concrete next step"},
        "annotation_method": "llm_assisted_v1",
        "annotator_confidence": "High",
        "flag_for_human_review": False,
    }


def test_parse_sections_extracts_tagged_blocks():
    s = parse_sections(SAMPLE)
    assert "count_words" in s["problem"]
    assert "return 0" in s["bug_code"]
    assert "Off-by-one" in s["bug_desc"]
    assert s["dialogue"].startswith("User: Hi!")


def test_alt_lines_are_not_turns():
    """The load-bearing assumption of the whole join.

    <alt> lines are alternative phrasings of the preceding Assistant turn. If
    they were counted as turns, every turn_id would resolve to the wrong
    response and every label would attach to the wrong text -- silently.
    """
    turns = parse_turns(parse_sections(SAMPLE)["dialogue"])
    assert [s for s, _ in turns] == [
        "Student",
        "Tutor",
        "Student",
        "Tutor",
        "Student",
        "Tutor",
        "Student",
    ]
    assert not any("walk through your code line by line" in t for _, t in turns)
    assert not any("trailing space" in t for _, t in turns)


def test_turn_id_indexes_tutor_turns_only():
    """turn_id counts Assistant turns, not all turns."""
    for turn_id, expected in enumerate(
        [
            "What seems to be the issue?",
            "Does this happen with all inputs?",
            "Can you try perturbing the inputs?",
        ]
    ):
        rec = build_record(_annotation(turn_id), SAMPLE)
        assert rec.tutor_response == expected


def test_context_ends_on_the_student_turn_being_answered():
    rec = build_record(_annotation(2), SAMPLE)
    lines = rec.dialogue_context.split("\n")
    assert lines[-1].startswith("Student:")
    assert "It happens with all of them so far" in lines[-1]
    # The rated response must not leak into its own context.
    assert "perturbing the inputs" not in rec.dialogue_context


def test_context_matches_the_math_turn_format():
    """Both domains must be encoded identically or transfer is not measurable."""
    ctx = render_context(parse_sections(SAMPLE), [])
    assert ctx.startswith("Tutor: Could you solve this problem?")
    assert "Student: ```python" in ctx
    for line in ctx.split("\n"):
        assert (
            line.startswith(("Tutor:", "Student:"))
            or not line.strip()
            or "```" in line
            or line[0].isdigit()
        )


def test_labels_and_provenance_are_carried_through():
    rec = build_record(_annotation(2), SAMPLE)
    assert rec.labels["mistake_identification"] == "No"
    assert rec.labels["actionability"] == "Yes"
    assert rec.is_labeled
    assert rec.domain == "programming"
    assert rec.meta["annotation_method"] == "llm_assisted_v1"
    assert rec.meta["turn_id"] == 2
    assert set(rec.meta["evidence"]) == set(DIMENSIONS)


def test_out_of_range_turn_id_raises_rather_than_silently_shifting():
    with pytest.raises(ValueError, match="only 3 tutor turns"):
        build_record(_annotation(99), SAMPLE)
