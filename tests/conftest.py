"""Shared fixtures. These use a hand-built miniature of the real dev-set JSON so
the data-loader tests run offline and in milliseconds."""

from __future__ import annotations

import json

import pytest

from data.schema import Record


@pytest.fixture
def raw_devset() -> list[dict]:
    """Two dialogues in the exact shape of mrbench_v3_devset.json.

    Mirrors the real file: non-breaking spaces in the history, one tutor
    missing from the second dialogue (as 'Novice' is in the real data).
    """
    return [
        {
            "conversation_id": "221-362eb11a-f190-42a6",
            "conversation_history": (
                "Tutor: Could you solve this? \n Student: I got 100. \n"
                " Tutor: How many pounds of meat? \n Student: One pound."
            ),
            "tutor_responses": {
                "Sonnet": {
                    "response": "Good, now find the total cost of the meat.",
                    "annotation": {
                        "Mistake_Identification": "Yes",
                        "Mistake_Location": "Yes",
                        "Providing_Guidance": "Yes",
                        "Actionability": "Yes",
                    },
                },
                "Phi3": {
                    "response": "The answer is 50.",
                    "annotation": {
                        "Mistake_Identification": "No",
                        "Mistake_Location": "No",
                        "Providing_Guidance": "To some extent",
                        "Actionability": "No",
                    },
                },
            },
        },
        {
            "conversation_id": "221_1-9f0cabcd-0000-1111",
            "conversation_history": "Tutor: Try again.\n Student: Is it 12?",
            "tutor_responses": {
                "GPT4": {
                    "response": "Not quite -- check your second step.",
                    "annotation": {
                        "Mistake_Identification": "Yes",
                        "Mistake_Location": "To some extent",
                        "Providing_Guidance": "To some extent",
                        "Actionability": "Yes",
                    },
                }
            },
        },
    ]


@pytest.fixture
def devset_file(tmp_path, raw_devset):
    path = tmp_path / "mrbench_v3_devset.json"
    path.write_text(json.dumps(raw_devset), encoding="utf-8")
    return path


@pytest.fixture
def many_records() -> list[Record]:
    """40 records over 10 dialogues, 4 tutors each."""
    out = []
    for d in range(10):
        for t in range(4):
            out.append(
                Record(
                    dialogue_id=f"{d}-abcd{d}",
                    tutor_response=f"response {d}-{t}",
                    dialogue_context=f"Tutor: q{d}\nStudent: a{d}",
                    labels={
                        "mistake_identification": "Yes",
                        "mistake_location": "No",
                        "providing_guidance": "To some extent",
                        "actionability": "Yes",
                    },
                    domain="programming",
                    tutor_name=f"tutor{t}",
                )
            )
    return out
