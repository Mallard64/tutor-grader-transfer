"""MRBench / BEA 2025 rubric text.

Single source of truth, used by three consumers so they cannot drift apart:
  * the Streamlit labeling tool (shown to the human annotator),
  * the zero-shot LLM judge prompt (experiment b),
  * the README / docs.

Wording follows the BEA 2025 shared task definitions of the four pedagogical
dimensions. The annotation guideline PDF lives in the upstream repo at
Human_Annotation_Guidelines_Resources/.
"""

from __future__ import annotations

from data.schema import DIMENSIONS

RUBRIC: dict[str, dict[str, str]] = {
    "mistake_identification": {
        "title": "Mistake Identification",
        "question": "Has the tutor identified that the student made a mistake?",
        "Yes": "The tutor clearly recognizes that the student's last response contains a mistake.",
        "To some extent": (
            "The tutor seems to notice something is off but is vague, hedged, or "
            "does not commit to there being a mistake."
        ),
        "No": (
            "The tutor does not recognize the mistake, or treats an incorrect "
            "student response as correct."
        ),
    },
    "mistake_location": {
        "title": "Mistake Location",
        "question": "Does the tutor's response point to the actual location of the mistake?",
        "Yes": "The tutor accurately points to where in the student's work the mistake occurs.",
        "To some extent": (
            "The tutor gestures at the region of the mistake but is imprecise, "
            "incomplete, or partly wrong about where it is."
        ),
        "No": ("The tutor does not indicate where the mistake is, or points to the wrong place."),
    },
    "providing_guidance": {
        "title": "Providing Guidance",
        "question": (
            "Does the tutor offer correct and relevant guidance, such as an "
            "explanation, elaboration, hint, or example?"
        ),
        "Yes": "The tutor gives guidance that is correct, relevant, and useful to the student.",
        "To some extent": (
            "The tutor gives some guidance, but it is vague, generic, incomplete, "
            "or only partially correct."
        ),
        "No": ("The tutor gives no guidance, or the guidance is irrelevant or incorrect."),
    },
    "actionability": {
        "title": "Actionability",
        "question": "Is it clear from the tutor's feedback what the student should do next?",
        "Yes": "The feedback makes the student's next step clear and concrete.",
        "To some extent": (
            "The feedback suggests a direction but leaves the next step unclear or underspecified."
        ),
        "No": (
            "The feedback gives the student nothing to act on, for example it "
            "just states the answer or ends the conversation."
        ),
    },
}

# A note shown to human labelers of *programming* responses. The rubric itself
# is unchanged across domains; only the surface form of the work differs.
PROGRAMMING_NOTE: str = (
    "The rubric is identical for programming. Read 'the student's work' as the "
    "student's buggy code plus anything they said about it, and 'the mistake' "
    "as the specific defect in that code. Revealing the corrected code is the "
    "programming analogue of revealing the answer: it usually scores high on "
    "guidance and low on actionability, but judge it against the wording above, "
    "not against your own teaching preferences."
)


def dimension_block(dim: str) -> str:
    """Render one dimension's rubric as plain text (for prompts and the UI)."""
    r = RUBRIC[dim]
    return (
        f"{r['title']}\n"
        f"{r['question']}\n"
        f"  - Yes: {r['Yes']}\n"
        f"  - To some extent: {r['To some extent']}\n"
        f"  - No: {r['No']}"
    )


def full_rubric_text() -> str:
    """All four dimensions as one plain-text block."""
    return "\n\n".join(dimension_block(dim) for dim in DIMENSIONS)
