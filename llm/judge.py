"""Zero-shot LLM judge: score a tutor response against the MRBench rubric.

This is experiment (b), a reference point for the trained graders. It is NOT a
labeling tool -- its outputs are predictions to be evaluated against human
labels, never written into a labeled dataset. The only labels in this repo come
from a human using labeling/app.py.

The judge sees exactly the rubric text in data/rubric.py, the same text the
human annotator sees, so that any gap between judge and human is not a gap in
what they were told.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

import numpy as np

from data.rubric import PROGRAMMING_NOTE, full_rubric_text
from data.schema import DIMENSIONS, LABEL2ID, LABELS, Record

JUDGE_MODEL = "claude-opus-5"

SYSTEM_PROMPT = f"""You are an expert annotator for a study of AI tutor quality. \
You will read a tutor-student conversation and one candidate tutor response, \
then rate that response on four dimensions.

Use exactly these labels: {", ".join(repr(x) for x in LABELS)}.

{full_rubric_text()}

Rate only the candidate tutor response, not the earlier turns. Judge against the \
rubric wording above, not against your own teaching preferences."""

# Structured output schema: forces one of the three labels per dimension, so
# there is no free-text parsing and no "unparseable" bucket to explain away.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {dim: {"type": "string", "enum": list(LABELS)} for dim in DIMENSIONS},
    "required": list(DIMENSIONS),
    "additionalProperties": False,
}

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def build_judge_prompt(record: Record) -> str:
    """Render one record as the judge's user turn."""
    note = f"\n\n{PROGRAMMING_NOTE}" if record.domain == "programming" else ""
    return (
        f"Conversation so far:\n---\n{record.dialogue_context}\n---\n\n"
        f"Candidate tutor response:\n---\n{record.tutor_response}\n---{note}\n\n"
        "Rate the candidate tutor response on all four dimensions."
    )


def judge_one(record: Record, model: str = JUDGE_MODEL, max_retries: int = 3) -> dict[str, str]:
    """Return {dimension: label} for one record."""
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        # Identical on every call; caching makes a full sweep
                        # over the test set substantially cheaper.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": build_judge_prompt(record)}],
                output_config={
                    "effort": "medium",
                    "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                },
            )
            if response.stop_reason == "refusal":
                raise RuntimeError("judge refused to rate this response")
            text = next(b.text for b in response.content if b.type == "text")
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001 - retry any transient API failure
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    raise RuntimeError(f"judge failed on {record.response_id}: {last_error}")


def judge_records(
    records: Sequence[Record],
    model: str = JUDGE_MODEL,
    verbose: bool = True,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Judge a list of records.

    Returns (preds [n, n_dimensions] of class indices, raw label dicts).
    """
    raw: list[dict[str, str]] = []
    preds = np.zeros((len(records), len(DIMENSIONS)), dtype=int)

    for i, rec in enumerate(records):
        labels = judge_one(rec, model=model)
        raw.append(labels)
        for j, dim in enumerate(DIMENSIONS):
            preds[i, j] = LABEL2ID[labels[dim]]
        if verbose and (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(records)}")

    return preds, raw
