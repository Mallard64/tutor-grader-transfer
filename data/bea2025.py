"""Loader for the BEA 2025 shared task / MRBench dev set.

Source: https://github.com/kaushal0494/UnifyingAITutorEvaluation

Verified layout of `BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json`
(checked against the released file):

    [
      {
        "conversation_id": "221-362eb11a-...",
        "conversation_history": "Tutor: ...\nStudent: ...\nTutor: ...",
        "tutor_responses": {
          "Sonnet": {
            "response": "...",
            "annotation": {
              "Mistake_Identification": "Yes",
              "Mistake_Location": "To some extent",
              "Providing_Guidance": "Yes",
              "Actionability": "No"
            }
          },
          ...
        }
      },
      ...
    ]

The dev set has 300 dialogues and 2476 annotated responses across 9 tutors
(8 tutors on every dialogue, plus a 'Novice' human tutor on 76 of them). The
released *test* set has the same shape but no `annotation` field, so the dev
set is the only labeled math data and we split it three ways ourselves.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from data.schema import DIMENSIONS, Record

DEVSET_URL = (
    "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/"
    "main/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json"
)
TESTSET_URL = (
    "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/"
    "main/BEA_Shared_Task_2025_Datasets/mrbench_v3_testset.json"
)

DEFAULT_RAW_DIR = Path("data/raw")
DEVSET_FILENAME = "mrbench_v3_devset.json"

# Annotation keys in the released file -> our canonical dimension names.
ANNOTATION_KEY_MAP: dict[str, str] = {
    "Mistake_Identification": "mistake_identification",
    "Mistake_Location": "mistake_location",
    "Providing_Guidance": "providing_guidance",
    "Actionability": "actionability",
}


def download_devset(dest_dir: str | Path = DEFAULT_RAW_DIR, force: bool = False) -> Path:
    """Download the dev set JSON if it is not already on disk. Returns its path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / DEVSET_FILENAME
    if dest.exists() and not force:
        return dest
    with urllib.request.urlopen(DEVSET_URL, timeout=120) as resp:  # noqa: S310
        payload = resp.read()
    dest.write_bytes(payload)
    return dest


def parse_devset(raw: list[dict[str, Any]], require_labels: bool = True) -> list[Record]:
    """Convert the raw dev-set JSON into flat Records (one per tutor response)."""
    records: list[Record] = []
    for conv in raw:
        dialogue_id = conv["conversation_id"]
        context = normalize_context(conv["conversation_history"])
        for tutor_name, payload in conv["tutor_responses"].items():
            annotation = payload.get("annotation") or {}
            labels = {DIMENSIONS[i]: None for i in range(len(DIMENSIONS))}
            for raw_key, canonical in ANNOTATION_KEY_MAP.items():
                if raw_key in annotation:
                    labels[canonical] = annotation[raw_key]
            rec = Record(
                dialogue_id=dialogue_id,
                tutor_response=payload["response"].strip(),
                dialogue_context=context,
                labels=labels,
                domain="math",
                tutor_name=tutor_name,
                response_id=f"{dialogue_id}::{tutor_name}",
            )
            if require_labels and not rec.is_labeled:
                continue
            records.append(rec)
    return records


def normalize_context(history: str) -> str:
    """Tidy the dialogue history string.

    The released histories use non-breaking spaces and inconsistent newlines
    around speaker turns. Normalize whitespace but keep one turn per line so
    that 'the student's last turn' stays identifiable.
    """
    text = history.replace(" ", " ").replace("\r\n", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def load_bea2025(
    path: str | Path | None = None,
    download: bool = True,
    require_labels: bool = True,
) -> list[Record]:
    """Load the BEA 2025 dev set as Records.

    Args:
        path: explicit path to the dev-set JSON; defaults to data/raw/.
        download: fetch the file if it is missing.
        require_labels: drop responses that lack a complete 4-dimension annotation.
    """
    if path is None:
        path = DEFAULT_RAW_DIR / DEVSET_FILENAME
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} not found. Run: uv run python scripts/download_data.py"
            )
        path = download_devset(path.parent)

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON list at {path}, got {type(raw).__name__}")
    return parse_devset(raw, require_labels=require_labels)


def last_student_turn(context: str) -> str:
    """Extract the student's most recent turn, used by the shortcut probes."""
    for line in reversed(context.split("\n")):
        if line.startswith("Student:"):
            return line[len("Student:") :].strip()
    return ""
