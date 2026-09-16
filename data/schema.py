"""Shared record schema for both domains (math and programming).

One record = one tutor response to one dialogue context, with up to four
3-class labels. Programming records use the exact same schema so that a model
trained on one domain can be evaluated on the other without any adaptation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# The four BEA 2025 / MRBench pedagogical dimensions, in canonical order.
DIMENSIONS: tuple[str, ...] = (
    "mistake_identification",
    "mistake_location",
    "providing_guidance",
    "actionability",
)

# 3-class label space, ordered worst -> best so that class index is ordinal.
LABELS: tuple[str, ...] = ("No", "To some extent", "Yes")
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: name for name, i in LABEL2ID.items()}

# Sentinel for "this dimension is unlabeled" (used for unlabeled candidates and
# ignored by the loss via ignore_index).
UNLABELED: int = -100

# Short codes used in the BEA leaderboards and in our results tables.
DIMENSION_CODES: dict[str, str] = {
    "mistake_identification": "MI",
    "mistake_location": "ML",
    "providing_guidance": "PG",
    "actionability": "AC",
}


def normalize_label(value: str | None) -> str | None:
    """Map raw annotation strings onto the canonical 3-class label space.

    The released files use 'Yes' / 'To some extent' / 'No', but casing and
    spacing vary across the annotation exports, so normalize defensively.
    """
    if value is None:
        return None
    key = " ".join(str(value).split()).strip().lower()
    aliases = {
        "yes": "Yes",
        "to some extent": "To some extent",
        "somewhat": "To some extent",
        "partially": "To some extent",
        "no": "No",
    }
    if key not in aliases:
        raise ValueError(f"Unrecognized label value: {value!r}")
    return aliases[key]


@dataclass
class Record:
    """A single (context, response) pair with its labels."""

    dialogue_id: str
    tutor_response: str
    dialogue_context: str
    labels: dict[str, str | None] = field(default_factory=dict)
    domain: str = "math"
    tutor_name: str = "unknown"
    response_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.response_id:
            self.response_id = f"{self.dialogue_id}::{self.tutor_name}"
        self.labels = {dim: normalize_label(self.labels.get(dim)) for dim in DIMENSIONS}

    @property
    def is_labeled(self) -> bool:
        return all(self.labels.get(dim) is not None for dim in DIMENSIONS)

    def label_ids(self) -> list[int]:
        """Labels as class indices; UNLABELED where a dimension is missing."""
        return [
            UNLABELED if self.labels.get(dim) is None else LABEL2ID[self.labels[dim]]
            for dim in DIMENSIONS
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Record:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    """Write records as JSONL; returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Record]:
    """Read a JSONL file of records."""
    out: list[Record] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Record.from_dict(json.loads(line)))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream raw dicts from a JSONL file without constructing Records."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def gold_matrix(records: Sequence[Record]) -> np.ndarray:
    """[n, n_dimensions] gold label ids, with UNLABELED where a label is missing.

    Lives here rather than in models/ so that the baselines and the reporting
    scripts can score predictions without importing torch.
    """
    return np.array([rec.label_ids() for rec in records], dtype=int)
