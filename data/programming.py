"""Programming-domain data: TutorCode-style problems in, Records out.

Two separate things live here:

1. `load_problems` reads buggy-student-code problems in TutorCode format. These
   are the *inputs* to response generation (scripts/build_candidates.py); they
   carry no tutor responses and no labels.

2. `load_programming` reads the JSONL produced by the labeling tool, which uses
   exactly the same `Record` schema as the math data. This is what the
   transfer experiments consume.

Field names for TutorCode releases vary, so `normalize_problem` accepts a few
aliases and fails loudly on anything it cannot map. If you are using a release
whose field names differ, add them to the alias table rather than reshaping
your copy of the data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.schema import DIMENSIONS, Record, read_jsonl

DEFAULT_PROBLEM_PATH = Path("data/raw/tutorcode_problems.json")
DEFAULT_LABELED_PATH = Path("data/programming/labeled_responses.jsonl")

# TutorCode-style field -> canonical field. Extend as needed.
_ALIASES: dict[str, str] = {
    "problem": "problem_statement",
    "problem_statement": "problem_statement",
    "problem_description": "problem_statement",
    "question": "problem_statement",
    "description": "problem_statement",
    "buggy_code": "buggy_code",
    "student_code": "buggy_code",
    "code": "buggy_code",
    "wrong_code": "buggy_code",
    "submission": "buggy_code",
    "reference_solution": "reference_solution",
    "correct_code": "reference_solution",
    "solution": "reference_solution",
    "test_cases": "test_cases",
    "tests": "test_cases",
    "bug_description": "bug_description",
    "error_description": "bug_description",
    "student_question": "student_question",
    "student_message": "student_question",
    "intent": "student_question",
    "problem_id": "problem_id",
    "id": "problem_id",
    "task_id": "problem_id",
    "language": "language",
    "lang": "language",
}

_REQUIRED = ("problem_id", "problem_statement", "buggy_code")


@dataclass
class Problem:
    """A buggy student submission awaiting a tutor response."""

    problem_id: str
    problem_statement: str
    buggy_code: str
    language: str = "python"
    reference_solution: str = ""
    test_cases: str = ""
    bug_description: str = ""
    student_question: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dialogue_context(self) -> str:
        """Render the problem as a dialogue context in the MRBench turn format.

        The grader sees programming data in the same shape as math data: a
        newline-separated transcript ending on a Student turn. Keeping this
        identical across domains is the whole point of the experiment, so it
        must not be customized per run.
        """
        student_turn = self.student_question.strip() or (
            "Here is my solution but it is not passing the tests."
        )
        lines = [
            "Tutor: Could you solve this problem? The problem is: "
            f"{self.problem_statement.strip()}",
            f"Student: {student_turn}",
            f"Student: ```{self.language}\n{self.buggy_code.strip()}\n```",
        ]
        return "\n".join(lines)


def normalize_problem(raw: dict[str, Any], index: int = 0) -> Problem:
    """Map a raw TutorCode-style dict onto a `Problem`."""
    mapped: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _ALIASES.get(key.strip().lower())
        if canonical is None:
            extra[key] = value
        elif canonical not in mapped or not mapped[canonical]:
            mapped[canonical] = value

    mapped.setdefault("problem_id", f"prob{index:04d}")
    if isinstance(mapped.get("test_cases"), (list, dict)):
        mapped["test_cases"] = json.dumps(mapped["test_cases"], indent=2)

    missing = [f for f in _REQUIRED if not str(mapped.get(f, "")).strip()]
    if missing:
        raise ValueError(
            f"problem at index {index} is missing {missing}; "
            f"available keys were {sorted(raw)}. Add an alias in data/programming.py."
        )

    return Problem(
        problem_id=str(mapped["problem_id"]),
        problem_statement=str(mapped["problem_statement"]),
        buggy_code=str(mapped["buggy_code"]),
        language=str(mapped.get("language") or "python"),
        reference_solution=str(mapped.get("reference_solution") or ""),
        test_cases=str(mapped.get("test_cases") or ""),
        bug_description=str(mapped.get("bug_description") or ""),
        student_question=str(mapped.get("student_question") or ""),
        extra=extra,
    )


def load_problems(path: str | Path = DEFAULT_PROBLEM_PATH) -> list[Problem]:
    """Load TutorCode-format problems from a JSON list or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place your TutorCode-format problems there; "
            "see data/programming.py for the expected fields."
        )
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        if isinstance(raw, dict):
            raw = list(raw.values())
    return [normalize_problem(item, i) for i, item in enumerate(raw)]


def load_programming(
    path: str | Path = DEFAULT_LABELED_PATH,
    require_labels: bool = True,
) -> list[Record]:
    """Load human-labeled programming records (same schema as the math data)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate candidates with scripts/build_candidates.py, "
            "then label them with: uv run streamlit run labeling/app.py"
        )
    records = read_jsonl(path)
    for rec in records:
        if rec.domain != "programming":
            raise ValueError(
                f"record {rec.response_id} has domain={rec.domain!r}, expected 'programming'"
            )
    if require_labels:
        records = [r for r in records if r.is_labeled]
    return records


def summarize(records: list[Record]) -> dict[str, Any]:
    """Counts used by the CLI scripts and tests."""
    per_dim: dict[str, dict[str, int]] = {}
    for dim in DIMENSIONS:
        counts: dict[str, int] = {}
        for rec in records:
            label = rec.labels.get(dim)
            key = label if label is not None else "unlabeled"
            counts[key] = counts.get(key, 0) + 1
        per_dim[dim] = counts
    return {
        "n_records": len(records),
        "n_dialogues": len({r.dialogue_id for r in records}),
        "tutors": sorted({r.tutor_name for r in records}),
        "labels": per_dim,
    }
