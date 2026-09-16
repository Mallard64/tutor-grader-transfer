"""Programming-domain loaders: TutorCode normalization and labeled records."""

from __future__ import annotations

import json

import pytest

from data.programming import Problem, load_problems, load_programming, normalize_problem
from data.schema import DIMENSIONS, Record, write_jsonl

RAW_TUTORCODE = {
    "task_id": "p001",
    "problem_description": "Return the sum of a list.",
    "buggy_code": "def total(xs):\n    return xs[0]",
    "reference_solution": "def total(xs):\n    return sum(xs)",
    "tests": [{"input": "[1,2]", "output": "3"}],
    "lang": "python",
}


def test_normalize_problem_maps_aliases():
    p = normalize_problem(RAW_TUTORCODE)
    assert p.problem_id == "p001"
    assert p.problem_statement == "Return the sum of a list."
    assert p.language == "python"
    assert "sum(xs)" in p.reference_solution


def test_normalize_problem_serializes_structured_tests():
    p = normalize_problem(RAW_TUTORCODE)
    assert json.loads(p.test_cases) == RAW_TUTORCODE["tests"]


def test_normalize_problem_keeps_unknown_fields_in_extra():
    p = normalize_problem({**RAW_TUTORCODE, "difficulty": "easy"})
    assert p.extra == {"difficulty": "easy"}


def test_normalize_problem_generates_id_when_absent():
    raw = {k: v for k, v in RAW_TUTORCODE.items() if k != "task_id"}
    assert normalize_problem(raw, index=7).problem_id == "prob0007"


def test_normalize_problem_reports_missing_required_fields():
    with pytest.raises(ValueError, match="buggy_code"):
        normalize_problem({"task_id": "p", "problem": "do a thing"})


def test_dialogue_context_matches_mrbench_turn_format():
    ctx = normalize_problem(RAW_TUTORCODE).as_dialogue_context()
    lines = ctx.split("\n")
    assert lines[0].startswith("Tutor:")
    assert lines[1].startswith("Student:")
    # Must end on a student turn, like the math contexts do.
    assert "Student:" in ctx.rsplit("Tutor:", 1)[-1] or ctx.count("Tutor:") == 1
    assert "```python" in ctx


def test_dialogue_context_uses_student_question_when_present():
    p = Problem("p1", "stmt", "code", student_question="Why is my loop off by one?")
    assert "Why is my loop off by one?" in p.as_dialogue_context()


def test_load_problems_reads_json_list(tmp_path):
    path = tmp_path / "problems.json"
    path.write_text(json.dumps([RAW_TUTORCODE, RAW_TUTORCODE]), encoding="utf-8")
    assert len(load_problems(path)) == 2


def test_load_problems_reads_jsonl(tmp_path):
    path = tmp_path / "problems.jsonl"
    path.write_text("\n".join(json.dumps(RAW_TUTORCODE) for _ in range(3)), encoding="utf-8")
    assert len(load_problems(path)) == 3


def test_load_problems_missing_file_points_at_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="TutorCode-format"):
        load_problems(tmp_path / "nope.json")


def test_load_programming_filters_unlabeled(tmp_path, many_records):
    unlabeled = Record(
        dialogue_id="d99",
        tutor_response="?",
        dialogue_context="ctx",
        labels={},
        domain="programming",
    )
    path = tmp_path / "labeled.jsonl"
    write_jsonl([*many_records, unlabeled], path)

    assert len(load_programming(path)) == len(many_records)
    assert len(load_programming(path, require_labels=False)) == len(many_records) + 1


def test_load_programming_rejects_wrong_domain(tmp_path):
    rec = Record("d1", "r", "c", dict.fromkeys(DIMENSIONS, "Yes"), domain="math")
    path = tmp_path / "wrong.jsonl"
    write_jsonl([rec], path)
    with pytest.raises(ValueError, match="expected 'programming'"):
        load_programming(path)


def test_load_programming_missing_file_points_at_the_tools(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_candidates"):
        load_programming(tmp_path / "nope.jsonl")
