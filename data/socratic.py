"""Socratic Debugging Benchmark -> the repo's Record schema.

Source: https://github.com/taisazero/socratic-debugging-benchmark

Each dialogue file is tagged plain text:

    <problem>   task statement and example cases
    <bug_code>  the student's buggy code, line-numbered
    <bug_desc>  what is wrong with it
    <bug_fixes> how to fix it
    <unit_tests>
    <stu_desc>  what the student says is happening
    <dialogue>  User:/Assistant: turns, with indented <alt> lines giving
                alternative Assistant responses

The annotations we join against identify a rated response by (source_file,
turn_id), where turn_id is the 0-based index of the *Assistant* turn. `<alt>`
lines are alternatives to a turn, not turns of their own, and are skipped when
numbering -- getting this wrong silently shifts every label by one.

The rendered dialogue_context deliberately matches the math domain's format
(newline-separated `Tutor:` / `Student:` turns, ending on a Student turn), since
a grader trained on one domain has to read the other unchanged. Assistant maps
to Tutor and User maps to Student.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data.schema import DIMENSIONS, Record, normalize_label

RAW_BASE = (
    "https://raw.githubusercontent.com/taisazero/socratic-debugging-benchmark/main/"
    "socratic_debugging_benchmark/v2_sigcse/final_dataset"
)
DEFAULT_DIALOGUE_DIR = Path("data/raw/socratic")

_TAG = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
_SPEAKER = re.compile(r"^(User|Assistant):\s?(.*)$")


def download_dialogue(source_file: str, dest_dir: str | Path = DEFAULT_DIALOGUE_DIR) -> Path:
    """Fetch one dialogue file if it is not already on disk."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_file
    if dest.exists():
        return dest
    with urllib.request.urlopen(f"{RAW_BASE}/{source_file}", timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())
    return dest


def parse_sections(text: str) -> dict[str, str]:
    """Pull out the <tag>...</tag> sections.

    <dialogue> is handled separately because some files leave it unclosed.
    """
    out = {name: body.strip() for name, body in _TAG.findall(text)}
    if "dialogue" not in out:
        i = text.find("<dialogue>")
        if i >= 0:
            body = text[i + len("<dialogue>") :]
            out["dialogue"] = body.split("</dialogue>")[0].strip()
    return out


def parse_turns(dialogue: str) -> list[tuple[str, str]]:
    """Return [(speaker, text)] for the main turns, dropping <alt> lines.

    A turn continues across lines until the next `User:` / `Assistant:` line, so
    code pasted inside a turn stays attached to that turn.
    """
    turns: list[tuple[str, str]] = []
    for raw_line in dialogue.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("<alt>"):
            # <alt> is an alternative phrasing of the previous Assistant turn.
            continue
        match = _SPEAKER.match(stripped)
        if match:
            speaker = "Tutor" if match.group(1) == "Assistant" else "Student"
            turns.append((speaker, match.group(2).strip()))
        elif turns:
            turns[-1] = (turns[-1][0], (turns[-1][1] + "\n" + stripped).strip())
    return [(s, t) for s, t in turns if t]


def render_context(sections: dict[str, str], prior_turns: list[tuple[str, str]]) -> str:
    """Build the transcript the grader sees, in the math domain's turn format."""
    problem = " ".join(sections.get("problem", "").split())
    code = sections.get("bug_code", "").strip()
    student_desc = " ".join(sections.get("stu_desc", "").split())

    lines = [f"Tutor: Could you solve this problem? The problem is: {problem}"]
    if student_desc:
        lines.append(f"Student: {student_desc}")
    if code:
        lines.append(f"Student: ```python\n{code}\n```")
    lines.extend(f"{speaker}: {text}" for speaker, text in prior_turns)
    return "\n".join(lines)


def build_record(
    annotation: dict[str, Any],
    text: str,
    domain: str = "programming",
) -> Record:
    """Join one annotation row to its dialogue file and produce a Record."""
    sections = parse_sections(text)
    turns = parse_turns(sections.get("dialogue", ""))
    tutor_indices = [i for i, (speaker, _) in enumerate(turns) if speaker == "Tutor"]

    turn_id = int(annotation["turn_id"])
    if turn_id >= len(tutor_indices):
        raise ValueError(
            f"{annotation['source_file']}: turn_id {turn_id} but only "
            f"{len(tutor_indices)} tutor turns were parsed"
        )
    idx = tutor_indices[turn_id]

    labels: dict[str, str | None] = {}
    for dim in DIMENSIONS:
        value = annotation.get(dim)
        if isinstance(value, dict):
            value = value.get("label")
        labels[dim] = normalize_label(value) if value else None

    source = annotation["source_file"]

    return Record(
        dialogue_id=f"{annotation['problem_id']}_{annotation['bug_id']}",
        tutor_response=turns[idx][1],
        dialogue_context=render_context(sections, turns[:idx]),
        labels=labels,
        domain=domain,
        tutor_name="socratic_benchmark",
        response_id=f"{source}::turn{turn_id}",
        meta={
            "source_file": source,
            "turn_id": turn_id,
            "problem_id": annotation.get("problem_id"),
            "bug_id": annotation.get("bug_id"),
            "bug_desc": sections.get("bug_desc", ""),
            # Provenance. These labels are model-generated, not human, so they
            # must never be mistaken for ground truth downstream.
            "annotation_method": annotation.get("annotation_method", "unknown"),
            "annotator_confidence": annotation.get("annotator_confidence"),
            "flag_for_human_review": annotation.get("flag_for_human_review"),
            "evidence": {
                dim: annotation[dim].get("evidence")
                for dim in DIMENSIONS
                if isinstance(annotation.get(dim), dict)
            },
        },
    )


def build_records(
    annotations: Iterable[dict[str, Any]],
    dialogue_dir: str | Path = DEFAULT_DIALOGUE_DIR,
    download: bool = True,
) -> list[Record]:
    """Join a set of annotation rows to their dialogue files."""
    dialogue_dir = Path(dialogue_dir)
    records: list[Record] = []
    for annotation in annotations:
        source = annotation["source_file"]
        path = dialogue_dir / source
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"{path} not found and download=False")
            path = download_dialogue(source, dialogue_dir)
        records.append(build_record(annotation, path.read_text(encoding="utf-8")))
    return records


def provenance_summary(records: Iterable[Record]) -> dict[str, Any]:
    """Counts that make the label provenance impossible to overlook."""
    records = list(records)
    methods: dict[str, int] = {}
    for rec in records:
        key = str(rec.meta.get("annotation_method", "unknown"))
        methods[key] = methods.get(key, 0) + 1
    return {
        "n_records": len(records),
        "annotation_method": methods,
        "flagged_for_human_review": sum(1 for r in records if r.meta.get("flag_for_human_review")),
        "human_labeled": all(
            str(r.meta.get("annotation_method", "")).startswith("human") for r in records
        ),
    }
