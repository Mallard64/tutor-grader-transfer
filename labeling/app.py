"""Streamlit labeling tool for programming tutor responses.

Run:  uv run streamlit run labeling/app.py

Shows one candidate response at a time with the MRBench rubric text for each
dimension, and writes human labels to data/programming/labeled_responses.jsonl.

There is deliberately no "suggest labels" button and no model call anywhere in
this file. The whole experiment compares graders against human labels; labels
produced by a model would make that comparison circular.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.rubric import PROGRAMMING_NOTE, RUBRIC  # noqa: E402
from data.schema import DIMENSIONS, LABELS, Record, iter_jsonl  # noqa: E402

CANDIDATE_DIR = REPO_ROOT / "data" / "programming"
OUTPUT_PATH = CANDIDATE_DIR / "labeled_responses.jsonl"

st.set_page_config(page_title="Tutor response labeling", layout="wide")


def list_candidate_files() -> list[Path]:
    return sorted(CANDIDATE_DIR.glob("candidates_*.jsonl"))


@st.cache_data(show_spinner=False)
def load_candidates(path_str: str) -> list[dict]:
    return list(iter_jsonl(path_str))


def load_existing_labels() -> dict[str, dict]:
    """response_id -> labeled record, so relabeling overwrites cleanly."""
    if not OUTPUT_PATH.exists():
        return {}
    return {row["response_id"]: row for row in iter_jsonl(OUTPUT_PATH)}


def save_all(labeled: dict[str, dict]) -> None:
    """Rewrite the whole file, keyed by response_id.

    Rewriting rather than appending means a corrected label replaces the old
    one instead of leaving two conflicting rows for the same response.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for response_id in sorted(labeled):
            fh.write(json.dumps(labeled[response_id], ensure_ascii=False) + "\n")
    tmp.replace(OUTPUT_PATH)


# --- sidebar ------------------------------------------------------------

st.sidebar.header("Setup")

candidate_files = list_candidate_files()
if not candidate_files:
    st.error(
        f"No candidate files found in {CANDIDATE_DIR}/.\n\n"
        "Generate some first:\n\n"
        "`uv run python scripts/build_candidates.py --problems data/raw/tutorcode_problems.json`"
    )
    st.stop()

chosen_file = st.sidebar.selectbox("Candidate file", candidate_files, format_func=lambda p: p.name)
annotator = st.sidebar.text_input("Annotator name or initials", value="")

candidates = load_candidates(str(chosen_file))

if "labeled" not in st.session_state:
    st.session_state.labeled = load_existing_labels()
labeled: dict[str, dict] = st.session_state.labeled

only_unlabeled = st.sidebar.checkbox("Hide already-labeled responses", value=True)
queue = [c for c in candidates if not (only_unlabeled and c["response_id"] in labeled)]

st.sidebar.metric("Labeled", f"{len(labeled)} / {len(candidates)}")
if not queue:
    st.success("Nothing left in the queue. Uncheck 'Hide already-labeled' to review.")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0
st.session_state.idx = max(0, min(st.session_state.idx, len(queue) - 1))
item = queue[st.session_state.idx]

st.sidebar.write(f"Item {st.session_state.idx + 1} of {len(queue)}")
col_prev, col_next = st.sidebar.columns(2)
if col_prev.button("Previous", use_container_width=True):
    st.session_state.idx = max(0, st.session_state.idx - 1)
    st.rerun()
if col_next.button("Skip", use_container_width=True):
    st.session_state.idx = min(len(queue) - 1, st.session_state.idx + 1)
    st.rerun()

with st.sidebar.expander("Rubric note for programming"):
    st.write(PROGRAMMING_NOTE)

# --- main pane ----------------------------------------------------------

st.title("Label a tutor response")
st.caption(
    f"`{item['response_id']}` - the tutor model is hidden below to reduce bias; "
    "judge the response, not the model."
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Conversation so far")
    st.text(item["dialogue_context"])

with right:
    st.subheader("Tutor response to rate")
    st.info(item["tutor_response"])

    with st.expander("Show which model wrote this"):
        st.write(item.get("tutor_name", "unknown"))

st.divider()
st.subheader("Ratings")

existing = labeled.get(item["response_id"], {}).get("labels", {})
selections: dict[str, str | None] = {}

for dim in DIMENSIONS:
    rubric = RUBRIC[dim]
    st.markdown(f"**{rubric['title']}** - {rubric['question']}")
    for label in LABELS:
        st.caption(f"*{label}*: {rubric[label]}")

    prior = existing.get(dim)
    selections[dim] = st.radio(
        label=rubric["title"],
        options=LABELS,
        index=LABELS.index(prior) if prior in LABELS else None,
        key=f"{item['response_id']}::{dim}",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.write("")

notes = st.text_area(
    "Notes (optional - use this when the rubric felt ambiguous)",
    value=labeled.get(item["response_id"], {}).get("meta", {}).get("notes", ""),
)

missing = [d for d in DIMENSIONS if selections[d] is None]
if st.button("Save and next", type="primary", disabled=bool(missing)):
    record = Record(
        dialogue_id=item["dialogue_id"],
        tutor_response=item["tutor_response"],
        dialogue_context=item["dialogue_context"],
        labels=selections,
        domain="programming",
        tutor_name=item.get("tutor_name", "unknown"),
        response_id=item["response_id"],
        meta={
            **item.get("meta", {}),
            "annotator": annotator or "unknown",
            "labeled_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        },
    )
    labeled[item["response_id"]] = record.to_dict()
    save_all(labeled)
    st.session_state.idx = min(len(queue) - 1, st.session_state.idx + 1)
    st.rerun()

if missing:
    st.warning(f"Rate all four dimensions before saving. Missing: {', '.join(missing)}")

st.caption(f"Labels are written to `{OUTPUT_PATH.relative_to(REPO_ROOT)}`.")
