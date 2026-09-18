#!/usr/bin/env python
"""Join pilot annotations to Socratic Debugging Benchmark dialogues.

    uv run python scripts/build_pilot.py --annotations path/to/annotations.jsonl

Downloads the referenced dialogue files, joins them by (source_file, turn_id),
and writes Records to data/programming/labeled_responses.jsonl -- the same path
and schema the transfer experiments already read.

The script refuses to stay quiet about label provenance. If the annotations are
model-generated it says so on every run and writes the method into each record's
meta, because a pilot file is exactly the kind of thing that quietly becomes the
headline dataset three weeks later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schema import DIMENSIONS, write_jsonl  # noqa: E402
from data.socratic import DEFAULT_DIALOGUE_DIR, build_records, provenance_summary  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "programming" / "labeled_responses.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--annotations", required=True, help="JSONL of annotation rows")
    p.add_argument("--dialogues", default=str(DEFAULT_DIALOGUE_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    text = Path(args.annotations).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    print(f"Read {len(rows)} annotation rows from {args.annotations}")

    records = build_records(rows, dialogue_dir=args.dialogues, download=not args.no_download)
    n = write_jsonl(records, args.out)
    print(f"Wrote {n} records to {args.out}\n")

    prov = provenance_summary(records)
    print(f"Label provenance: {prov['annotation_method']}")
    print(f"Flagged for human review: {prov['flagged_for_human_review']}/{prov['n_records']}")
    if not prov["human_labeled"]:
        print(
            "\n  !! These labels are NOT human-annotated. Fine for exercising the\n"
            "     pipeline. Not valid as ground truth: experiments (c)/(d)/(e) would\n"
            "     measure agreement with a model rather than with teachers, and (b)\n"
            "     -- the LLM judge -- would be scored against LLM labels, which is\n"
            "     circular. Replace with human labels before reporting any result.\n"
        )

    print(f"Dialogues: {len({r.dialogue_id for r in records})}")
    for dim in DIMENSIONS:
        counts = Counter(r.labels[dim] for r in records)
        print(f"  {dim:24} {dict(counts)}")

    lens = sorted(len(r.tutor_response.split()) for r in records)
    print(
        f"\nResponse length (words): min {lens[0]}, median {lens[len(lens) // 2]}, max {lens[-1]}"
    )
    print("\nSample record:")
    sample = records[0]
    print(f"  response_id : {sample.response_id}")
    print(f"  response    : {sample.tutor_response[:110]}")
    print(f"  context tail: ...{sample.dialogue_context[-110:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
