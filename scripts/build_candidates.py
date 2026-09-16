#!/usr/bin/env python
"""Generate unlabeled programming tutor responses from several models.

Input:  TutorCode-format problems (buggy student code + problem statement).
Output: unlabeled JSONL in the same Record schema as the math data, ready for
        human labeling with labeling/app.py.

This script never assigns labels. Every record it writes has all four label
fields set to null.

Usage:
    uv run python scripts/build_candidates.py \
        --problems data/raw/tutorcode_problems.json \
        --models claude-opus-5 claude-haiku-4-5 gpt-4o \
        --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.programming import load_problems  # noqa: E402
from data.schema import DIMENSIONS, Record, iter_jsonl  # noqa: E402
from llm.client import MODEL_REGISTRY, build_tutor_prompt, generate_response  # noqa: E402

DEFAULT_MODELS = ["claude-opus-5", "claude-haiku-4-5", "gpt-4o"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--problems", default="data/raw/tutorcode_problems.json")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"choose from: {', '.join(MODEL_REGISTRY)}",
    )
    p.add_argument("--out", default=None, help="output JSONL (default: timestamped)")
    p.add_argument("--limit", type=int, default=None, help="only use the first N problems")
    p.add_argument(
        "--resume", default=None, help="existing JSONL whose response_ids should be skipped"
    )
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    unknown = [m for m in args.models if m not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown model(s): {unknown}. Known: {sorted(MODEL_REGISTRY)}", file=sys.stderr)
        return 2
    if len(args.models) < 3:
        print(
            f"Warning: only {len(args.models)} model(s) selected. The candidate pool is "
            "meant to span at least 3 so that response style varies.",
            file=sys.stderr,
        )

    problems = load_problems(args.problems)
    if args.limit:
        problems = problems[: args.limit]
    print(f"Loaded {len(problems)} problems; generating with {len(args.models)} models.")

    done: set[str] = set()
    if args.resume:
        done = {row["response_id"] for row in iter_jsonl(args.resume)}
        print(f"Resuming: {len(done)} responses already generated.")

    out_path = Path(
        args.out or f"data/programming/candidates_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failures = 0
    # Append line by line rather than collecting in memory: generation is slow
    # and a rate-limit failure partway through should not lose completed work.
    with out_path.open("a", encoding="utf-8") as fh:
        for problem in problems:
            context = problem.as_dialogue_context()
            prompt = build_tutor_prompt(context)

            for model_name in args.models:
                spec = MODEL_REGISTRY[model_name]
                response_id = f"{problem.problem_id}::{spec.name}"
                if response_id in done:
                    continue

                try:
                    text = generate_response(spec, prompt)
                except Exception as exc:  # noqa: BLE001 - log and keep going
                    failures += 1
                    print(f"  FAIL {response_id}: {exc}", file=sys.stderr)
                    continue

                if not text:
                    failures += 1
                    print(f"  FAIL {response_id}: empty response", file=sys.stderr)
                    continue

                record = Record(
                    dialogue_id=problem.problem_id,
                    tutor_response=text,
                    dialogue_context=context,
                    labels={dim: None for dim in DIMENSIONS},  # human labels only
                    domain="programming",
                    tutor_name=spec.name,
                    response_id=response_id,
                    meta={
                        "model_id": spec.model_id,
                        "backend": spec.backend,
                        "language": problem.language,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
                written += 1

                if args.sleep:
                    time.sleep(args.sleep)

            print(f"{problem.problem_id}: {written} written so far")

    print(f"\nWrote {written} unlabeled responses to {out_path} ({failures} failures).")
    print("Next: uv run streamlit run labeling/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
