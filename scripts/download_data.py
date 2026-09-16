#!/usr/bin/env python
"""Download the BEA 2025 dev set and report what was loaded.

Usage:  uv run python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bea2025 import DEVSET_URL, download_devset, load_bea2025  # noqa: E402
from data.prepare import build_splits  # noqa: E402
from data.programming import summarize  # noqa: E402


def main() -> int:
    print(f"Downloading from {DEVSET_URL}")
    path = download_devset()
    print(f"Saved to {path} ({path.stat().st_size / 1e6:.1f} MB)\n")

    records = load_bea2025(path)
    info = summarize(records)
    print(f"Parsed {info['n_records']} labeled responses over {info['n_dialogues']} dialogues")
    print(f"Tutors: {', '.join(info['tutors'])}\n")
    for dim, counts in info["labels"].items():
        total = sum(counts.values())
        pretty = "  ".join(f"{k}={v} ({v / total:.0%})" for k, v in sorted(counts.items()))
        print(f"  {dim}: {pretty}")

    splits = build_splits("math")
    print("\nDialogue-grouped split:")
    for name, recs in splits.items():
        print(f"  {name}: {len(recs)} responses / {len({r.dialogue_id for r in recs})} dialogues")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
