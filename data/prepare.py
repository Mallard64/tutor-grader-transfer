"""One entry point both domains go through, so experiments never special-case a domain."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from data.bea2025 import load_bea2025
from data.programming import load_programming
from data.schema import Record
from data.splits import assert_no_group_leakage, split_records

DOMAINS = ("math", "programming")

# Fixed across every experiment. Changing these invalidates all cached results.
DEFAULT_FRACTIONS: tuple[float, float, float] = (0.7, 0.15, 0.15)
SPLIT_SEED: int = 12345


def load_domain(domain: str, path: str | Path | None = None) -> list[Record]:
    """Load all labeled records for a domain."""
    if domain == "math":
        return load_bea2025(path=path)
    if domain == "programming":
        return load_programming(path=path or "data/programming/labeled_responses.jsonl")
    raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")


def build_splits(
    domain: str,
    path: str | Path | None = None,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seed: int = SPLIT_SEED,
) -> dict[str, list[Record]]:
    """Load a domain and split it by dialogue.

    The split seed is deliberately *not* the experiment seed: the test set must
    be identical across all runs and all seeds, otherwise the learning curve
    measures split noise instead of sample efficiency.
    """
    records = load_domain(domain, path=path)
    splits = split_records(records, fractions=fractions, seed=seed)
    assert_no_group_leakage(splits)
    return splits
