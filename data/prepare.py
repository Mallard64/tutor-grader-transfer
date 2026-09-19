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


# Fraction of the programming set held out as the shared evaluation set for the
# k-curve. Fixed across k, seeds and model families so every point on the curve
# -- and both the DeBERTa and TF-IDF versions of it -- is scored on the same
# responses. Comparing arms measured on different subsets would be meaningless.
EVAL_FRACTION = 0.5
POOL_SPLIT_SEED = 12345


def programming_pool_and_eval(
    seed: int = POOL_SPLIT_SEED,
) -> tuple[list[Record], list[Record]]:
    """Split programming into (pool to draw k from, fixed evaluation set).

    Dialogue-grouped, and the split seed is independent of the experiment seed
    so the evaluation set does not move between runs.
    """
    splits = build_splits("programming")
    everything = splits["train"] + splits["val"] + splits["test"]
    halves = split_records(
        everything, fractions=(EVAL_FRACTION, 0.0, 1.0 - EVAL_FRACTION), seed=seed
    )
    return halves["test"], halves["train"]
