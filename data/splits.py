"""Dialogue-grouped train/val/test splitting.

Every dialogue in MRBench is answered by ~8 different tutors, so splitting at
the response level would put near-identical contexts in both train and test and
inflate scores. All splits here are made over *dialogue groups*: every response
to a given dialogue lands in exactly one split.

A second, subtler leak: some conversation_ids share a base problem id (e.g.
'292861665' and '292861665_1' are two dialogues over the same source problem).
`group_key` collapses those onto the same group by default.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from data.schema import Record

SplitName = str


def group_key(dialogue_id: str, collapse_variants: bool = True) -> str:
    """Return the grouping key for a dialogue id.

    MRBench ids look like '221-362eb11a-...' or '292861665_1-9f0c...'. The text
    before the first '-' is the source problem id; a '_N' suffix marks a variant
    dialogue over that same problem.
    """
    head = dialogue_id.split("-", 1)[0]
    if collapse_variants:
        head = head.split("_", 1)[0]
    return head


def _stable_hash(value: str, seed: int) -> float:
    """Deterministic value in [0, 1) for a key, independent of PYTHONHASHSEED."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_groups(
    groups: Iterable[str],
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, SplitName]:
    """Assign each group to 'train' / 'val' / 'test'.

    Uses a stable hash rather than a shuffle so that adding new dialogues later
    does not reshuffle the existing assignment.
    """
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test)")
    total = sum(fractions)
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    train_hi = fractions[0]
    val_hi = fractions[0] + fractions[1]
    assignment: dict[str, SplitName] = {}
    for g in sorted(set(groups)):
        h = _stable_hash(g, seed)
        if h < train_hi:
            assignment[g] = "train"
        elif h < val_hi:
            assignment[g] = "val"
        else:
            assignment[g] = "test"
    return assignment


def split_records(
    records: Sequence[Record],
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    collapse_variants: bool = True,
) -> dict[SplitName, list[Record]]:
    """Split records into train/val/test with no dialogue group spanning splits."""
    keys = {r.dialogue_id: group_key(r.dialogue_id, collapse_variants) for r in records}
    assignment = split_groups(keys.values(), fractions=fractions, seed=seed)

    out: dict[SplitName, list[Record]] = {"train": [], "val": [], "test": []}
    for rec in records:
        out[assignment[keys[rec.dialogue_id]]].append(rec)
    return out


def assert_no_group_leakage(
    splits: dict[SplitName, list[Record]], collapse_variants: bool = True
) -> None:
    """Raise if any dialogue group appears in more than one split."""
    seen: dict[str, SplitName] = {}
    for name, recs in splits.items():
        for rec in recs:
            g = group_key(rec.dialogue_id, collapse_variants)
            if g in seen and seen[g] != name:
                raise AssertionError(
                    f"dialogue group {g!r} appears in both {seen[g]!r} and {name!r}"
                )
            seen[g] = name


def sample_k_examples(
    records: Sequence[Record],
    k: int,
    seed: int,
    by_dialogue: bool = True,
) -> list[Record]:
    """Draw k labeled examples for the few-shot curve.

    With `by_dialogue=True` we draw whole dialogues in random order and take
    their responses until we have k, which is what a real annotation budget
    looks like: you pick a dialogue and label the responses to it. Records are
    returned in a deterministic order for a given seed.
    """
    if k <= 0:
        return []
    if k > len(records):
        raise ValueError(f"requested k={k} but only {len(records)} records available")

    rng = random.Random(seed)
    if not by_dialogue:
        return sorted(rng.sample(list(records), k), key=lambda r: r.response_id)

    by_dlg: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        by_dlg[rec.dialogue_id].append(rec)
    dialogue_ids = sorted(by_dlg)
    rng.shuffle(dialogue_ids)

    picked: list[Record] = []
    for did in dialogue_ids:
        for rec in sorted(by_dlg[did], key=lambda r: r.response_id):
            if len(picked) >= k:
                return picked
            picked.append(rec)
    return picked
