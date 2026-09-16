"""Tokenization of (dialogue_context, tutor_response) pairs.

Encoding is identical for both domains -- that is what makes the transfer
question well-posed. If you change the input format, change it for both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from data.schema import Record


class GraderDataset(Dataset):
    """Encodes records as sentence pairs: context first, response second."""

    def __init__(
        self,
        records: Sequence[Record],
        tokenizer: Any,
        max_length: int = 512,
        with_labels: bool = True,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

        # Truncate from the left so the student's most recent turn -- the thing
        # the tutor is actually responding to -- always survives. The response
        # itself is short and is never truncated in practice.
        self.tokenizer.truncation_side = "left"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec.dialogue_context,
            rec.tutor_response,
            truncation="longest_first",
            max_length=self.max_length,
            padding=False,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        if self.with_labels:
            item["labels"] = torch.tensor(rec.label_ids(), dtype=torch.long)
        return item


def build_collator(tokenizer: Any) -> Any:
    """Dynamic padding collator that also stacks the [batch, n_dims] labels."""

    def collate(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = None
        if "labels" in features[0]:
            labels = torch.stack([f.pop("labels") for f in features])
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        if labels is not None:
            batch["labels"] = labels
        return batch

    return collate
