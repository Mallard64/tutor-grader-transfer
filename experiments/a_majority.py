"""(a) Majority-class baseline.

The floor every other system has to clear. It matters more than it looks: the
label distribution is skewed (in the BEA dev set 'Yes' is about 78% of Mistake
Identification), so accuracy looks respectable while macro-F1 does not. That
gap is exactly why the headline metric here is macro-F1.

The majority class is fit on the *training* split and applied to test, like any
other model -- reading it off the test set would leak.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data.prepare import build_splits
from data.schema import DIMENSIONS, UNLABELED, Record, gold_matrix
from experiments.common import build_result, save_result, summarize


def fit_majority(train_records: Sequence[Record]) -> np.ndarray:
    """Most frequent class per dimension, from the training split."""
    y = gold_matrix(train_records)
    majority = np.zeros(len(DIMENSIONS), dtype=int)
    for j in range(len(DIMENSIONS)):
        col = y[:, j]
        col = col[col != UNLABELED]
        majority[j] = np.bincount(col).argmax() if len(col) else 0
    return majority


def run(domain: str, seed: int = 0) -> dict:
    """Fit on `domain` train, evaluate on `domain` test."""
    splits = build_splits(domain)
    majority = fit_majority(splits["train"])

    test = splits["test"]
    y_true = gold_matrix(test)
    y_pred = np.tile(majority, (len(test), 1))

    result = build_result(
        experiment="a",
        system="majority",
        eval_domain=domain,
        records=test,
        y_true=y_true,
        y_pred=y_pred,
        probs=None,
        k=0,
        seed=seed,
        extra={"majority_class_ids": majority.tolist()},
    )
    save_result(result)
    print(summarize(result))
    return result


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "math")
