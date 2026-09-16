"""(a2) TF-IDF + logistic regression baseline.

Sits between the majority floor and the DeBERTa grader, and answers a question
neither of them does: how much of this task is solvable from surface lexical
features alone? If a bag of words gets most of the way, the transformer is not
buying much pedagogical understanding.

It is also the only trained baseline that runs on a laptop in seconds, which
makes it the practical reference point when the RAM or GPU for a DeBERTa run is
not available.

Deliberately plain: word 1-2 grams, one multinomial logistic regression per
dimension, C chosen on the validation split. `class_weight="balanced"` is used
because the headline metric is macro-F1 and the labels are heavily skewed --
without it the model collapses onto 'Yes' and the comparison against the
majority baseline stops being informative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data.prepare import build_splits
from data.schema import DIMENSIONS, LABELS, UNLABELED, Record, gold_matrix
from eval.metrics import macro_f1
from experiments.common import build_result, save_result, summarize
from utils import set_seed

C_GRID = (0.25, 1.0, 4.0)


def featurize(records: Sequence[Record]) -> list[str]:
    """Context and response, separated by a marker token.

    The marker lets the model pick up response-only cues without discarding the
    context, which matters for mistake_location.
    """
    return [f"{r.dialogue_context} [RESPONSE] {r.tutor_response}" for r in records]


def run(domain: str = "math", seed: int = 0) -> dict:
    """Fit on train, pick C on val, score on test."""
    set_seed(seed)
    splits = build_splits(domain)
    train, val, test = splits["train"], splits["val"], splits["test"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(featurize(train))
    x_val = vectorizer.transform(featurize(val))
    x_test = vectorizer.transform(featurize(test))

    y_train, y_val, y_test = gold_matrix(train), gold_matrix(val), gold_matrix(test)

    probs = np.zeros((len(test), len(DIMENSIONS), len(LABELS)))
    preds = np.zeros((len(test), len(DIMENSIONS)), dtype=int)
    chosen: dict[str, float] = {}

    for j, dim in enumerate(DIMENSIONS):
        train_mask = y_train[:, j] != UNLABELED
        val_mask = y_val[:, j] != UNLABELED

        best_c, best_score = C_GRID[0], -1.0
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed)
            clf.fit(x_train[train_mask], y_train[train_mask, j])
            score = macro_f1(y_val[val_mask, j], clf.predict(x_val[val_mask]))
            if score > best_score:
                best_c, best_score = c, score
        chosen[dim] = best_c

        clf = LogisticRegression(
            C=best_c, max_iter=2000, class_weight="balanced", random_state=seed
        )
        clf.fit(x_train[train_mask], y_train[train_mask, j])

        # classes_ may omit a class absent from training; scatter the predicted
        # columns into the full 3-class layout so ECE stays well defined.
        p = clf.predict_proba(x_test)
        for col, cls in enumerate(clf.classes_):
            probs[:, j, int(cls)] = p[:, col]
        preds[:, j] = probs[:, j, :].argmax(axis=1)

    result = build_result(
        experiment="a2",
        system="tfidf_logreg",
        eval_domain=domain,
        records=test,
        y_true=y_test,
        y_pred=preds,
        probs=probs,
        k=0,
        seed=seed,
        extra={"C_per_dimension": chosen, "n_features": int(x_train.shape[1])},
    )
    save_result(result)
    print(summarize(result))
    return result


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "math")
