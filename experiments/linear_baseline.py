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


def fit_and_predict(
    train: Sequence[Record],
    val: Sequence[Record],
    test: Sequence[Record],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit on train, pick C per dimension on val, predict test.

    Returns (probs [n, dims, classes], preds [n, dims], info). Shared by the
    in-domain baseline and the transfer experiments so the two cannot drift.

    `val` may be empty, in which case C falls back to 1.0 -- there is nothing
    to select on, and inventing a selection criterion from train would just
    pick the least regularized model.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(featurize(train))
    x_test = vectorizer.transform(featurize(test))
    x_val = vectorizer.transform(featurize(val)) if len(val) else None

    y_train = gold_matrix(train)
    y_val = gold_matrix(val) if len(val) else None

    probs = np.zeros((len(test), len(DIMENSIONS), len(LABELS)))
    preds = np.zeros((len(test), len(DIMENSIONS)), dtype=int)
    chosen: dict[str, float] = {}

    for j, dim in enumerate(DIMENSIONS):
        train_mask = y_train[:, j] != UNLABELED
        if train_mask.sum() == 0 or len(np.unique(y_train[train_mask, j])) < 2:
            # Nothing to learn for this dimension; leave the uniform prior.
            probs[:, j, :] = 1.0 / len(LABELS)
            preds[:, j] = probs[:, j, :].argmax(axis=1)
            chosen[dim] = float("nan")
            continue

        best_c = 1.0
        if y_val is not None:
            val_mask = y_val[:, j] != UNLABELED
            best_score = -1.0
            for c in C_GRID:
                clf = LogisticRegression(
                    C=c, max_iter=2000, class_weight="balanced", random_state=seed
                )
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

    info = {
        "C_per_dimension": chosen,
        "n_features": int(x_train.shape[1]),
        "oov_rate": oov_rate(vectorizer, test),
    }
    return probs, preds, info


def oov_rate(vectorizer: TfidfVectorizer, records: Sequence[Record]) -> float:
    """Fraction of evaluation tokens absent from the fitted vocabulary.

    The number that decides whether a lexical model can transfer at all. A
    vocabulary fitted on grade-school word problems has never seen `def`,
    `loop` or `index`, and a response made entirely of unseen tokens becomes an
    all-zero feature vector -- the model then predicts the same class for every
    such input, regardless of what it says.
    """
    analyzer = vectorizer.build_analyzer()
    vocab = vectorizer.vocabulary_
    total = seen = 0
    for text in featurize(records):
        for token in analyzer(text):
            total += 1
            seen += token in vocab
    return 1.0 - (seen / total) if total else float("nan")


def run(domain: str = "math", seed: int = 0) -> dict:
    """Fit on train, pick C on val, score on test (in-domain baseline)."""
    set_seed(seed)
    splits = build_splits(domain)
    train, val, test = splits["train"], splits["val"], splits["test"]

    probs, preds, info = fit_and_predict(train, val, test, seed=seed)

    result = build_result(
        experiment="a2",
        system="tfidf_logreg",
        eval_domain=domain,
        records=test,
        y_true=gold_matrix(test),
        y_pred=preds,
        probs=probs,
        k=0,
        seed=seed,
        extra=info,
    )
    save_result(result)
    print(summarize(result))
    return result


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "math")
