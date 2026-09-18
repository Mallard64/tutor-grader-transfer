"""(c), (d), (e) for the TF-IDF + logistic regression grader.

The same three transfer questions as experiments/transfer.py, but with the
linear model instead of DeBERTa. It runs on a CPU in seconds, which is what
makes the k-curve (60 fits) tractable without a GPU.

  (c) fit on math, predict programming, no programming training at all
  (d) fit on math + k programming examples
  (e) fit on k programming examples only

For a linear model there is no "warm start", so (d) is pooled training rather
than fine-tuning. That is a real difference from the DeBERTa version and the
two are not interchangeable: DeBERTa's (d) adapts pretrained weights, this one
just adds rows to the training matrix.

A caveat specific to lexical features, and the thing to look at first in these
results: the vocabulary is fitted on the training domain. A model fitted on
grade-school word problems has never seen `def`, `index` or `loop`, so much of
a programming response lands out of vocabulary and its feature vector goes
sparse or empty. `oov_rate` in each result's extra measures this directly. If
it is high, (c) is not really measuring pedagogical transfer -- it is measuring
what a model does when handed a near-zero input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data.prepare import build_splits
from data.schema import DIMENSIONS, LABELS, Record, gold_matrix
from data.splits import sample_k_examples, split_records
from experiments.common import build_result, save_result, summarize
from experiments.linear_baseline import fit_and_predict
from utils import set_seed

# Fraction of the programming set held out as the shared evaluation set for
# (d) and (e). Fixed across k and seeds so the curve is comparable.
EVAL_FRACTION = 0.5


def programming_pool_and_eval(seed: int = 12345) -> tuple[list[Record], list[Record]]:
    """Split programming into (pool to draw k from, fixed evaluation set).

    Dialogue-grouped, and the split seed is deliberately independent of the
    experiment seed so every k and every run scores on the same responses.
    """
    splits = build_splits("programming")
    everything = splits["train"] + splits["val"] + splits["test"]
    halves = split_records(
        everything, fractions=(EVAL_FRACTION, 0.0, 1.0 - EVAL_FRACTION), seed=seed
    )
    return halves["test"], halves["train"]


def run_c(seed: int = 0) -> list[dict]:
    """(c) Fit on math, predict programming. Also reports the in-domain number.

    Evaluates on *every* programming record rather than a test split: nothing
    here trains on programming, so there is no leakage to guard against and
    holding data back would only widen the interval for nothing.
    """
    set_seed(seed)
    math = build_splits("math")
    prog = build_splits("programming")
    all_prog = prog["train"] + prog["val"] + prog["test"]

    results = []
    for name, evalset in (("math", math["test"]), ("programming", all_prog)):
        probs, preds, info = fit_and_predict(math["train"], math["val"], evalset, seed=seed)
        extra = dict(info)
        if name == "programming":
            extra["eval_note"] = "all programming records; zero-shot, no leakage"
        result = build_result(
            experiment="c",
            system="tfidf_math_trained",
            eval_domain=name,
            records=evalset,
            y_true=gold_matrix(evalset),
            y_pred=preds,
            probs=probs,
            k=0,
            seed=seed,
            extra=extra,
        )
        save_result(result)
        print(summarize(result) + f"   oov={info['oov_rate']:.1%}")
        results.append(result)
    return results


def _run_k(experiment: str, system: str, k: int, seed: int, with_math: bool) -> dict:
    set_seed(seed)
    pool, evalset = programming_pool_and_eval()
    if k > len(pool):
        raise ValueError(
            f"k={k} exceeds the programming pool ({len(pool)} responses). "
            "Annotate more data or lower k."
        )
    subset = sample_k_examples(pool, k, seed=seed) if k else []

    train: Sequence[Record]
    train = (list(build_splits("math")["train"]) if with_math else []) + list(subset)

    if not train:
        # (e) at k=0 has nothing to fit at all; fall back to the uniform prior.
        probs = np.full((len(evalset), len(DIMENSIONS), len(LABELS)), 1 / len(LABELS))
        preds = probs.argmax(axis=-1)
        info = {"note": "k=0 and no math data; uniform prior", "oov_rate": float("nan")}
    else:
        probs, preds, info = fit_and_predict(train, [], evalset, seed=seed)

    result = build_result(
        experiment=experiment,
        system=system,
        eval_domain="programming",
        records=evalset,
        y_true=gold_matrix(evalset),
        y_pred=preds,
        probs=probs,
        k=k,
        seed=seed,
        extra={**info, "n_train": len(train), "n_eval": len(evalset)},
    )
    save_result(result)
    print(summarize(result))
    return result


def run_d(k: int, seed: int = 0) -> dict:
    """(d) Math training data pooled with k programming examples."""
    return _run_k("d", "tfidf_math_plus_k", k, seed, with_math=True)


def run_e(k: int, seed: int = 0) -> dict:
    """(e) k programming examples only."""
    return _run_k("e", "tfidf_prog_only", k, seed, with_math=False)


if __name__ == "__main__":
    import sys

    run_c(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
