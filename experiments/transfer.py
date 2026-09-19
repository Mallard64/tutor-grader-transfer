"""(c), (d), (e): the transfer experiments.

  (c) math-trained grader, evaluated zero-shot on programming
  (d) math-trained grader + k labeled programming examples
  (e) programming-only grader with k labeled programming examples

They live in one module because they differ in exactly two places -- what the
model is initialized from, and what it is fine-tuned on. Splitting them into
three files would triplicate the same twenty lines and invite them to drift.

The test set is the same programming test split in all three, and is fixed
across seeds (see data/prepare.SPLIT_SEED), so the learning curve measures
sample efficiency rather than split noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.prepare import build_splits
from data.schema import gold_matrix
from data.splits import sample_k_examples
from experiments.a_majority import fit_majority
from experiments.common import build_result, save_result, summarize
from models.train import load_grader, predict, train_grader
from utils import RunConfig, set_seed

MATH_RUN_PREFIX = "math_base"


def math_checkpoint(seed: int, config: RunConfig | None = None, force: bool = False) -> Path:
    """Train (or reuse) the math-only grader for a seed.

    Cached on disk: experiment (c) and every (d) run at that seed share one
    checkpoint, so the math model is trained five times total, not thirty-five.
    """
    cfg = config or RunConfig(name=f"{MATH_RUN_PREFIX}_s{seed}", seed=seed)
    cfg.name = f"{MATH_RUN_PREFIX}_s{seed}"
    cfg.train_domain = "math"
    cfg.seed = seed

    if cfg.run_dir.joinpath("pytorch_model.bin").exists() and not force:
        print(f"Reusing math checkpoint at {cfg.run_dir}")
        return cfg.run_dir

    math = build_splits("math")
    print(f"Training math grader (seed {seed}) on {len(math['train'])} responses...")
    train_grader(cfg, math["train"], math["val"])
    return cfg.run_dir


def _evaluate(model, tokenizer, records, experiment, system, k, seed, extra=None) -> dict:
    probs, preds = predict(model, records, tokenizer=tokenizer)
    result = build_result(
        experiment=experiment,
        system=system,
        eval_domain=records[0].domain if records else "unknown",
        records=records,
        y_true=gold_matrix(records),
        y_pred=preds,
        probs=probs,
        k=k,
        seed=seed,
        extra=extra,
    )
    save_result(result)
    print(summarize(result))
    return result


def run_c(seed: int, config: RunConfig | None = None) -> list[dict]:
    """(c) Train on math, evaluate on programming zero-shot.

    Also evaluates on the math test set: without the in-domain number, a low
    programming score is unreadable -- it could mean the transfer failed, or
    just that the grader is weak everywhere.
    """
    ckpt = math_checkpoint(seed, config)
    model, tokenizer = load_grader(ckpt)

    results = []
    math_test = build_splits("math")["test"]
    results.append(_evaluate(model, tokenizer, math_test, "c", "math_trained", 0, seed))

    # The in-domain half above stands on its own as a baseline, so a missing
    # programming set is a skip rather than a failure: the math reference
    # number can be produced before any programming labeling has happened.
    try:
        prog = build_splits("programming")
    except FileNotFoundError as exc:
        print(f"  skipping programming eval: {exc}")
        return results

    # Every programming record, not just the test split. Nothing in (c) trains
    # on programming, so there is no leakage to guard against and holding data
    # back would only widen the interval. Matches linear_transfer.run_c so the
    # DeBERTa and TF-IDF transfer numbers are measured on the same responses.
    all_prog = prog["train"] + prog["val"] + prog["test"]
    results.append(
        _evaluate(
            model,
            tokenizer,
            all_prog,
            "c",
            "math_trained",
            0,
            seed,
            {"eval_note": "all programming records; zero-shot, no leakage"},
        )
    )
    return results


def run_d(k: int, seed: int, config: RunConfig | None = None) -> dict:
    """(d) Math-trained, then fine-tuned on k programming examples."""
    set_seed(seed)
    prog = build_splits("programming")
    ckpt = math_checkpoint(seed, config)

    if k == 0:
        # Identical to (c) by construction; re-scored here so the curve has a
        # k=0 point without retraining anything.
        model, tokenizer = load_grader(ckpt)
        return _evaluate(model, tokenizer, prog["test"], "d", "math_plus_k", 0, seed)

    subset = sample_k_examples(prog["train"], k, seed=seed)
    cfg = config or RunConfig(name="tmp")
    cfg.name = f"d_math_plus_k{k}_s{seed}"
    cfg.train_domain = "programming"
    cfg.k_programming = k
    cfg.seed = seed
    cfg.init_from = str(ckpt)

    print(f"(d) fine-tuning math checkpoint on k={k} programming examples (seed {seed})")
    model, _ = train_grader(cfg, subset, prog["val"])
    _, tokenizer = load_grader(cfg.run_dir)
    return _evaluate(
        model, tokenizer, prog["test"], "d", "math_plus_k", k, seed, {"init_from": str(ckpt)}
    )


def run_e(k: int, seed: int, config: RunConfig | None = None) -> dict:
    """(e) Programming-only, k examples, no math pretraining.

    At k=0 there are no programming labels to learn from, so the only system
    available is a majority-class predictor -- and its majority class has to
    come from somewhere other than programming labels, so we take it from the
    math training split. This point is included so the curve starts somewhere
    honest, not because it is an interesting model.
    """
    set_seed(seed)
    prog = build_splits("programming")

    if k == 0:
        majority = fit_majority(build_splits("math")["train"])
        test = prog["test"]
        y_pred = np.tile(majority, (len(test), 1))
        result = build_result(
            experiment="e",
            system="prog_only",
            eval_domain="programming",
            records=test,
            y_true=gold_matrix(test),
            y_pred=y_pred,
            probs=None,
            k=0,
            seed=seed,
            extra={"note": "k=0 has no programming labels; majority-class predictor"},
        )
        save_result(result)
        print(summarize(result))
        return result

    subset = sample_k_examples(prog["train"], k, seed=seed)
    cfg = config or RunConfig(name="tmp")
    cfg.name = f"e_prog_only_k{k}_s{seed}"
    cfg.train_domain = "programming"
    cfg.k_programming = k
    cfg.seed = seed
    cfg.init_from = ""  # fresh DeBERTa, no math training

    print(f"(e) training programming-only on k={k} examples (seed {seed})")
    model, _ = train_grader(cfg, subset, prog["val"])
    _, tokenizer = load_grader(cfg.run_dir)
    return _evaluate(model, tokenizer, prog["test"], "e", "prog_only", k, seed)
