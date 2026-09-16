"""Training and inference for the multi-head grader (HF Trainer)."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from data.schema import DIMENSIONS, UNLABELED, Record, gold_matrix  # noqa: F401
from eval.metrics import macro_f1
from models.dataset import GraderDataset, build_collator
from models.modeling import MultiHeadGrader
from utils import RunConfig, set_seed


def compute_metrics(eval_pred) -> dict[str, float]:
    """Macro-F1 per dimension plus the mean, for model selection on val."""
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = logits.argmax(axis=-1)

    out: dict[str, float] = {}
    scores = []
    for j, dim in enumerate(DIMENSIONS):
        mask = labels[:, j] != UNLABELED
        score = macro_f1(labels[mask, j], preds[mask, j]) if mask.any() else float("nan")
        out[f"f1_{dim}"] = score
        if not np.isnan(score):
            scores.append(score)
    out["mean_macro_f1"] = float(np.mean(scores)) if scores else 0.0
    return out


def _warmup_kwargs(config: RunConfig, n_train: int) -> dict[str, float | int]:
    """Express the warmup as whatever the installed transformers accepts.

    transformers 5.x dropped `warmup_ratio` and kept only `warmup_steps`, so
    convert when the ratio is unavailable rather than pinning the library.
    """
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": config.warmup_ratio}

    per_epoch = math.ceil(n_train / max(1, config.batch_size * config.gradient_accumulation_steps))
    total_steps = max(1, int(per_epoch * config.num_epochs))
    return {"warmup_steps": int(total_steps * config.warmup_ratio)}


def train_grader(
    config: RunConfig,
    train_records: Sequence[Record],
    val_records: Sequence[Record],
) -> tuple[MultiHeadGrader, dict]:
    """Fine-tune a grader and return (model, val metrics)."""
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if config.init_from:
        # Experiment (d): warm-start from the math-trained checkpoint.
        model = MultiHeadGrader.load(config.init_from)
    else:
        model = MultiHeadGrader(model_name=config.model_name)

    train_ds = GraderDataset(train_records, tokenizer, max_length=config.max_length)
    val_ds = GraderDataset(val_records, tokenizer, max_length=config.max_length)

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_dir / "hf"),
        seed=config.seed,
        data_seed=config.seed,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to=[],
        # MultiHeadGrader is a plain nn.Module, so Trainer cannot infer the
        # label column name; without this, compute_metrics receives no labels.
        label_names=["labels"],
        remove_unused_columns=False,
        **_warmup_kwargs(config, n_train=len(train_ds)),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=build_collator(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    val_metrics = trainer.evaluate()

    model.save(run_dir)
    tokenizer.save_pretrained(run_dir)
    config.to_yaml(run_dir / "config.yaml")
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    return model, val_metrics


@torch.no_grad()
def predict(
    model: MultiHeadGrader,
    records: Sequence[Record],
    tokenizer=None,
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs [n, dims, classes], preds [n, dims])."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model.eval().to(device)

    ds = GraderDataset(records, tokenizer, max_length=max_length, with_labels=False)
    collate = build_collator(tokenizer)

    all_probs: list[np.ndarray] = []
    for start in range(0, len(ds), batch_size):
        batch = collate([ds[i] for i in range(start, min(start + batch_size, len(ds)))])
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)["logits"]
        all_probs.append(torch.softmax(logits, dim=-1).float().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    return probs, probs.argmax(axis=-1)


def load_grader(path: str | Path) -> tuple[MultiHeadGrader, object]:
    """Load a saved grader and its tokenizer."""
    model = MultiHeadGrader.load(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
