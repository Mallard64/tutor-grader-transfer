"""Training and inference for the multi-head grader (HF Trainer)."""

from __future__ import annotations

import inspect
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from data.schema import DIMENSIONS, LABELS, UNLABELED, Record, gold_matrix  # noqa: F401
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


def balanced_class_weights(records: Sequence[Record]) -> np.ndarray:
    """Per-dimension inverse-frequency weights, shaped [n_dimensions, n_classes].

    Uses sklearn's "balanced" formula, n / (n_classes * count), computed
    separately per dimension because the skew differs: 'Yes' is 78% of
    mistake_identification but only 53% of actionability. A class absent from
    the training split gets weight 1.0 rather than infinity.
    """
    y = gold_matrix(records)
    weights = np.ones((len(DIMENSIONS), len(LABELS)), dtype=float)
    for j in range(len(DIMENSIONS)):
        col = y[:, j]
        col = col[col != UNLABELED]
        if len(col) == 0:
            continue
        counts = np.bincount(col, minlength=len(LABELS))
        present = counts > 0
        weights[j, present] = len(col) / (present.sum() * counts[present])
    return weights


def resolve_device(preferred: str = "auto") -> str:
    """Pick a compute device; TGT_DEVICE overrides the choice.

    MPS is included: it works for this model provided the encoder is fp32 (see
    MultiHeadGrader.__init__). A mixed-dtype model instead dies on MPS with a
    Metal assertion -- "Destination NDArray and Accumulator NDArray cannot have
    different datatype" -- which aborts the process rather than raising, so if
    that ever reappears, suspect dtype before suspecting the backend.
    """
    override = os.environ.get("TGT_DEVICE")
    if override:
        return override
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _supported(**kwargs) -> dict:
    """Drop TrainingArguments kwargs the installed transformers does not accept.

    The save/checkpoint surface has churned across versions (5.x dropped
    `save_safetensors`, for instance), and a hard failure here would be a
    version pin in disguise.
    """
    allowed = inspect.signature(TrainingArguments.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in allowed}


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

    weights = None
    if config.class_weights:
        weights = torch.tensor(balanced_class_weights(train_records), dtype=torch.float)

    if config.init_from:
        # Experiment (d): warm-start from the math-trained checkpoint.
        model = MultiHeadGrader.load(config.init_from)
    else:
        model = MultiHeadGrader(model_name=config.model_name, class_weights=weights)

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
        # Keep the best epoch on validation, not whatever the last epoch landed
        # on. Validation climbs late and unevenly here -- it can sit near the
        # majority floor for two epochs and then jump -- so the final weights
        # are close to an arbitrary draw from the tail of the curve.
        logging_steps=25,
        report_to=[],
        # MultiHeadGrader is a plain nn.Module, so Trainer cannot infer the
        # label column name; without this, compute_metrics receives no labels.
        label_names=["labels"],
        remove_unused_columns=False,
        # Trainer picks up CUDA/MPS on its own; this only forces CPU when
        # resolve_device() says so (e.g. TGT_DEVICE=cpu).
        use_cpu=resolve_device() == "cpu",
        **_supported(
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_mean_macro_f1",
            greater_is_better=True,
            save_safetensors=False,  # plain nn.Module, not a PreTrainedModel
        ),
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
    device = resolve_device() if device is None else device
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
