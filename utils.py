"""Small shared helpers: seeding, config loading, JSON IO."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# The five experiment seeds used everywhere. Fixed so results are reproducible.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# The labeled-example budgets on the learning curve.
K_VALUES: tuple[int, ...] = (0, 8, 16, 32, 64, 128)


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunConfig:
    """One training run. Mirrors the YAML files in configs/."""

    name: str
    model_name: str = "microsoft/deberta-v3-base"
    train_domain: str = "math"
    eval_domain: str = "math"
    k_programming: int = 0
    seed: int = 0
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: float = 4.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    class_weights: bool = False  # balanced per-dimension weights in the loss
    init_from: str = ""  # checkpoint dir to warm-start from (for d)
    output_dir: str = "models/runs"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = set(cls.__dataclass_fields__)
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(**{k: v for k, v in raw.items() if k in known}, extra=extra)

    def to_yaml(self, path: str | Path) -> None:
        payload = {k: v for k, v in self.__dict__.items() if k != "extra"}
        payload.update(self.extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
