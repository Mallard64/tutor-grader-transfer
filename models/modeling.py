"""DeBERTa-v3 encoder with one 3-class head per pedagogical dimension.

One shared encoder, four linear heads. Sharing the encoder is the point: the
four dimensions are highly correlated (a response that does not identify the
mistake rarely locates it either), so four separate models would waste the
signal and quadruple training cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from data.schema import DIMENSIONS, LABELS, UNLABELED


class MultiHeadGrader(nn.Module):
    """Encoder + per-dimension classification heads.

    forward() returns {'loss', 'logits'} with logits shaped
    [batch, n_dimensions, n_classes], which is what HF Trainer expects when the
    module computes its own loss.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        n_dimensions: int = len(DIMENSIONS),
        n_classes: int = len(LABELS),
        dropout: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(n_dimensions)])
        # UNLABELED entries contribute no gradient, which is what lets us train
        # on partially annotated data without imputing labels.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=UNLABELED, weight=class_weights)

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over non-padding tokens.

        DeBERTa-v3 has no pretrained [CLS] pooler, and mean pooling is more
        stable than taking position 0 on small training sets.
        """
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = self.dropout(self.pool(out.last_hidden_state, attention_mask))
        logits = torch.stack([head(pooled) for head in self.heads], dim=1)

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits.reshape(-1, self.n_classes), labels.reshape(-1))
        return result

    # --- checkpointing -------------------------------------------------
    # The module is not a PreTrainedModel, so save/load explicitly rather than
    # relying on Trainer's save path.

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        (path / "grader_config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "n_dimensions": self.n_dimensions,
                    "n_classes": self.n_classes,
                    "dimensions": list(DIMENSIONS),
                    "labels": list(LABELS),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> MultiHeadGrader:
        path = Path(path)
        cfg = json.loads((path / "grader_config.json").read_text(encoding="utf-8"))
        model = cls(
            model_name=cfg["model_name"],
            n_dimensions=cfg["n_dimensions"],
            n_classes=cfg["n_classes"],
        )
        state = torch.load(path / "pytorch_model.bin", map_location=map_location)
        model.load_state_dict(state)
        return model
