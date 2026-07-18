from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


@dataclass(frozen=True)
class ScoringHeadConfig:
    mode: str = "multitask"
    dropout: float = 0.15
    regression_weight: float = 0.65
    ordinal_weight: float = 0.35
    monotonicity_weight: float = 0.05
    raw_regression_weight: float = 0.60


class EssayScoringModel(nn.Module):
    def __init__(
        self,
        model_path: Path,
        head_config: ScoringHeadConfig,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if head_config.mode not in {"regression", "multitask"}:
            raise ValueError("mode must be 'regression' or 'multitask'")
        self.head_config = head_config
        self.backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(head_config.dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(head_config.dropout),
        )
        self.regression_head = nn.Linear(hidden_size, 1)
        self.ordinal_head = nn.Linear(hidden_size, 5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        ordinal_pos_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        cls_pool = hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.projection(torch.cat((cls_pool, mean_pool), dim=-1))

        regression_logit = self.regression_head(pooled).squeeze(-1)
        regression_score = 1.0 + 5.0 * torch.sigmoid(regression_logit)
        ordinal_logits = self.ordinal_head(pooled)
        ordinal_probabilities = torch.sigmoid(ordinal_logits)
        monotonic_probabilities = torch.cummin(ordinal_probabilities, dim=-1).values
        ordinal_score = 1.0 + monotonic_probabilities.sum(dim=-1)

        if self.head_config.mode == "regression":
            raw_score = regression_score
        else:
            weight = self.head_config.raw_regression_weight
            raw_score = weight * regression_score + (1.0 - weight) * ordinal_score

        result = {
            "raw_score": raw_score,
            "regression_score": regression_score,
            "ordinal_score": ordinal_score,
            "ordinal_logits": ordinal_logits,
        }
        if labels is not None:
            regression_loss = F.smooth_l1_loss(
                regression_score, labels, beta=0.5
            )
            if self.head_config.mode == "regression":
                loss = regression_loss
                ordinal_loss = torch.zeros_like(loss)
                monotonicity_loss = torch.zeros_like(loss)
            else:
                thresholds = torch.arange(1, 6, device=labels.device)
                ordinal_targets = (labels[:, None] > thresholds[None, :]).to(
                    ordinal_logits.dtype
                )
                ordinal_loss = F.binary_cross_entropy_with_logits(
                    ordinal_logits,
                    ordinal_targets,
                    pos_weight=ordinal_pos_weight,
                )
                monotonicity_loss = F.relu(
                    ordinal_probabilities[:, 1:]
                    - ordinal_probabilities[:, :-1]
                ).mean()
                loss = (
                    self.head_config.regression_weight * regression_loss
                    + self.head_config.ordinal_weight * ordinal_loss
                    + self.head_config.monotonicity_weight * monotonicity_loss
                )
            result.update(
                {
                    "loss": loss,
                    "regression_loss": regression_loss,
                    "ordinal_loss": ordinal_loss,
                    "monotonicity_loss": monotonicity_loss,
                }
            )
        return result


def ordinal_positive_weights(labels: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(1, 6, device=labels.device)
    positives = (labels[:, None] > thresholds[None, :]).sum(dim=0).float()
    negatives = len(labels) - positives
    weights = torch.sqrt(negatives / positives.clamp_min(1.0))
    return weights.clamp(0.5, 4.0)
