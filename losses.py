from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou.mean()


class FalseAlarmLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        background = 1.0 - target
        return (pred * background).sum() / (background.sum() + 1e-6)


class IRSTDLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 1.0, fa_weight: float = 0.2) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.fa_weight = fa_weight
        self.iou = SoftIoULoss()
        self.fa = FalseAlarmLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        iou = self.iou(logits, target)
        fa = self.fa(logits, target)
        total = self.bce_weight * bce + self.iou_weight * iou + self.fa_weight * fa
        return total, {
            "loss": float(total.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "iou_loss": float(iou.detach().cpu()),
            "fa_loss": float(fa.detach().cpu()),
        }
