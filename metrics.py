from __future__ import annotations

from collections import deque

import numpy as np
import torch


def _components(binary: np.ndarray) -> list[np.ndarray]:
    binary = binary.astype(bool)
    visited = np.zeros_like(binary, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = binary.shape

    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            coords: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            comps.append(np.asarray(coords, dtype=np.int32))
    return comps


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.intersection = 0.0
        self.union = 0.0
        self.tp_pixels = 0.0
        self.fp_pixels = 0.0
        self.fn_pixels = 0.0
        self.correct_pixels = 0.0
        self.total_pixels = 0.0
        self.detected_objects = 0
        self.total_objects = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = (torch.sigmoid(logits) > self.threshold).detach().cpu().numpy().astype(np.uint8)
        gt = (target > 0.5).detach().cpu().numpy().astype(np.uint8)

        for pred_i, gt_i in zip(pred[:, 0], gt[:, 0]):
            pred_bool = pred_i.astype(bool)
            gt_bool = gt_i.astype(bool)
            inter = np.logical_and(pred_bool, gt_bool).sum()
            union = np.logical_or(pred_bool, gt_bool).sum()
            fp = np.logical_and(pred_bool, ~gt_bool).sum()
            fn = np.logical_and(~pred_bool, gt_bool).sum()

            self.intersection += float(inter)
            self.union += float(union)
            self.tp_pixels += float(inter)
            self.fp_pixels += float(fp)
            self.fn_pixels += float(fn)
            self.correct_pixels += float((pred_bool == gt_bool).sum())
            self.total_pixels += float(gt_bool.size)

            gt_components = _components(gt_bool)
            self.total_objects += len(gt_components)
            for comp in gt_components:
                ys, xs = comp[:, 0], comp[:, 1]
                if pred_bool[ys, xs].any():
                    self.detected_objects += 1

    def compute(self) -> dict[str, float]:
        miou = self.intersection / (self.union + 1e-6)
        precision = self.tp_pixels / (self.tp_pixels + self.fp_pixels + 1e-6)
        recall = self.tp_pixels / (self.tp_pixels + self.fn_pixels + 1e-6)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
        pix_acc = self.correct_pixels / (self.total_pixels + 1e-6)
        pd = self.detected_objects / (self.total_objects + 1e-6)
        fa = self.fp_pixels / (self.total_pixels + 1e-6)
        return {
            "mIoU": miou,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "pixAcc": pix_acc,
            "Pd": pd,
            "Fa": fa,
        }
