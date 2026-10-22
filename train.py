from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
try:
    from torch import amp
except ImportError:  # PyTorch 1.x exposes GradScaler under torch.cuda.amp.
    from torch.cuda import amp  # type: ignore[no-redef]
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from datasets import NUDTMIRSDTDataset, TemporalSegmentationDataset
from losses import IRSTDLoss
from metrics import SegmentationMetrics
from model import TFGNetV101, TFGNetV9
from utils import save_json, set_seed


try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    ndimage = None


def parse_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    width, height = value.lower().split("x")
    return int(width), int(height)

def is_nudt_split(data_root: str, split: str | None) -> bool:
    root_name = Path(data_root).name.lower()
    if root_name == "nudt-mirsdt":
        return True
    if split is None:
        return False
    split_path = Path(split)
    if not split_path.exists():
        return False
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().replace("\\", "/")
        if line and not line.startswith("#"):
            return "/Mix/" in line and line.lower().endswith(".mat")
    return False

def load_checkpoint_state(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model": checkpoint}


def load_pretrained(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool,
) -> tuple[int, list[str], list[str], list[str]]:
    if strict:
        result = model.load_state_dict(state_dict, strict=True)
        return len(model.state_dict()), list(result.missing_keys), list(result.unexpected_keys), []

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = [
        key
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    result = model.load_state_dict(compatible, strict=False)
    unexpected = [key for key in state_dict if key not in model_state]
    return len(compatible), list(result.missing_keys), unexpected, skipped

def gaussian_center_target(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return mask
    radius = max(int(round(sigma * 3.0)), 1)
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.max().clamp_min(1e-6)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    heat = F.conv2d(mask, kernel, padding=radius).clamp(0.0, 1.0)
    max_per_sample = heat.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return heat / max_per_sample


def objectness_target(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = 1.0 + target * pos_weight
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)

def parse_thresholds(value: str) -> list[float]:
    if not value or value.lower() == "none":
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


DIAGNOSTIC_KEYS = (
    "temporal_confidence",
    "temporal_anomaly_energy",
    "temporal_center_residual",
    "mean_recover_delta",
    "mean_suppress_delta",
    "recover_strength",
    "suppress_strength",
)


def make_grad_scaler(
    enabled: bool,
    init_scale: float,
    growth_interval: int,
) -> amp.GradScaler:
    try:
        return amp.GradScaler(
            device="cuda",
            enabled=enabled,
            init_scale=init_scale,
            growth_interval=growth_interval,
        )
    except TypeError:
        return amp.GradScaler(
            enabled=enabled,
            init_scale=init_scale,
            growth_interval=growth_interval,
        )


def topk_mean(values: torch.Tensor, ratio: float, minimum: int = 1) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    count = max(int(math.ceil(values.numel() * ratio)), minimum)
    count = min(count, values.numel())
    return torch.topk(values, count, sorted=False).values.mean()


def foreground_recovery_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    coarse_probability: torch.Tensor,
    negative_ratio: float,
) -> torch.Tensor:
    target = target.float()
    positive = target > 0.5
    negative = ~positive
    positive_loss = F.softplus(-logits.float())
    miss_weight = 1.0 + 2.0 * (1.0 - coarse_probability.detach())
    if positive.any():
        positive_term = (
            positive_loss[positive] * miss_weight[positive]
        ).mean()
        minimum = max(int(positive.sum().item() * 4), 1)
    else:
        positive_term = logits.sum() * 0.0
        minimum = 1
    negative_term = topk_mean(
        F.softplus(logits.float())[negative],
        ratio=negative_ratio,
        minimum=minimum,
    )
    return positive_term + 0.25 * negative_term


def suppression_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    coarse_probability: torch.Tensor,
    candidate_ratio: float,
    coarse_threshold: float,
) -> torch.Tensor:
    target = target.float()
    foreground = target > 0.5
    background = ~foreground
    candidate = (
        background
        & (coarse_probability.detach() > coarse_threshold)
    )
    suppress_background = F.softplus(-logits.float())
    if candidate.any():
        background_term = suppress_background[candidate].mean()
    else:
        background_term = topk_mean(
            (
                suppress_background
                * coarse_probability.detach()
            )[background],
            ratio=candidate_ratio,
        )
    if foreground.any():
        protection_term = F.softplus(logits.float())[foreground].mean()
    else:
        protection_term = logits.sum() * 0.0
    return background_term + 2.0 * protection_term


def safe_hard_negative_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ratio: float,
    exclusion_kernel: int,
) -> torch.Tensor:
    if exclusion_kernel % 2 == 0:
        exclusion_kernel += 1
    exclusion = F.max_pool2d(
        target.float(),
        kernel_size=exclusion_kernel,
        stride=1,
        padding=exclusion_kernel // 2,
    )
    valid_background = exclusion < 0.5
    losses = []
    for sample_logits, sample_valid in zip(logits.float(), valid_background):
        values = F.softplus(sample_logits[sample_valid])
        losses.append(topk_mean(values, ratio=ratio))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def positive_teacher_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    background_weight: float,
) -> torch.Tensor:
    teacher_probability = torch.sigmoid(teacher_logits.detach().float())
    target = target.float()
    positive_weight = teacher_probability * target
    positive = (
        F.softplus(-student_logits.float()) * positive_weight
    ).sum() / (positive_weight.sum() + 1e-6)
    negative_weight = (
        (1.0 - teacher_probability) * (1.0 - target)
    )
    negative = (
        F.softplus(student_logits.float()) * negative_weight
    ).sum() / (negative_weight.sum() + 1e-6)
    return positive + background_weight * negative


def compute_loss(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    criterion: IRSTDLoss,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_loss, base_parts = criterion(outputs["logits"], mask)
    coarse_loss, _ = criterion(outputs["coarse_logits"], mask)
    coarse_probability = torch.sigmoid(outputs["coarse_logits"].float())
    if args.no_correction:
        recovery = outputs["logits"].sum() * 0.0
        suppression = outputs["logits"].sum() * 0.0
    else:
        recovery = foreground_recovery_loss(
            outputs["recover_logits"],
            mask,
            coarse_probability,
            negative_ratio=args.correction_negative_ratio,
        )
        suppression = suppression_loss(
            outputs["suppress_logits"],
            mask,
            coarse_probability,
            candidate_ratio=args.suppression_candidate_ratio,
            coarse_threshold=args.suppression_coarse_threshold,
        )
    hard_negative = safe_hard_negative_loss(
        outputs["logits"],
        mask,
        ratio=args.hard_negative_ratio,
        exclusion_kernel=args.hard_negative_exclusion_kernel,
    )
    center_target = gaussian_center_target(mask, args.center_sigma)
    center = weighted_bce_with_logits(
        outputs["center_logits"],
        center_target,
        args.center_pos_weight,
    )
    if "teacher_logits" in outputs:
        distillation = positive_teacher_distillation(
            outputs["logits"],
            outputs["teacher_logits"],
            mask,
            background_weight=args.distill_background_weight,
        )
    else:
        distillation = outputs["logits"].sum() * 0.0

    total = (
        final_loss
        + args.coarse_weight * coarse_loss
        + args.recovery_weight * recovery
        + args.suppression_weight * suppression
        + args.hard_negative_weight * hard_negative
        + args.center_weight * center
        + args.current_distill_weight * distillation
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "mask_loss": float(final_loss.detach().cpu()),
        "bce": base_parts["bce"],
        "iou_loss": base_parts["iou_loss"],
        "fa_loss": base_parts["fa_loss"],
        "coarse_loss": float(coarse_loss.detach().cpu()),
        "recovery_loss": float(recovery.detach().cpu()),
        "suppression_loss": float(suppression.detach().cpu()),
        "hard_negative_loss": float(hard_negative.detach().cpu()),
        "center_loss": float(center.detach().cpu()),
        "distillation_loss": float(distillation.detach().cpu()),
        "distill_weight": float(args.current_distill_weight),
    }
    parts.update(
        {
            key: float(outputs[key].detach().cpu())
            for key in DIAGNOSTIC_KEYS
        }
    )
    return total, parts


def autocast_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16


def parameter_group(name: str) -> str:
    if name.startswith("correction."):
        return "correction"
    if name.startswith(("up3.", "up2.", "up1.", "head.")):
        return "decoder"
    return "backbone"



class PadToSizeDataset(Dataset):
    """Read native NUDT clips and pad them to a fixed canvas without resizing."""

    def __init__(
        self,
        root: str | Path,
        split: str | Path,
        num_frames: int,
        target_size: tuple[int, int],
        image_dir_name: str,
        mask_dir_name: str,
        allow_resize_larger: bool = False,
    ) -> None:
        roots = _multi_values(root)
        splits = _multi_values(split)
        if len(splits) == 1 and len(roots) > 1:
            splits = splits * len(roots)
        if len(roots) != len(splits):
            raise ValueError(
                "--data-root and split must have the same number of ';'-separated entries."
            )
        bases: list[Dataset] = []
        for root_item, split_item in zip(roots, splits):
            if is_nudt_split(str(root_item), str(split_item)):
                bases.append(
                    NUDTMIRSDTDataset(
                        root=root_item,
                        split=split_item,
                        num_frames=num_frames,
                        size=None,
                        image_dir_name=image_dir_name,
                        mask_dir_name=mask_dir_name,
                        native_crop_size=None,
                        training=False,
                    )
                )
            else:
                bases.append(
                    TemporalSegmentationDataset(
                        root=root_item,
                        split=split_item,
                        num_frames=num_frames,
                        size=None,
                        image_dir_name=image_dir_name,
                        mask_dir_name=mask_dir_name,
                    )
                )
        self.base = bases[0] if len(bases) == 1 else ConcatDataset(bases)
        self.target_width, self.target_height = target_size
        self.allow_resize_larger = allow_resize_larger

    def __len__(self) -> int:
        return len(self.base)

    def _pad_image(self, value: torch.Tensor, is_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = value.shape[-2:]
        if height > self.target_height or width > self.target_width:
            if not self.allow_resize_larger:
                raise ValueError(
                    f"Input {width}x{height} exceeds target "
                    f"{self.target_width}x{self.target_height}; "
                    "pass --allow-resize-larger if resizing is intended."
                )
            mode = "nearest" if is_mask else "bilinear"
            flat = value.reshape(-1, 1, height, width)
            kwargs = {"size": (self.target_height, self.target_width), "mode": mode}
            if mode == "bilinear":
                kwargs["align_corners"] = False
            resized = torch.nn.functional.interpolate(flat, **kwargs).reshape(
                *value.shape[:-2],
                self.target_height,
                self.target_width,
            )
            valid = torch.ones(
                (1, self.target_height, self.target_width),
                dtype=value.dtype,
            )
            return resized, valid

        padded = value.new_zeros(
            (*value.shape[:-2], self.target_height, self.target_width)
        )
        padded[..., :height, :width] = value
        valid = value.new_zeros((1, self.target_height, self.target_width))
        valid[..., :height, :width] = 1.0
        return padded, valid

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base[index]
        frames = sample["frames"]
        mask = sample["mask"]
        frames, valid = self._pad_image(frames, is_mask=False)
        mask, _ = self._pad_image(mask, is_mask=True)
        return {
            "frames": frames,
            "mask": mask,
            "valid_mask": valid,
            "crop_mode": "pad512",
            "sequence": sample["sequence"],
            "frame": sample["frame"],
        }


def _multi_values(value: str | Path) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def build_loader(args: argparse.Namespace, split: str, training: bool) -> DataLoader:
    dataset = PadToSizeDataset(
        root=args.data_root,
        split=split,
        num_frames=args.num_frames,
        target_size=parse_size(args.pad_size),
        image_dir_name=args.image_dir_name,
        mask_dir_name=args.mask_dir_name,
        allow_resize_larger=args.allow_resize_larger,
    )
    max_samples = args.max_train_samples if training else args.max_val_samples
    if max_samples is not None and 0 < max_samples < len(dataset):
        generator = torch.Generator().manual_seed(args.seed + (0 if training else 1))
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if training else args.val_batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=training,
    )


def valid_size(valid_mask: torch.Tensor) -> tuple[int, int]:
    valid = valid_mask[0] > 0.5
    coords = torch.nonzero(valid, as_tuple=False)
    if coords.numel() == 0:
        return int(valid.shape[-2]), int(valid.shape[-1])
    height = int(coords[:, 0].max().item()) + 1
    width = int(coords[:, 1].max().item()) + 1
    return height, width


def component_stats(binary: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    binary = binary.astype(bool)
    if ndimage is not None:
        labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        result: list[dict[str, np.ndarray | int]] = []
        for label_index in range(1, count + 1):
            ys, xs = np.nonzero(labels == label_index)
            if ys.size == 0:
                continue
            coords = np.stack([ys, xs], axis=1)
            result.append(
                {
                    "center": coords.astype(np.float64).mean(axis=0),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                }
            )
        return result

    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    result: list[dict[str, np.ndarray | int]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords_list: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords_list.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            coords = np.asarray(coords_list, dtype=np.int64)
            result.append(
                {
                    "center": coords.astype(np.float64).mean(axis=0),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                }
            )
    return result


def dqaligner_component_pd_fa(
    prediction: np.ndarray,
    target: np.ndarray,
    match_radius: float,
    inclusive: bool,
) -> tuple[int, int, int, int, int]:
    pred_components = component_stats(prediction)
    target_components = component_stats(target)
    matched_pred: set[int] = set()
    detected_targets = 0

    for target_component in target_components:
        target_center = target_component["center"]
        for pred_index, pred_component in enumerate(pred_components):
            if pred_index in matched_pred:
                continue
            distance = float(np.linalg.norm(pred_component["center"] - target_center))
            is_match = distance <= match_radius if inclusive else distance < match_radius
            if is_match:
                matched_pred.add(pred_index)
                detected_targets += 1
                break

    false_pixels = sum(
        int(pred_component["area"])
        for pred_index, pred_component in enumerate(pred_components)
        if pred_index not in matched_pred
    )
    false_targets = len(pred_components) - len(matched_pred)
    return (
        detected_targets,
        len(target_components),
        false_targets,
        int(false_pixels),
        int(target.size),
    )


def train_one_epoch(
    model: TFGNetV101,
    loader: DataLoader,
    criterion: IRSTDLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: amp.GradScaler,
    use_amp: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    accumulation_steps = max(args.accumulation_steps, 1)
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        frames = batch["frames"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype(args),
            enabled=use_amp,
        ):
            outputs = model(frames, return_aux=True)
            loss, parts = compute_loss(outputs, mask, criterion, args)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite V10.1-SCMT512 loss at batch {batch_index}: {parts}"
            )
        group_position = batch_index % accumulation_steps
        remaining_batches = len(loader) - (batch_index - group_position)
        group_size = min(accumulation_steps, remaining_batches)
        scaler.scale(loss / group_size).backward()
        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            trainable = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.grad_clip,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        count += 1
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: TFGNetV101,
    loader: DataLoader,
    criterion: IRSTDLoss,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics(threshold=args.eval_threshold)
    coarse_metrics = SegmentationMetrics(threshold=args.eval_threshold)
    totals: dict[str, float] = {}
    count = 0
    thresholds = torch.tensor(
        args.sweep_thresholds,
        device=device,
        dtype=torch.float32,
    ).view(-1, 1, 1, 1, 1)
    sweep_inter = torch.zeros(thresholds.shape[0], device=device)
    sweep_union = torch.zeros(thresholds.shape[0], device=device)
    sweep_tp = torch.zeros(thresholds.shape[0], device=device)
    sweep_fp = torch.zeros(thresholds.shape[0], device=device)
    sweep_fn = torch.zeros(thresholds.shape[0], device=device)
    total_pixels = 0
    dq_detected_targets = 0
    dq_total_targets = 0
    dq_false_targets = 0
    dq_false_pixels = 0
    dq_total_pixels = 0

    for batch in tqdm(loader, desc="val", leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype(args),
            enabled=use_amp,
        ):
            outputs = model(frames, return_aux=True)
            _, parts = compute_loss(outputs, mask, criterion, args)
        if not all(math.isfinite(value) for value in parts.values()):
            raise FloatingPointError(
                f"Non-finite V10.1-SCMT512 validation value: {parts}"
            )
        for sample_index in range(mask.shape[0]):
            crop_height, crop_width = valid_size(valid_mask[sample_index])
            logits_i = outputs["logits"][
                sample_index : sample_index + 1,
                :,
                :crop_height,
                :crop_width,
            ]
            coarse_logits_i = outputs["coarse_logits"][
                sample_index : sample_index + 1,
                :,
                :crop_height,
                :crop_width,
            ]
            mask_i = mask[
                sample_index : sample_index + 1,
                :,
                :crop_height,
                :crop_width,
            ]
            metrics.update(logits_i, mask_i)
            coarse_metrics.update(coarse_logits_i, mask_i)

            probability = torch.sigmoid(logits_i).float().unsqueeze(0)
            ground_truth = (mask_i > 0.5).unsqueeze(0)
            prediction = probability > thresholds
            sweep_inter += (
                prediction & ground_truth
            ).sum(dim=(1, 2, 3, 4)).float()
            sweep_union += (
                prediction | ground_truth
            ).sum(dim=(1, 2, 3, 4)).float()
            sweep_tp += (
                prediction & ground_truth
            ).sum(dim=(1, 2, 3, 4)).float()
            sweep_fp += (
                prediction & ~ground_truth
            ).sum(dim=(1, 2, 3, 4)).float()
            sweep_fn += (
                ~prediction & ground_truth
            ).sum(dim=(1, 2, 3, 4)).float()
            total_pixels += mask_i.numel()

            pred_np = (
                torch.sigmoid(logits_i)[0, 0].detach().cpu().numpy()
                > args.eval_threshold
            )
            target_np = mask_i[0, 0].detach().cpu().numpy() > 0.5
            (
                detected_targets,
                total_targets,
                false_targets,
                false_pixels,
                pixels,
            ) = dqaligner_component_pd_fa(
                pred_np,
                target_np,
                match_radius=args.component_match_radius,
                inclusive=args.component_match_inclusive,
            )
            dq_detected_targets += detected_targets
            dq_total_targets += total_targets
            dq_false_targets += false_targets
            dq_false_pixels += false_pixels
            dq_total_pixels += pixels
        count += 1
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value

    result = {key: value / max(count, 1) for key, value in totals.items()}
    result.update(metrics.compute())
    result.update(
        {
            f"coarse_{key}": value
            for key, value in coarse_metrics.compute().items()
        }
    )
    result["threshold"] = float(args.eval_threshold)
    miou = sweep_inter / (sweep_union + 1e-6)
    best_index = int(torch.argmax(miou).item())
    precision = sweep_tp / (sweep_tp + sweep_fp + 1e-6)
    recall = sweep_tp / (sweep_tp + sweep_fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    fa = sweep_fp / (float(total_pixels) + 1e-6)
    result["best_pixel_threshold"] = float(args.sweep_thresholds[best_index])
    result["best_pixel_mIoU"] = float(miou[best_index].cpu())
    result["best_pixel_precision"] = float(precision[best_index].cpu())
    result["best_pixel_recall"] = float(recall[best_index].cpu())
    result["best_pixel_F1"] = float(f1[best_index].cpu())
    result["best_pixel_Fa"] = float(fa[best_index].cpu())
    result["dq_Pd"] = dq_detected_targets / (dq_total_targets + 1e-6)
    result["dq_Fa"] = dq_false_pixels / (dq_total_pixels + 1e-6)
    result["dq_Fa_x1e6"] = result["dq_Fa"] * 1_000_000.0
    result["dq_detected_targets"] = float(dq_detected_targets)
    result["dq_total_targets"] = float(dq_total_targets)
    result["dq_false_targets"] = float(dq_false_targets)
    result["dq_false_pixels"] = float(dq_false_pixels)
    result["dq_total_pixels"] = float(dq_total_pixels)
    result["validation_region"] = "valid_original_region"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TFGNet V10.1 from scratch with SCMT/DEMNet-style padding."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--output-dir", default="runs/nudt_tfgnet_v101_scmt512")
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--mask-dir-name", default="masks")
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--pad-size", default="512x512")
    parser.add_argument("--allow-resize-larger", action="store_true")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Ablation: bypass the dual-error correction head and use coarse logits as final logits.",
    )
    parser.add_argument(
        "--no-frequency-memory",
        action="store_true",
        help="Ablation: replace the s3 frequency-aware temporal memory with plain temporal aggregation.",
    )
    parser.add_argument(
        "--no-reliability",
        action="store_true",
        help="Ablation: inject temporal evidence without reliability gating.",
    )

    parser.add_argument("--coarse-weight", type=float, default=0.20)
    parser.add_argument("--recovery-weight", type=float, default=0.15)
    parser.add_argument("--suppression-weight", type=float, default=0.15)
    parser.add_argument("--correction-negative-ratio", type=float, default=0.002)
    parser.add_argument("--suppression-candidate-ratio", type=float, default=0.002)
    parser.add_argument("--suppression-coarse-threshold", type=float, default=0.01)
    parser.add_argument("--hard-negative-weight", type=float, default=0.10)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.002)
    parser.add_argument("--hard-negative-exclusion-kernel", type=int, default=5)
    parser.add_argument("--center-weight", type=float, default=0.10)
    parser.add_argument("--center-sigma", type=float, default=1.5)
    parser.add_argument("--center-pos-weight", type=float, default=15.0)
    parser.add_argument("--distill-background-weight", type=float, default=0.02)

    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--component-match-radius", type=float, default=3.0)
    parser.add_argument(
        "--component-match-inclusive",
        action="store_true",
        help="Use <= radius for component matching. DQAligner uses strict < radius.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
    )
    parser.add_argument(
        "--save-best-metric",
        default="mIoU",
        choices=["mIoU", "F1", "dq_Pd", "best_pixel_mIoU", "best_pixel_F1"],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--refine-epochs", type=int, default=0)
    parser.add_argument("--log-every-epochs", type=int, default=1)
    parser.add_argument("--val-every-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Use a fixed random subset for fast protocol probes; omit for full training.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Use a fixed random validation subset during fast probes; omit for full validation.",
    )
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--step-gamma", type=float, default=0.2)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--fa-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    args.sweep_thresholds = parse_thresholds(args.sweep_thresholds)
    args.current_distill_weight = 0.0
    if args.no_correction and args.warmup_epochs > 0:
        print(
            "no_correction=True: skipping correction-only warmup "
            f"(warmup_epochs {args.warmup_epochs} -> 0)."
        )
        args.warmup_epochs = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device.type
    save_json(output_dir / "args.json", vars(args))

    train_loader = build_loader(args, args.train_split, training=True)
    val_loader = build_loader(args, args.val_split, training=False)
    model = TFGNetV101(
        base_channels=args.base_channels,
        use_temporal=not args.no_temporal,
        use_frequency_memory=not args.no_frequency_memory,
        use_reliability=not args.no_reliability,
        use_correction=not args.no_correction,
    ).to(device)
    if args.pretrained:
        checkpoint = load_checkpoint_state(args.pretrained, device)
        loaded, missing, unexpected, skipped = load_pretrained(
            model,
            checkpoint["model"],
            strict=False,
        )
        print(
            f"loaded pretrained={args.pretrained} loaded={loaded} "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"skipped={len(skipped)}"
        )

    groups: dict[str, list[torch.nn.Parameter]] = {
        "correction": [],
        "decoder": [],
        "backbone": [],
    }
    for name, parameter in model.named_parameters():
        groups[parameter_group(name)].append(parameter)
    optimizer_groups = [
        {"params": groups["correction"], "lr": args.lr, "name": "correction"},
        {"params": groups["decoder"], "lr": args.decoder_lr, "name": "decoder"},
        {"params": groups["backbone"], "lr": args.backbone_lr, "name": "backbone"},
    ]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            optimizer_groups,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=args.weight_decay,
        )
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.step_gamma,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    criterion = IRSTDLoss(fa_weight=args.fa_weight)
    use_amp = args.amp and device.type == "cuda"
    if (
        use_amp
        and args.amp_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("This GPU does not support BF16 AMP.")
    scaler = make_grad_scaler(
        enabled=use_amp and args.amp_dtype == "float16",
        init_scale=512.0,
        growth_interval=10000,
    )
    start_epoch = 1
    best_score = -float("inf")
    if args.resume:
        resume = load_checkpoint_state(args.resume, device)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = float(resume.get("best_score", best_score))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} pad_size={args.pad_size} "
        f"params={parameter_count / 1e6:.3f}M"
    )
    log_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            training_stage = "warmup"
        elif epoch <= args.refine_epochs:
            training_stage = "refine"
        else:
            training_stage = "finetune"
        model.set_training_stage(training_stage)

        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            args,
        )
        scheduler.step()
        if args.min_lr > 0.0:
            for group in optimizer.param_groups:
                group["lr"] = max(float(group["lr"]), args.min_lr)
        row = {
            "epoch": epoch,
            "training_stage": training_stage,
            "correction_lr": float(optimizer.param_groups[0]["lr"]),
            "decoder_lr": float(optimizer.param_groups[1]["lr"]),
            "backbone_lr": float(optimizer.param_groups[2]["lr"]),
            **{f"train_{key}": value for key, value in train_stats.items()},
        }
        should_validate = (
            epoch == args.epochs
            or epoch % max(args.val_every_epochs, 1) == 0
        )
        if should_validate:
            val_stats = validate(
                model,
                val_loader,
                criterion,
                device,
                use_amp,
                args,
            )
            row.update({f"val_{key}": value for key, value in val_stats.items()})
            score = val_stats[args.save_best_metric]
            if not math.isfinite(score):
                raise FloatingPointError(f"Non-finite validation score: {score}")
            if score > best_score:
                best_score = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "score": score,
                        "metrics": val_stats,
                    },
                    output_dir / "best.pt",
                )

        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("V10.1-SCMT512 contains non-finite parameters.")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "epoch": epoch,
                "best_score": best_score,
                "log": row,
            },
            output_dir / "last.pt",
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        if (
            epoch == 1
            or epoch == args.epochs
            or epoch % max(args.log_every_epochs, 1) == 0
        ):
            print(
                " ".join(
                    f"{key}={value:.6f}"
                    if isinstance(value, float)
                    else f"{key}={value}"
                    for key, value in row.items()
                )
            )


if __name__ == "__main__":
    main()
