from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
try:
    from torch import amp
except ImportError:  # PyTorch 1.x exposes autocast under torch.cuda.amp.
    from torch.cuda import amp  # type: ignore[no-redef]
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import NUDTMIRSDTDataset, TemporalSegmentationDataset  # noqa: E402
from model import TFGNetV101  # noqa: E402




def parse_size(value: str) -> tuple[int, int]:
    width, height = value.lower().replace(" ", "").split("x", maxsplit=1)
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


try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    ndimage = None


DEMNET_SHORT_THRESHOLDS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.8,
    0.9,
    0.95,
    1.0,
)

DEMNET_FULL_THRESHOLDS = (
    0.0,
    1e-30,
    1e-20,
    1e-19,
    1e-18,
    1e-17,
    1e-16,
    1e-15,
    1e-14,
    1e-13,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.98,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
    0.99999,
    0.999999,
    0.9999999,
    1.0,
)


def autocast_context(
    device_type: str,
    dtype: torch.dtype,
    enabled: bool,
):
    try:
        return amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    except TypeError:
        return amp.autocast(dtype=dtype, enabled=enabled)


@dataclass
class Component:
    center: np.ndarray
    area: int


class DemnetCounters:
    def __init__(self, thresholds: tuple[float, ...]) -> None:
        count = len(thresholds)
        self.true_targets = np.zeros(count, dtype=np.float64)
        self.total_targets = np.zeros(count, dtype=np.float64)
        self.false_targets = np.zeros(count, dtype=np.float64)
        self.false_pixels = np.zeros(count, dtype=np.float64)
        self.total_pixels = 0.0
        self.intersection = np.zeros(count, dtype=np.float64)
        self.union = np.zeros(count, dtype=np.float64)
        self.predicted_pixels = np.zeros(count, dtype=np.float64)
        self.target_pixels = np.zeros(count, dtype=np.float64)
        self.correct_pixels = np.zeros(count, dtype=np.float64)
        self.images = 0

    def add_image_metrics(
        self,
        threshold_index: int,
        prediction: np.ndarray,
        target: np.ndarray,
        true_targets: int,
        total_targets: int,
        false_targets: int,
        false_pixels: int,
    ) -> None:
        pred = prediction.astype(bool)
        gt = target.astype(bool)
        self.true_targets[threshold_index] += true_targets
        self.total_targets[threshold_index] += total_targets
        self.false_targets[threshold_index] += false_targets
        self.false_pixels[threshold_index] += false_pixels
        self.intersection[threshold_index] += np.logical_and(pred, gt).sum()
        self.union[threshold_index] += np.logical_or(pred, gt).sum()
        self.predicted_pixels[threshold_index] += pred.sum()
        self.target_pixels[threshold_index] += gt.sum()
        self.correct_pixels[threshold_index] += (pred == gt).sum()

    def add_image(self, pixels: int) -> None:
        self.total_pixels += pixels
        self.images += 1

    def rows(self, group: str, thresholds: tuple[float, ...]) -> list[dict[str, float | str]]:
        eps = np.finfo(np.float64).eps
        rows: list[dict[str, float | str]] = []
        for index, threshold in enumerate(thresholds):
            precision = self.intersection[index] / (self.predicted_pixels[index] + eps)
            recall = self.intersection[index] / (self.target_pixels[index] + eps)
            pd = self.true_targets[index] / (self.total_targets[index] + eps)
            fa = self.false_pixels[index] / (self.total_pixels + eps)
            fat = self.false_targets[index] / (self.total_targets[index] + eps)
            rows.append(
                {
                    "group": group,
                    "threshold": float(threshold),
                    "Pd": float(pd),
                    "Pd_percent": float(pd * 100.0),
                    "Fa": float(fa),
                    "Fa_x1e6": float(fa * 1_000_000.0),
                    "FaT": float(fat),
                    "FaT_percent": float(fat * 100.0),
                    "strict_Fa": float((self.predicted_pixels[index] - self.intersection[index]) / (self.total_pixels + eps)),
                    "strict_Fa_x1e6": float(((self.predicted_pixels[index] - self.intersection[index]) / (self.total_pixels + eps)) * 1_000_000.0),
                    "strict_false_pixels": float(self.predicted_pixels[index] - self.intersection[index]),
                    "true_targets": float(self.true_targets[index]),
                    "total_targets": float(self.total_targets[index]),
                    "false_targets": float(self.false_targets[index]),
                    "false_pixels": float(self.false_pixels[index]),
                    "pixels": float(self.total_pixels),
                    "IoU": float(self.intersection[index] / (self.union[index] + eps)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "F1": float(2.0 * precision * recall / (precision + recall + eps)),
                    "pixAcc": float(self.correct_pixels[index] / (self.total_pixels + eps)),
                    "images": float(self.images),
                }
            )
        return rows


def parse_thresholds(value: str) -> tuple[float, ...]:
    key = value.strip().lower()
    if key in {"demnet", "demnet_short", "short"}:
        return DEMNET_SHORT_THRESHOLDS
    if key in {"demnet_full", "full"}:
        return DEMNET_FULL_THRESHOLDS
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_int_list(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def sequence_number(sequence_name: str) -> int | None:
    if not sequence_name.startswith("Sequence"):
        return None
    try:
        return int(sequence_name.replace("Sequence", ""))
    except ValueError:
        return None


def read_split_sequences(split_path: Path) -> set[str]:
    sequences: set[str] = set()
    for raw in split_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sequences.add(line.replace("\\", "/").split("/")[0])
    return sequences


def _multi_values(value: str | Path) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def read_split_sequences_arg(value: str | Path) -> set[str]:
    sequences: set[str] = set()
    for split_item in _multi_values(value):
        sequences.update(read_split_sequences(Path(split_item)))
    return sequences


def components(binary: np.ndarray) -> list[Component]:
    binary = binary.astype(bool)
    if ndimage is not None:
        labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        result: list[Component] = []
        for label_index in range(1, count + 1):
            region = labels == label_index
            center = np.asarray(ndimage.center_of_mass(region), dtype=np.float64)
            result.append(Component(center=center, area=int(region.sum())))
        return result

    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    result = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            arr = np.asarray(coords, dtype=np.float64)
            result.append(Component(center=arr.mean(axis=0), area=arr.shape[0]))
    return result


def demnet_shooting_rules(
    prediction: np.ndarray,
    target: np.ndarray,
    match_radius: float,
    inclusive: bool,
    match_mode: str,
) -> tuple[int, int, int, int]:
    pred_components = components(prediction)
    target_components = components(target)
    if match_mode == "demnet":
        matched_targets: set[int] = set()
        true_count = 0
        false_targets = 0
        false_pixels = 0

        for pred_component in pred_components:
            min_distance = float("inf")
            closest_target = None
            for target_index, target_component in enumerate(target_components):
                if target_index in matched_targets:
                    continue
                distance = float(np.linalg.norm(pred_component.center - target_component.center))
                if distance < min_distance:
                    min_distance = distance
                    closest_target = target_index

            is_match = (
                closest_target is not None
                and (min_distance <= match_radius if inclusive else min_distance < match_radius)
            )
            if is_match:
                true_count += 1
                matched_targets.add(int(closest_target))
            else:
                false_targets += 1
                false_pixels += pred_component.area

        return true_count, len(target_components), false_targets, false_pixels

    if match_mode != "dqaligner":
        raise ValueError(f"Unknown match_mode: {match_mode}")

    matched_predictions: set[int] = set()
    true_count = 0

    for target_component in target_components:
        for pred_index, pred_component in enumerate(pred_components):
            if pred_index in matched_predictions:
                continue
            distance = float(np.linalg.norm(pred_component.center - target_component.center))
            is_match = distance <= match_radius if inclusive else distance < match_radius
            if is_match:
                true_count += 1
                matched_predictions.add(pred_index)
                break

    false_targets = len(pred_components) - len(matched_predictions)
    false_pixels = sum(
        pred_component.area
        for pred_index, pred_component in enumerate(pred_components)
        if pred_index not in matched_predictions
    )

    return true_count, len(target_components), false_targets, false_pixels


def pad_top_left(
    frames: torch.Tensor,
    target_size: tuple[int, int],
    allow_resize_larger: bool,
) -> tuple[torch.Tensor, int, int]:
    batch, num_frames, channels, height, width = frames.shape
    target_width, target_height = target_size
    if height > target_height or width > target_width:
        if not allow_resize_larger:
            raise ValueError(
                f"Input frame {width}x{height} is larger than target "
                f"{target_width}x{target_height}; pass --allow-resize-larger to resize."
            )
        flat = frames.view(batch * num_frames, channels, height, width)
        resized = torch.nn.functional.interpolate(
            flat,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.view(batch, num_frames, channels, target_height, target_width), target_height, target_width

    padded = frames.new_zeros((batch, num_frames, channels, target_height, target_width))
    padded[..., :height, :width] = frames
    return padded, height, width


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_frame_groups(
    path: str | Path,
    group_column: str,
    low_value: str,
) -> dict[tuple[str, str, str], str]:
    groups: dict[tuple[str, str, str], str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            root = row.get("root", "").strip()
            sequence = row.get("sequence", "").strip()
            frame = row.get("frame", "").strip()
            if not sequence or not frame:
                continue
            group_value = row.get(group_column, "").strip()
            group = "low_snr" if group_value == low_value else "high_snr"
            groups[(root, sequence, frame)] = group
            groups[("", sequence, frame)] = group
    return groups


def first_batch_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else default
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TFGNet V10.1 with DQAligner/DEMNet-style metrics, strict pixel Fa, and Pd-Fa AUC."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--mask-dir-name", default="masks")
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Evaluate a checkpoint trained without the temporal branch.",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Evaluate with the dual-error correction head bypassed.",
    )
    parser.add_argument(
        "--no-frequency-memory",
        action="store_true",
        help="Evaluate a checkpoint trained without s3 frequency-aware temporal memory.",
    )
    parser.add_argument(
        "--no-reliability",
        action="store_true",
        help="Evaluate with reliability gating disabled.",
    )
    parser.add_argument(
        "--input-size",
        default="512x512",
        help="Evaluation input canvas, e.g. 512x512, or native/original for unpadded full images.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="demnet")
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument(
        "--match-mode",
        choices=["dqaligner", "demnet"],
        default="dqaligner",
    )
    parser.add_argument(
        "--match-inclusive",
        action="store_true",
        help="Use <= radius. DQAligner source code uses strict < radius; DEMNet source uses <= radius.",
    )
    parser.add_argument(
        "--low-snr-sequences",
        default="47,56,59,76,92,101,105,119",
    )
    parser.add_argument(
        "--high-snr-sequences",
        default="85,86,87,88,89,90,91,93,94,95,96,97",
    )
    parser.add_argument(
        "--frame-group-csv",
        default="",
        help=(
            "Optional CSV with root, sequence, frame and a group column. "
            "When set, frame-level groups override low/high sequence lists."
        ),
    )
    parser.add_argument("--frame-group-column", default="group_by_median")
    parser.add_argument("--frame-group-low-value", default="SCR<=3")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--allow-resize-larger", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    thresholds = parse_thresholds(args.thresholds)
    if args.threshold not in thresholds:
        thresholds = tuple(sorted(set(thresholds + (args.threshold,))))
    threshold_index = thresholds.index(args.threshold)
    input_size = (
        None
        if args.input_size.strip().lower() in {"native", "original", "none"}
        else parse_size(args.input_size)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    roots = _multi_values(args.data_root)
    splits = _multi_values(args.val_split)
    if len(splits) == 1 and len(roots) > 1:
        splits = splits * len(roots)
    if len(roots) != len(splits):
        raise ValueError(
            "--data-root and --val-split must have the same number of "
            "';'-separated entries."
        )
    datasets = []
    for root_item, split_item in zip(roots, splits):
        if is_nudt_split(root_item, split_item):
            datasets.append(
                NUDTMIRSDTDataset(
                    root=root_item,
                    split=split_item,
                    num_frames=args.num_frames,
                    size=None,
                    image_dir_name=args.image_dir_name,
                    mask_dir_name=args.mask_dir_name,
                    native_crop_size=None,
                    training=False,
                )
            )
        else:
            datasets.append(
                TemporalSegmentationDataset(
                    root=root_item,
                    split=split_item,
                    num_frames=args.num_frames,
                    size=None,
                    image_dir_name=args.image_dir_name,
                    mask_dir_name=args.mask_dir_name,
                )
            )
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TFGNetV101(
        base_channels=args.base_channels,
        use_temporal=not args.no_temporal,
        use_frequency_memory=not args.no_frequency_memory,
        use_reliability=not args.no_reliability,
        use_correction=not args.no_correction,
    ).to(device)
    checkpoint = load_checkpoint_state(args.checkpoint, device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    loaded, missing, unexpected, skipped = load_pretrained(model, state_dict, strict=False)
    if missing or unexpected or skipped:
        raise RuntimeError(
            "Checkpoint did not exactly match TFGNetV101: "
            f"loaded={loaded} missing={len(missing)} "
            f"unexpected={len(unexpected)} skipped={len(skipped)}"
        )
    model.eval()

    low_snr = parse_int_list(args.low_snr_sequences)
    high_snr = parse_int_list(args.high_snr_sequences)
    frame_groups = (
        load_frame_groups(args.frame_group_csv, args.frame_group_column, args.frame_group_low_value)
        if args.frame_group_csv
        else None
    )
    split_sequences = read_split_sequences_arg(args.val_split)
    unknown_sequences = sorted(
        sequence
        for sequence in split_sequences
        if (sequence_number(sequence) not in low_snr | high_snr)
    )

    counters = {
        "all": DemnetCounters(thresholds),
        "low_snr": DemnetCounters(thresholds),
        "high_snr": DemnetCounters(thresholds),
    }
    if unknown_sequences or frame_groups is not None:
        counters["unknown"] = DemnetCounters(thresholds)

    per_image_rows: list[dict[str, object]] = []
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    use_amp = args.amp and device.type == "cuda"
    processed = 0

    for batch in tqdm(loader, desc="demnet-protocol-eval"):
        frames = batch["frames"].to(device, non_blocking=True)
        mask = batch["mask"]
        root_name = first_batch_value(batch.get("root"), "")
        sequence = str(batch["sequence"][0])
        frame = str(batch["frame"][0])
        seq_num = sequence_number(sequence)
        if frame_groups is not None:
            current_group = frame_groups.get(
                (root_name, sequence, frame),
                frame_groups.get(("", sequence, frame), "unknown"),
            )
            group_names = ("all", current_group)
        elif seq_num in low_snr:
            current_group = "low_snr"
            group_names = ("all", "low_snr")
        elif seq_num in high_snr:
            current_group = "high_snr"
            group_names = ("all", "high_snr")
        else:
            current_group = "unknown"
            group_names = ("all", "unknown")

        original_height = int(mask.shape[-2])
        original_width = int(mask.shape[-1])
        if input_size is None:
            padded_frames = frames
            valid_height = original_height
            valid_width = original_width
        else:
            padded_frames, valid_height, valid_width = pad_top_left(
                frames,
                input_size,
                allow_resize_larger=args.allow_resize_larger,
            )
        with autocast_context(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(padded_frames, return_aux=True)
            logits = output["logits"] if isinstance(output, dict) else output
        probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        if valid_height != original_height or valid_width != original_width:
            target_np = (
                torch.nn.functional.interpolate(
                    mask.float(),
                    size=(valid_height, valid_width),
                    mode="nearest",
                )[0, 0]
                .cpu()
                .numpy()
                > 0.5
            )
            probability_valid = probability[:valid_height, :valid_width]
        else:
            target_np = mask[0, 0].cpu().numpy() > 0.5
            probability_valid = probability[:original_height, :original_width]

        pixels = int(target_np.size)
        for group_name in group_names:
            counters[group_name].add_image(pixels)

        fixed_row: dict[str, object] | None = None
        for index, threshold in enumerate(thresholds):
            prediction = probability_valid >= threshold
            true_num, tgt_num, fat_num, false_num = demnet_shooting_rules(
                prediction,
                target_np,
                match_radius=args.match_radius,
                inclusive=args.match_inclusive,
                match_mode=args.match_mode,
            )
            for group_name in group_names:
                counters[group_name].add_image_metrics(
                    index,
                    prediction,
                    target_np,
                    true_num,
                    tgt_num,
                    fat_num,
                    false_num,
                )
            if index == threshold_index:
                inter = int(np.logical_and(prediction, target_np).sum())
                union = int(np.logical_or(prediction, target_np).sum())
                pred_pixels = int(prediction.sum())
                target_pixels = int(target_np.sum())
                fixed_row = {
                    "sequence": sequence,
                    "frame": frame,
                    "root": root_name,
                    "group": current_group,
                    "threshold": args.threshold,
                    "width": original_width,
                    "height": original_height,
                    "eval_width": valid_width,
                    "eval_height": valid_height,
                    "target_pixels": target_pixels,
                    "predicted_pixels": pred_pixels,
                    "intersection": inter,
                    "union": union,
                    "IoU": float(inter / (union + np.finfo(np.float64).eps)),
                    "true_targets": true_num,
                    "total_targets": tgt_num,
                    "false_targets": fat_num,
                    "false_pixels": false_num,
                }
        if fixed_row is not None:
            per_image_rows.append(fixed_row)
        processed += 1
        if args.max_samples is not None and processed >= args.max_samples:
            break

    threshold_rows: list[dict[str, float | str]] = []
    summary: dict[str, object] = {
        "protocol": (
            "DEMNet-style: top-left 512 padding, crop output to original valid "
            "region, DQAligner target-first connected-component centroid matching "
            "or DEMNet pred-first nearest-component matching depending on --match-mode, "
            "center-frame V10.1 output"
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": args.data_root,
        "val_split": args.val_split,
        "input_size": args.input_size,
        "num_frames": args.num_frames,
        "threshold": args.threshold,
        "match_inclusive": args.match_inclusive,
        "match_mode": args.match_mode,
        "thresholds": list(thresholds),
        "low_snr_sequences": sorted(low_snr),
        "high_snr_sequences": sorted(high_snr),
        "unknown_sequences": unknown_sequences,
        "frame_group_csv": args.frame_group_csv,
        "frame_group_column": args.frame_group_column,
        "frame_group_low_value": args.frame_group_low_value,
        "groups": {},
    }
    for group_name, counter in counters.items():
        rows = counter.rows(group_name, thresholds)
        threshold_rows.extend(rows)
        fixed = rows[threshold_index]
        best_pd_index = int(np.argmax([row["Pd"] for row in rows]))
        best_iou_index = int(np.argmax([row["IoU"] for row in rows]))
        # Approximate Pd-Fa AUC over the evaluated thresholds.
        fa_values = np.asarray([float(row["Fa"]) for row in rows], dtype=np.float64)
        pd_values = np.asarray([float(row["Pd"]) for row in rows], dtype=np.float64)
        order = np.argsort(fa_values)
        auc = float(np.trapezoid(pd_values[order], fa_values[order]))
        summary["groups"][group_name] = {
            "fixed_threshold": fixed,
            "best_pd_threshold": rows[best_pd_index],
            "best_iou_threshold": rows[best_iou_index],
            "pd_fa_auc_over_thresholds": auc,
        }

    write_csv(output_dir / "threshold_metrics.csv", threshold_rows)
    write_csv(output_dir / f"per_image_threshold_{str(args.threshold).replace('.', '_')}.csv", per_image_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fixed_all = summary["groups"]["all"]["fixed_threshold"]  # type: ignore[index]
    print(
        "fixed-threshold all "
        f"threshold={args.threshold} "
        f"Pd={fixed_all['Pd_percent']:.4f}% "
        f"Fa={fixed_all['Fa_x1e6']:.6f}e-6 "
        f"FaT={fixed_all['FaT_percent']:.4f}% "
        f"IoU={fixed_all['IoU']:.6f}"
    )
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
