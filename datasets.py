from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ===== Source: datasets.py =====

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    image_dir: Path
    mask_dir: Path
    frames: tuple[Path, ...]


@dataclass(frozen=True)
class SampleRef:
    sequence_index: int
    frame_index: int


def _list_images(path: Path) -> list[Path]:
    def natural_key(value: Path) -> tuple[object, ...]:
        return tuple(
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value.stem)
        )

    return sorted(
        (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _discover_sequences(root: Path, image_dir_name: str, mask_dir_name: str) -> list[SequenceRecord]:
    layouts: list[SequenceRecord] = []

    image_root = root / image_dir_name
    mask_root = root / mask_dir_name
    if image_root.is_dir() and mask_root.is_dir():
        subdirs = [p for p in image_root.iterdir() if p.is_dir()]
        if subdirs:
            for image_dir in sorted(subdirs):
                mask_dir = mask_root / image_dir.name
                if mask_dir.is_dir():
                    frames = tuple(_list_images(image_dir))
                    if frames:
                        layouts.append(SequenceRecord(image_dir.name, image_dir, mask_dir, frames))
        else:
            frames = tuple(_list_images(image_root))
            if frames:
                layouts.append(SequenceRecord(".", image_root, mask_root, frames))

    if layouts:
        return layouts

    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        image_dir = seq_dir / image_dir_name
        mask_dir = seq_dir / mask_dir_name
        if image_dir.is_dir() and mask_dir.is_dir():
            frames = tuple(_list_images(image_dir))
            if frames:
                layouts.append(SequenceRecord(seq_dir.name, image_dir, mask_dir, frames))

    return layouts


def _load_split(split_path: Path | None) -> dict[str, set[str] | None] | None:
    if split_path is None:
        return None

    selected: dict[str, set[str] | None] = {}
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) == 1:
            selected[parts[0]] = None
        else:
            selected.setdefault(parts[0], set())
            stems = selected[parts[0]]
            if stems is not None:
                stems.add(parts[1])
    return selected


def _read_gray(path: Path, size: tuple[int, int] | None, is_mask: bool) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None:
            resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
            image = image.resize(size, resample=resample)
        array = np.asarray(image, dtype=np.float32)

    if is_mask:
        array = (array > 0).astype(np.float32)
    else:
        array = array / 255.0

    return torch.from_numpy(array).unsqueeze(0)


def _resize_like(value: torch.Tensor, reference: torch.Tensor, is_mask: bool) -> torch.Tensor:
    if value.shape[-2:] == reference.shape[-2:]:
        return value
    mode = "nearest" if is_mask else "bilinear"
    kwargs = {
        "size": reference.shape[-2:],
        "mode": mode,
    }
    if mode == "bilinear":
        kwargs["align_corners"] = False
    return torch.nn.functional.interpolate(value.unsqueeze(0), **kwargs).squeeze(0)


class TemporalSegmentationDataset(Dataset):
    """Temporal infrared small-target segmentation dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str | Path | None = None,
        num_frames: int = 5,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        size: tuple[int, int] | None = None,
    ) -> None:
        if num_frames % 2 != 1:
            raise ValueError("num_frames must be odd so that a center frame exists.")

        self.root = Path(root).resolve()
        self.num_frames = num_frames
        self.radius = num_frames // 2
        self.size = size

        split_path = Path(split).resolve() if split is not None else None
        selected = _load_split(split_path)
        sequences = _discover_sequences(self.root, image_dir_name, mask_dir_name)
        if not sequences:
            raise FileNotFoundError(
                f"No sequences found under {self.root}. Expected either "
                f"{image_dir_name}/<seq> and {mask_dir_name}/<seq>, or "
                f"<seq>/{image_dir_name} and <seq>/{mask_dir_name}."
            )

        self.sequences: list[SequenceRecord] = []
        self.samples: list[SampleRef] = []

        for sequence in sequences:
            # Two split styles are supported:
            # 1) sequence-level: "seq_001" or "seq_001 frame_0001"
            # 2) flat image list for single-folder datasets like IRSTD-1K: "XDU123"
            if selected is None:
                frame_stems = None
                allow_sequence = True
                flat_image_split = False
            else:
                flat_keys = all(value is None for value in selected.values())
                allow_sequence = sequence.name in selected
                flat_image_split = len(sequences) == 1 and not allow_sequence and flat_keys
                frame_stems = selected.get(sequence.name) if allow_sequence else (set(selected.keys()) if flat_image_split else None)
                if not allow_sequence and not flat_image_split:
                    continue

            kept_indices: list[int] = []
            for idx, frame_path in enumerate(sequence.frames):
                if frame_stems is not None and frame_path.stem not in frame_stems:
                    continue
                if _find_mask(sequence.mask_dir, frame_path.stem) is not None:
                    kept_indices.append(idx)

            if kept_indices:
                seq_index = len(self.sequences)
                self.sequences.append(sequence)
                self.samples.extend(SampleRef(seq_index, idx) for idx in kept_indices)

        if not self.samples:
            raise FileNotFoundError("No image/mask pairs were found after applying the split.")

    def __len__(self) -> int:
        return len(self.samples)

    def _window_indices(self, frame_index: int, num_total: int) -> Iterable[int]:
        for offset in range(-self.radius, self.radius + 1):
            yield min(max(frame_index + offset, 0), num_total - 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        ref = self.samples[index]
        sequence = self.sequences[ref.sequence_index]
        center_path = sequence.frames[ref.frame_index]
        mask_path = _find_mask(sequence.mask_dir, center_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {center_path}")

        frames = [
            _read_gray(sequence.frames[i], self.size, is_mask=False)
            for i in self._window_indices(ref.frame_index, len(sequence.frames))
        ]
        mask = _read_gray(mask_path, self.size, is_mask=True)
        if self.size is None:
            frames = [_resize_like(frame, mask, is_mask=False) for frame in frames]

        return {
            "frames": torch.stack(frames, dim=0),
            "mask": mask,
            "root": self.root.name,
            "sequence": sequence.name,
            "frame": center_path.stem,
        }


# ===== Source: nudt_dataset.py =====

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class NUDTSampleRef:
    sequence_name: str
    frame_stem: str


def _read_split(split_path: Path) -> dict[str, set[str] | None]:
    selected: dict[str, set[str] | None] = {}
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        normalized = line.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if len(parts) == 1 and parts[0].startswith("Sequence"):
            selected[parts[0]] = None
            continue
        if len(parts) < 3:
            raise ValueError(
                f"Unexpected split entry '{line}'. Expected entries like 'Sequence1/Mix/00001.mat'."
            )

        sequence_name = parts[0]
        frame_stem = Path(parts[-1]).stem
        if sequence_name not in selected:
            selected[sequence_name] = set()
        frame_selection = selected[sequence_name]
        if frame_selection is not None:
            frame_selection.add(frame_stem)

    if not selected:
        raise ValueError(f"No valid entries found in split file: {split_path}")

    return selected


def _read_gray(path: Path, size: tuple[int, int] | None, is_mask: bool) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None:
            resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
            image = image.resize(size, resample=resample)
        array = np.asarray(image, dtype=np.float32)

    if is_mask:
        array = (array > 0).astype(np.float32)
    else:
        array = array / 255.0

    return torch.from_numpy(array).unsqueeze(0)


def _read_gray_letterbox(
    path: Path,
    size: tuple[int, int],
    is_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_width, target_height = size
    with Image.open(path) as image:
        image = image.convert("L")
        source_width, source_height = image.size
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(int(round(source_width * scale)), 1)
        resized_height = max(int(round(source_height * scale)), 1)
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        image = image.resize((resized_width, resized_height), resample=resample)
        array = np.asarray(image, dtype=np.float32)

    canvas = np.zeros((target_height, target_width), dtype=np.float32)
    valid = np.zeros((target_height, target_width), dtype=np.float32)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = array
    valid[top : top + resized_height, left : left + resized_width] = 1.0

    if is_mask:
        canvas = (canvas > 0).astype(np.float32)
    else:
        canvas = canvas / 255.0
    return torch.from_numpy(canvas).unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)


def _native_target_crop(
    frames: list[torch.Tensor],
    mask: torch.Tensor,
    crop_size: tuple[int, int],
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Crop without rescaling while retaining the complete target when possible."""

    crop_width, crop_height = crop_size
    source_height, source_width = mask.shape[-2:]
    pad_width = max(crop_width - source_width, 0)
    pad_height = max(crop_height - source_height, 0)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    if pad_width or pad_height:
        frames = [
            torch.nn.functional.pad(
                frame,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            )
            for frame in frames
        ]
        mask = torch.nn.functional.pad(
            mask,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )

    padded_height, padded_width = mask.shape[-2:]
    max_x = padded_width - crop_width
    max_y = padded_height - crop_height
    target_coords = torch.nonzero(mask[0] > 0.5, as_tuple=False)
    if target_coords.numel() > 0:
        y_min = int(target_coords[:, 0].min())
        y_max = int(target_coords[:, 0].max())
        x_min = int(target_coords[:, 1].min())
        x_max = int(target_coords[:, 1].max())
        x_low = max(0, x_max - crop_width + 1)
        x_high = min(x_min, max_x)
        y_low = max(0, y_max - crop_height + 1)
        y_high = min(y_min, max_y)
        x0 = random.randint(x_low, max(x_low, x_high))
        y0 = random.randint(y_low, max(y_low, y_high))
    else:
        x0 = random.randint(0, max_x) if max_x > 0 else 0
        y0 = random.randint(0, max_y) if max_y > 0 else 0

    x1 = x0 + crop_width
    y1 = y0 + crop_height
    frames = [frame[..., y0:y1, x0:x1] for frame in frames]
    mask = mask[..., y0:y1, x0:x1]

    valid = torch.ones(
        (1, source_height, source_width),
        dtype=mask.dtype,
    )
    if pad_width or pad_height:
        valid = torch.nn.functional.pad(
            valid,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
    return frames, mask, valid[..., y0:y1, x0:x1]


def _balanced_temporal_crop(
    frames: list[torch.Tensor],
    temporal_masks: list[torch.Tensor],
    center_mask: torch.Tensor,
    crop_size: tuple[int, int],
    target_probability: float,
    background_probability: float,
    hard_background_probability: float,
    max_attempts: int,
    forced_mode: str | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, str]:
    """Apply one native-pixel crop to the entire clip and its masks."""

    crop_width, crop_height = crop_size
    source_height, source_width = center_mask.shape[-2:]
    pad_width = max(crop_width - source_width, 0)
    pad_height = max(crop_height - source_height, 0)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    if pad_width or pad_height:
        frames = [
            torch.nn.functional.pad(
                frame,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            )
            for frame in frames
        ]
        temporal_masks = [
            torch.nn.functional.pad(
                temporal_mask,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            )
            for temporal_mask in temporal_masks
        ]
        center_mask = torch.nn.functional.pad(
            center_mask,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )

    padded_height, padded_width = center_mask.shape[-2:]
    max_x = padded_width - crop_width
    max_y = padded_height - crop_height
    temporal_union = torch.stack(temporal_masks, dim=0).amax(dim=0)

    def random_origin() -> tuple[int, int]:
        x0 = random.randint(0, max_x) if max_x > 0 else 0
        y0 = random.randint(0, max_y) if max_y > 0 else 0
        return x0, y0

    def crop_sum(value: torch.Tensor, x0: int, y0: int) -> float:
        return float(value[..., y0 : y0 + crop_height, x0 : x0 + crop_width].sum())

    if forced_mode is not None:
        crop_mode = forced_mode
    else:
        draw = random.random()
        target_cutoff = target_probability
        background_cutoff = target_cutoff + background_probability
        if draw < target_cutoff:
            crop_mode = "target"
        elif draw < background_cutoff:
            crop_mode = "background"
        else:
            crop_mode = "hard_background"

    target_coords = torch.nonzero(center_mask[0] > 0.5, as_tuple=False)
    if crop_mode == "target" and target_coords.numel() > 0:
        y_min = int(target_coords[:, 0].min())
        y_max = int(target_coords[:, 0].max())
        x_min = int(target_coords[:, 1].min())
        x_max = int(target_coords[:, 1].max())
        x_low = max(0, x_max - crop_width + 1)
        x_high = min(x_min, max_x)
        y_low = max(0, y_max - crop_height + 1)
        y_high = min(y_min, max_y)
        x0 = random.randint(x_low, max(x_low, x_high))
        y0 = random.randint(y_low, max(y_low, y_high))
    else:
        def all_crop_sums(value: torch.Tensor) -> torch.Tensor:
            plane = value[0].float()
            integral = F.pad(plane, (1, 0, 1, 0)).cumsum(0).cumsum(1)
            return (
                integral[crop_height:, crop_width:]
                - integral[:-crop_height, crop_width:]
                - integral[crop_height:, :-crop_width]
                + integral[:-crop_height, :-crop_width]
            )

        center_sums = all_crop_sums(center_mask)
        temporal_sums = all_crop_sums(temporal_union)
        if crop_mode == "hard_background":
            candidates = torch.nonzero(
                (center_sums == 0) & (temporal_sums > 0),
                as_tuple=False,
            )
        else:
            candidates = torch.nonzero(temporal_sums == 0, as_tuple=False)

        if candidates.numel() > 0:
            selected = candidates[random.randrange(candidates.shape[0])]
            y0 = int(selected[0])
            x0 = int(selected[1])
            found = True
        else:
            found = False
            if crop_mode == "hard_background":
                score = center_sums * 1e6 - temporal_sums
            else:
                score = temporal_sums
            flat_index = int(torch.argmin(score))
            y0 = flat_index // score.shape[1]
            x0 = flat_index % score.shape[1]

        if crop_mode == "hard_background" and not found:
            crop_mode = "background_fallback"
        elif crop_mode == "target" and target_coords.numel() == 0:
            crop_mode = "background_fallback"

    x1 = x0 + crop_width
    y1 = y0 + crop_height
    frames = [frame[..., y0:y1, x0:x1] for frame in frames]
    center_mask = center_mask[..., y0:y1, x0:x1]

    valid = torch.ones((1, source_height, source_width), dtype=center_mask.dtype)
    if pad_width or pad_height:
        valid = torch.nn.functional.pad(
            valid,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
    valid = valid[..., y0:y1, x0:x1]
    return frames, center_mask, valid, crop_mode


def _attenuate_temporal_targets(
    frames: list[torch.Tensor],
    masks: list[torch.Tensor],
    probability: float,
    min_factor: float,
    max_factor: float,
) -> list[torch.Tensor]:
    """Synthesize weak targets while preserving their temporal trajectories."""

    if probability <= 0.0 or random.random() >= probability:
        return frames
    factor = random.uniform(min_factor, max_factor)
    augmented = []
    for frame, target in zip(frames, masks):
        target = (target > 0.5).to(frame.dtype)
        if target.sum() == 0:
            augmented.append(frame)
            continue
        dilated = torch.nn.functional.max_pool2d(
            target.unsqueeze(0),
            kernel_size=15,
            stride=1,
            padding=7,
        ).squeeze(0)
        ring = (dilated - target).clamp(0.0, 1.0)
        background = (frame * ring).sum() / ring.sum().clamp_min(1.0)
        weakened = background + factor * (frame - background)
        augmented.append((frame * (1.0 - target) + weakened * target).clamp(0.0, 1.0))
    return augmented


class NUDTMIRSDTDataset(Dataset):
    """Temporal segmentation dataset loader for the NUDT-MIRSDT benchmark."""

    def __init__(
        self,
        root: str | Path,
        split: str | Path,
        num_frames: int = 5,
        size: tuple[int, int] | None = None,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        keep_aspect: bool = False,
        native_crop_size: tuple[int, int] | None = None,
        training: bool = False,
        weak_target_prob: float = 0.0,
        weak_target_min_factor: float = 0.2,
        weak_target_max_factor: float = 0.8,
        temporal_crop_strategy: str = "target",
        target_crop_probability: float = 0.5,
        background_crop_probability: float = 0.25,
        hard_background_crop_probability: float = 0.25,
        crop_max_attempts: int = 64,
    ) -> None:
        if num_frames % 2 != 1:
            raise ValueError("num_frames must be odd so that a center frame exists.")

        self.root = Path(root).resolve()
        self.size = size
        self.num_frames = num_frames
        self.radius = num_frames // 2
        self.image_dir_name = image_dir_name
        self.mask_dir_name = mask_dir_name
        self.keep_aspect = keep_aspect
        self.native_crop_size = native_crop_size
        self.training = training
        self.weak_target_prob = weak_target_prob
        self.weak_target_min_factor = weak_target_min_factor
        self.weak_target_max_factor = weak_target_max_factor
        self.temporal_crop_strategy = temporal_crop_strategy
        self.target_crop_probability = target_crop_probability
        self.background_crop_probability = background_crop_probability
        self.hard_background_crop_probability = hard_background_crop_probability
        self.crop_max_attempts = crop_max_attempts
        if not 0.0 <= weak_target_prob <= 1.0:
            raise ValueError("weak_target_prob must be between zero and one.")
        if not 0.0 < weak_target_min_factor <= weak_target_max_factor <= 1.0:
            raise ValueError("Weak-target factors must satisfy 0 < min <= max <= 1.")
        if temporal_crop_strategy not in {"target", "balanced_temporal"}:
            raise ValueError(
                "temporal_crop_strategy must be 'target' or 'balanced_temporal'."
            )
        crop_probability_sum = (
            target_crop_probability
            + background_crop_probability
            + hard_background_crop_probability
        )
        if abs(crop_probability_sum - 1.0) > 1e-6:
            raise ValueError("Temporal crop probabilities must sum to one.")

        self.sequence_to_frames: dict[str, tuple[str, ...]] = {}
        for seq_dir in sorted(path for path in self.root.iterdir() if path.is_dir() and path.name.startswith("Sequence")):
            image_dir = seq_dir / self.image_dir_name
            mask_dir = seq_dir / self.mask_dir_name
            if not image_dir.is_dir() or not mask_dir.is_dir():
                continue

            frame_stems = sorted(path.stem for path in image_dir.glob("*.png") if (mask_dir / path.name).exists())
            if frame_stems:
                self.sequence_to_frames[seq_dir.name] = tuple(frame_stems)

        if not self.sequence_to_frames:
            raise FileNotFoundError(
                f"No NUDT-MIRSDT sequences found under {self.root}. "
                f"Expected directories like Sequence1/images and Sequence1/masks."
            )

        selected = _read_split(Path(split).resolve())
        self.samples: list[NUDTSampleRef] = []
        for sequence_name, selected_frame_stems in selected.items():
            available = self.sequence_to_frames.get(sequence_name)
            if available is None:
                continue
            available_set = set(available)
            frame_stems = (
                available
                if selected_frame_stems is None
                else sorted(selected_frame_stems)
            )
            for frame_stem in frame_stems:
                if frame_stem in available_set:
                    self.samples.append(NUDTSampleRef(sequence_name=sequence_name, frame_stem=frame_stem))

        if not self.samples:
            raise FileNotFoundError("No valid NUDT-MIRSDT samples were found after applying the split file.")

        self.target_samples: list[NUDTSampleRef] = []
        self.pure_background_samples: list[NUDTSampleRef] = []
        self.hard_background_samples: list[NUDTSampleRef] = []
        if self.training and self.temporal_crop_strategy == "balanced_temporal":
            presence_by_sequence: dict[str, list[bool]] = {}
            for sequence_name, frame_stems in self.sequence_to_frames.items():
                mask_dir = self.root / sequence_name / self.mask_dir_name
                presence_by_sequence[sequence_name] = [
                    bool(
                        _read_gray(
                            mask_dir / f"{frame_stem}.png",
                            size=None,
                            is_mask=True,
                        ).sum()
                    )
                    for frame_stem in frame_stems
                ]

            for sample in self.samples:
                frame_stems = self.sequence_to_frames[sample.sequence_name]
                frame_index = frame_stems.index(sample.frame_stem)
                presence = presence_by_sequence[sample.sequence_name]
                temporal_presence = [
                    presence[index]
                    for index in self._window_indices(frame_index, len(frame_stems))
                ]
                if presence[frame_index]:
                    self.target_samples.append(sample)
                elif any(temporal_presence):
                    self.hard_background_samples.append(sample)
                else:
                    self.pure_background_samples.append(sample)

            if not self.target_samples:
                raise ValueError("Balanced temporal cropping found no target samples.")
            if not self.pure_background_samples:
                raise ValueError(
                    "Balanced temporal cropping found no all-background clips."
                )
            if not self.hard_background_samples:
                raise ValueError(
                    "Balanced temporal cropping found no hard-background clips."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _window_indices(self, frame_index: int, num_total: int) -> Iterable[int]:
        for offset in range(-self.radius, self.radius + 1):
            yield min(max(frame_index + offset, 0), num_total - 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        requested_crop_mode: str | None = None
        if self.training and self.temporal_crop_strategy == "balanced_temporal":
            draw = random.random()
            if draw < self.target_crop_probability:
                requested_crop_mode = "target"
                sample = random.choice(self.target_samples)
            elif draw < (
                self.target_crop_probability + self.background_crop_probability
            ):
                requested_crop_mode = "background"
                sample = random.choice(self.pure_background_samples)
            else:
                requested_crop_mode = "hard_background"
                sample = random.choice(self.hard_background_samples)
        else:
            sample = self.samples[index]
        frame_stems = self.sequence_to_frames[sample.sequence_name]
        frame_index = frame_stems.index(sample.frame_stem)

        seq_dir = self.root / sample.sequence_name
        image_dir = seq_dir / self.image_dir_name
        mask_dir = seq_dir / self.mask_dir_name
        window_indices = list(self._window_indices(frame_index, len(frame_stems)))

        if self.keep_aspect and self.size is not None:
            frame_items = [
                _read_gray_letterbox(
                    image_dir / f"{frame_stems[i]}.png",
                    self.size,
                    is_mask=False,
                )
                for i in window_indices
            ]
            frames = [item[0] for item in frame_items]
            mask, valid_mask = _read_gray_letterbox(
                mask_dir / f"{sample.frame_stem}.png",
                self.size,
                is_mask=True,
            )
        else:
            frames = [
                _read_gray(image_dir / f"{frame_stems[i]}.png", self.size, is_mask=False)
                for i in window_indices
            ]
            mask = _read_gray(mask_dir / f"{sample.frame_stem}.png", self.size, is_mask=True)
            valid_mask = torch.ones_like(mask)

        needs_temporal_masks = (
            self.temporal_crop_strategy == "balanced_temporal"
            or (
                self.training
                and self.weak_target_prob > 0.0
            )
        )
        temporal_masks = (
            [
                _read_gray(mask_dir / f"{frame_stems[i]}.png", self.size, is_mask=True)
                for i in window_indices
            ]
            if needs_temporal_masks
            else []
        )

        apply_weak_target = (
            self.training
            and self.weak_target_prob > 0.0
            and random.random() < self.weak_target_prob
        )
        if apply_weak_target:
            frames = _attenuate_temporal_targets(
                frames,
                temporal_masks,
                probability=1.0,
                min_factor=self.weak_target_min_factor,
                max_factor=self.weak_target_max_factor,
            )

        if self.training and self.native_crop_size is not None:
            if self.size is not None:
                raise ValueError("native_crop_size requires size=None so pixels are not rescaled.")
            if self.temporal_crop_strategy == "balanced_temporal":
                frames, mask, valid_mask, crop_mode = _balanced_temporal_crop(
                    frames,
                    temporal_masks,
                    mask,
                    self.native_crop_size,
                    self.target_crop_probability,
                    self.background_crop_probability,
                    self.hard_background_crop_probability,
                    self.crop_max_attempts,
                    forced_mode=requested_crop_mode,
                )
            else:
                frames, mask, valid_mask = _native_target_crop(
                    frames,
                    mask,
                    self.native_crop_size,
                )
                crop_mode = "target"
        else:
            crop_mode = "full"

        return {
            "frames": torch.stack(frames, dim=0),
            "mask": mask,
            "valid_mask": valid_mask,
            "crop_mode": crop_mode,
            "root": self.root.name,
            "sequence": sample.sequence_name,
            "frame": sample.frame_stem,
        }
