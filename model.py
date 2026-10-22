from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== Source: modelv5.py =====

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.stage = nn.Sequential(
            ConvBNAct(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class DualEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.s1 = EncoderStage(in_channels, base_channels, downsample=False)
        self.s2 = EncoderStage(base_channels, base_channels * 2, downsample=True)
        self.s3 = EncoderStage(base_channels * 2, base_channels * 4, downsample=True)
        self.s4 = EncoderStage(base_channels * 4, base_channels * 8, downsample=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return [f1, f2, f3, f4]


class DepthwiseCVO(nn.Module):
    """Fixed contrast vector operator adapted from DBCE U-Net."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        kernel_x = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]], dtype=torch.float32)
        kernel_y = torch.tensor([[2.0, 1.0], [-2.0, -1.0]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 2, 2).repeat(channels, 1, 1, 1))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 2, 2).repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(0, 1, 0, 1), mode="replicate")
        x_x = F.conv2d(x, self.kernel_x, groups=self.channels)
        x_y = F.conv2d(x, self.kernel_y, groups=self.channels)
        return torch.sqrt(torch.clamp(torch.abs(x_x) + torch.abs(x_y), min=self.eps))


class GatedContrastConv(nn.Module):
    """CVO enhancement controlled by a learnable residual gate."""

    def __init__(self, channels: int, scale_init: float = 0.1) -> None:
        super().__init__()
        left_channels = channels // 2
        right_channels = channels - left_channels
        self.left_channels = left_channels
        self.right_channels = right_channels
        self.contrast = DepthwiseCVO(right_channels)
        self.conv_branch = nn.Sequential(
            ConvBNAct(left_channels, left_channels, kernel_size=1),
            nn.Conv2d(left_channels, left_channels, kernel_size=3, padding=1, groups=left_channels, bias=False),
            nn.BatchNorm2d(left_channels),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = torch.split(x, [self.left_channels, self.right_channels], dim=1)
        left = self.conv_branch(left)
        right = right + self.contrast(right)
        enhanced = self.fuse(torch.cat([left, right], dim=1))
        gate = self.gate(torch.cat([x, enhanced], dim=1))
        return x + torch.tanh(self.scale) * gate * enhanced


class ChannelAttentionRefine(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, scale_init: float = 0.1) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.scale) * self.attn(x) * x


class FeatureAlignedTemporalMotion(nn.Module):
    """Aligns neighbouring frame features before building the motion cue."""

    def __init__(self, in_channels: int, feature_channels: int, max_offset: float = 3.0) -> None:
        super().__init__()
        self.max_offset = max_offset
        self.frame_stem = nn.Sequential(
            ConvBNAct(in_channels, feature_channels),
            ResidualBlock(feature_channels),
        )
        self.offset_net = nn.Sequential(
            ConvBNAct(feature_channels * 3, feature_channels),
            nn.Conv2d(feature_channels, 2, kernel_size=3, padding=1),
        )
        self.score_net = nn.Sequential(
            ConvBNAct(feature_channels * 3, max(feature_channels // 2, 8), kernel_size=1),
            nn.Conv2d(max(feature_channels // 2, 8), 1, kernel_size=1),
        )
        self.motion_refine = nn.Sequential(
            ConvBNAct(feature_channels, feature_channels),
            ResidualBlock(feature_channels),
        )

    @staticmethod
    def _base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([x_grid, y_grid], dim=-1)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    def _align(self, neighbour: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = neighbour.shape
        diff = torch.abs(neighbour - center)
        offset = torch.tanh(self.offset_net(torch.cat([center, neighbour, diff], dim=1))) * self.max_offset
        norm_x = offset[:, 0] * (2.0 / max(width - 1, 1))
        norm_y = offset[:, 1] * (2.0 / max(height - 1, 1))
        grid = self._base_grid(batch, height, width, neighbour.device, neighbour.dtype)
        grid = grid + torch.stack([norm_x, norm_y], dim=-1)
        return F.grid_sample(neighbour, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_frames, channels, height, width = frames.shape
        features = self.frame_stem(frames.reshape(batch * num_frames, channels, height, width))
        features = features.reshape(batch, num_frames, -1, height, width)
        center_index = num_frames // 2
        center = features[:, center_index]
        if num_frames == 1:
            return torch.zeros_like(center), center

        aligned_features = []
        score_logits = []
        for index in range(num_frames):
            if index == center_index:
                continue
            aligned = self._align(features[:, index], center)
            diff = torch.abs(aligned - center)
            aligned_features.append(diff)
            score_logits.append(self.score_net(torch.cat([center, aligned, diff], dim=1)))

        diffs = torch.stack(aligned_features, dim=1)
        scores = torch.stack(score_logits, dim=1)
        weights = torch.softmax(scores, dim=1)
        motion = torch.sum(weights * diffs, dim=1)
        return self.motion_refine(motion), center


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, height, width = x.shape
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


class TargetBackgroundFrequencyPurification(nn.Module):
    """Separates low-frequency context and CVO-enhanced high-frequency candidates."""

    def __init__(
        self,
        channels: int,
        use_frequency_contrast: bool = True,
        contrast_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.low_gate = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.motion_gate = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.background_gate = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.high_contrast = (
            GatedContrastConv(channels * 3, scale_init=contrast_scale_init)
            if use_frequency_contrast
            else nn.Identity()
        )
        self.high_proj = nn.Sequential(
            ConvBNAct(channels * 3, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out = nn.Sequential(ResidualBlock(channels), ConvBNAct(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor, motion: torch.Tensor | None = None) -> torch.Tensor:
        height, width = x.shape[-2:]
        ll, lh, hl, hh = haar_dwt(x)
        ll = F.interpolate(ll, size=(height, width), mode="bilinear", align_corners=False)
        high = torch.cat([lh, hl, hh], dim=1)
        high = F.interpolate(high, size=(height, width), mode="bilinear", align_corners=False)

        if motion is None:
            motion = torch.zeros_like(x)
        elif motion.shape[-2:] != (height, width):
            motion = F.interpolate(motion, size=(height, width), mode="bilinear", align_corners=False)

        target_gate = self.low_gate(ll) * self.motion_gate(motion)
        background_gate = self.background_gate(torch.cat([ll, motion], dim=1))
        high = self.high_contrast(high)
        high = self.high_proj(high) * target_gate * (1.0 - 0.5 * background_gate)
        return self.out(x + high)


class TemporalFrequencyFusionV5(nn.Module):
    def __init__(
        self,
        channels: int,
        use_temporal: bool = True,
        use_frequency: bool = True,
        use_shallow_contrast: bool = True,
        use_frequency_contrast: bool = True,
        contrast_scale_init: float = 0.1,
        fusion_refine_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.motion_gate = (
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_temporal
            else None
        )
        self.stage_refine = (
            GatedContrastConv(channels, scale_init=contrast_scale_init)
            if use_shallow_contrast
            else ChannelAttentionRefine(channels, scale_init=contrast_scale_init)
        )
        self.bff_refine = (
            nn.Sequential(
                ConvBNAct(channels * 4, channels, kernel_size=1),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            if use_temporal
            else None
        )
        self.refine_gate = (
            nn.Sequential(
                ConvBNAct(channels * 2, channels, kernel_size=1),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_temporal
            else None
        )
        self.refine_scale = nn.Parameter(torch.tensor(fusion_refine_scale_init, dtype=torch.float32))
        self.freq = (
            TargetBackgroundFrequencyPurification(
                channels,
                use_frequency_contrast=use_frequency_contrast,
                contrast_scale_init=contrast_scale_init,
            )
            if use_frequency
            else nn.Identity()
        )
        self.out = ResidualBlock(channels)

    def forward(self, appearance: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        if self.use_temporal:
            temporal_gate = self.motion_gate(torch.cat([appearance, motion], dim=1))
            base = appearance + temporal_gate * motion
            contrast_feature = self.stage_refine(base)
            diff_feature = torch.abs(appearance - motion)
            refine = self.bff_refine(torch.cat([appearance, motion, contrast_feature, diff_feature], dim=1))
            refine_gate = self.refine_gate(torch.cat([base, refine], dim=1))
            fused = base + torch.tanh(self.refine_scale) * refine_gate * refine
            fused = self.freq(fused, motion) if isinstance(self.freq, TargetBackgroundFrequencyPurification) else self.freq(fused)
        else:
            fused = self.stage_refine(appearance)
            fused = self.freq(fused, None) if isinstance(self.freq, TargetBackgroundFrequencyPurification) else self.freq(fused)
        return self.out(fused)


class CrossAttentionSkipUpBlock(nn.Module):
    """Lightweight deep-to-shallow skip filtering for small target boundaries."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.query = ConvBNAct(in_channels, skip_channels, kernel_size=1)
        self.key = ConvBNAct(skip_channels, skip_channels, kernel_size=1)
        self.value = ConvBNAct(skip_channels, skip_channels, kernel_size=1)
        self.skip_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.conv = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        query = self.query(x)
        key = self.key(skip)
        value = self.value(skip)
        attn = torch.sigmoid(query * key)
        skip = skip + torch.tanh(self.skip_scale) * attn * value
        return self.conv(torch.cat([x, skip], dim=1))


class HRTargetRefine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvBNAct(channels * 2, channels),
            ResidualBlock(channels),
        )

    def forward(self, decoder_feature: torch.Tensor, center_feature: torch.Tensor) -> torch.Tensor:
        if center_feature.shape[-2:] != decoder_feature.shape[-2:]:
            center_feature = F.interpolate(center_feature, size=decoder_feature.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([decoder_feature, center_feature], dim=1))
        center_feature = center_feature * gate
        return self.refine(torch.cat([decoder_feature, center_feature], dim=1))


class TFGNetV5(nn.Module):
    """Temporal-aligned, contrast-enhanced and center-supervised IR small target detector."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        use_temporal: bool = True,
        use_frequency: bool = True,
        contrast_stages: tuple[str, ...] = ("s1", "s2"),
        use_frequency_contrast: bool = True,
        use_aux_heads: bool = True,
        max_offset: float = 3.0,
        contrast_scale_init: float = 0.1,
        fusion_refine_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.use_aux_heads = use_aux_heads
        self.temporal_align = FeatureAlignedTemporalMotion(in_channels, base_channels, max_offset=max_offset)
        self.appearance_encoder = DualEncoder(in_channels, base_channels)
        self.motion_encoder = DualEncoder(base_channels, base_channels) if use_temporal else None

        stage_names = ("s1", "s2", "s3", "s4")
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.fusions = nn.ModuleList(
            [
                TemporalFrequencyFusionV5(
                    c,
                    use_temporal=use_temporal,
                    use_frequency=use_frequency,
                    use_shallow_contrast=stage in contrast_stages,
                    use_frequency_contrast=use_frequency_contrast and stage in contrast_stages,
                    contrast_scale_init=contrast_scale_init,
                    fusion_refine_scale_init=fusion_refine_scale_init,
                )
                for stage, c in zip(stage_names, channels)
            ]
        )

        self.up3 = CrossAttentionSkipUpBlock(channels[3], channels[2], channels[2])
        self.up2 = CrossAttentionSkipUpBlock(channels[2], channels[1], channels[1])
        self.up1 = CrossAttentionSkipUpBlock(channels[1], channels[0], channels[0])
        self.hr_refine = HRTargetRefine(channels[0])
        self.head_feature = ConvBNAct(channels[0], channels[0])
        self.seg_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.center_head = nn.Sequential(
            ConvBNAct(channels[0], channels[0]),
            nn.Conv2d(channels[0], 1, kernel_size=1),
        )
        self.objectness_head = nn.Sequential(
            ConvBNAct(channels[0], channels[0]),
            nn.Conv2d(channels[0], 1, kernel_size=1),
        )
        self.center_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.objectness_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, frames: torch.Tensor, return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError("TFGNetV5 expects input shape B x T x 1 x H x W.")

        center = frames[:, frames.shape[1] // 2]
        motion, center_feature = self.temporal_align(frames)
        appearance_features = self.appearance_encoder(center)
        if self.use_temporal:
            motion_features = self.motion_encoder(motion)
        else:
            motion_features = [torch.zeros_like(feature) for feature in appearance_features]

        fused = [fusion(a, m) for fusion, a, m in zip(self.fusions, appearance_features, motion_features)]
        x = self.up3(fused[3], fused[2])
        x = self.up2(x, fused[1])
        x = self.up1(x, fused[0])
        x = self.hr_refine(x, center_feature)
        x = self.head_feature(x)

        seg_logits = self.seg_head(x)
        center_logits = self.center_head(x)
        objectness_logits = self.objectness_head(x)
        logits = (
            seg_logits
            + torch.tanh(self.center_scale) * center_logits
            + torch.tanh(self.objectness_scale) * objectness_logits
        )

        if return_aux:
            return {
                "logits": logits,
                "seg_logits": seg_logits,
                "center_logits": center_logits,
                "objectness_logits": objectness_logits,
            }
        return logits


# ===== Source: modelv6.py =====

class MSFFBlock(nn.Module):
    """LVNet-style multi-scale low-level feature fusion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch3 = ConvBNAct(in_channels, out_channels, kernel_size=3)
        self.branch5 = ConvBNAct(in_channels, out_channels, kernel_size=5)
        self.branch7 = ConvBNAct(in_channels, out_channels, kernel_size=7)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        return self.act(self.proj(x) + self.fuse(multi_scale))


class SpaceToDepthDownsample(nn.Module):
    """Downsamples while retaining local spatial information in channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return self.proj(F.pixel_unshuffle(x, downscale_factor=2))


class LVLowLevelEncoder(nn.Module):
    """A shallow Conv U-Net-inspired encoder for local target details."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.s1 = nn.Sequential(
            MSFFBlock(in_channels, base_channels),
            ResidualBlock(base_channels),
        )
        self.down2 = SpaceToDepthDownsample(base_channels, base_channels * 2)
        self.s2 = nn.Sequential(
            MSFFBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2),
        )
        self.down3 = SpaceToDepthDownsample(base_channels * 2, base_channels * 4)
        self.s3 = nn.Sequential(
            MSFFBlock(base_channels * 4, base_channels * 4),
            ResidualBlock(base_channels * 4),
        )
        self.down4 = SpaceToDepthDownsample(base_channels * 4, base_channels * 8)
        self.s4 = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ChannelAttentionRefine(base_channels * 8),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.s1(x)
        f2 = self.s2(self.down2(f1))
        f3 = self.s3(self.down3(f2))
        f4 = self.s4(self.down4(f3))
        return [f1, f2, f3, f4]


class TemporalFrequencyFusionV6(nn.Module):
    def __init__(
        self,
        channels: int,
        use_temporal: bool = True,
        use_frequency: bool = True,
        use_shallow_contrast: bool = True,
        use_frequency_contrast: bool = True,
        contrast_scale_init: float = 0.08,
        fusion_refine_scale_init: float = 0.08,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.motion_gate = (
            nn.Sequential(
                ConvBNAct(channels * 2, channels, kernel_size=1),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_temporal
            else None
        )
        self.local_refine = (
            GatedContrastConv(channels, scale_init=contrast_scale_init)
            if use_shallow_contrast
            else ChannelAttentionRefine(channels, scale_init=contrast_scale_init)
        )
        self.msff = MSFFBlock(channels, channels) if use_shallow_contrast else ResidualBlock(channels)
        self.mix = (
            nn.Sequential(
                ConvBNAct(channels * 4, channels, kernel_size=1),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            if use_temporal
            else None
        )
        self.mix_gate = (
            nn.Sequential(
                ConvBNAct(channels * 2, channels, kernel_size=1),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_temporal
            else None
        )
        self.mix_scale = nn.Parameter(torch.tensor(fusion_refine_scale_init, dtype=torch.float32))
        self.freq = (
            TargetBackgroundFrequencyPurification(
                channels,
                use_frequency_contrast=use_frequency_contrast,
                contrast_scale_init=contrast_scale_init,
            )
            if use_frequency
            else nn.Identity()
        )
        self.out = ResidualBlock(channels)

    def forward(self, appearance: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        if self.use_temporal:
            temporal_gate = self.motion_gate(torch.cat([appearance, motion], dim=1))
            base = appearance + temporal_gate * motion
            local = self.msff(self.local_refine(base))
            diff = torch.abs(appearance - motion)
            mixed = self.mix(torch.cat([appearance, motion, local, diff], dim=1))
            gate = self.mix_gate(torch.cat([base, mixed], dim=1))
            fused = base + torch.tanh(self.mix_scale) * gate * mixed
            fused = self.freq(fused, motion) if isinstance(self.freq, TargetBackgroundFrequencyPurification) else self.freq(fused)
        else:
            fused = self.msff(self.local_refine(appearance))
            fused = self.freq(fused, None) if isinstance(self.freq, TargetBackgroundFrequencyPurification) else self.freq(fused)
        return self.out(fused)


class PatchExpandSkipBlock(nn.Module):
    """Patch-expanding decoder block inspired by LVNet."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, skip_channels * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
            ConvBNAct(skip_channels, skip_channels, kernel_size=3),
        )
        self.skip_gate = nn.Sequential(
            ConvBNAct(skip_channels * 2, skip_channels, kernel_size=1),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            MSFFBlock(skip_channels * 2, out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.skip_gate(torch.cat([x, skip], dim=1))
        skip = skip * gate
        return self.fuse(torch.cat([x, skip], dim=1))


class CompactMaskHead(nn.Module):
    """Predicts a compact mask while keeping center/objectness as auxiliary cues."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            MSFFBlock(channels * 2, channels),
            ResidualBlock(channels),
            ConvBNAct(channels, channels),
        )
        self.seg_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.center_head = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 1, kernel_size=1))
        self.objectness_head = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 1, kernel_size=1))
        self.boundary_head = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 1, kernel_size=1))
        self.center_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))

    def forward(self, x: torch.Tensor, low_level: torch.Tensor) -> dict[str, torch.Tensor]:
        if low_level.shape[-2:] != x.shape[-2:]:
            low_level = F.interpolate(low_level, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feature = self.refine(torch.cat([x, low_level], dim=1))
        seg_logits = self.seg_head(feature)
        center_logits = self.center_head(feature)
        objectness_logits = self.objectness_head(feature)
        boundary_logits = self.boundary_head(feature)
        logits = seg_logits + torch.tanh(self.center_scale) * center_logits
        return {
            "logits": logits,
            "seg_logits": seg_logits,
            "center_logits": center_logits,
            "objectness_logits": objectness_logits,
            "boundary_logits": boundary_logits,
        }


class TFGNetV6(nn.Module):
    """LVNet-inspired compact-mask version of TFGNet for high-IoU segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        use_temporal: bool = True,
        use_frequency: bool = True,
        contrast_stages: tuple[str, ...] = ("s1", "s2"),
        use_frequency_contrast: bool = True,
        max_offset: float = 3.0,
        contrast_scale_init: float = 0.08,
        fusion_refine_scale_init: float = 0.08,
        input_contrast_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.input_contrast_strength = input_contrast_strength
        self.temporal_align = FeatureAlignedTemporalMotion(in_channels, base_channels, max_offset=max_offset)
        self.appearance_encoder = LVLowLevelEncoder(in_channels, base_channels)
        self.motion_encoder = LVLowLevelEncoder(base_channels, base_channels) if use_temporal else None

        stage_names = ("s1", "s2", "s3", "s4")
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.fusions = nn.ModuleList(
            [
                TemporalFrequencyFusionV6(
                    c,
                    use_temporal=use_temporal,
                    use_frequency=use_frequency,
                    use_shallow_contrast=stage in contrast_stages,
                    use_frequency_contrast=use_frequency_contrast and stage in contrast_stages,
                    contrast_scale_init=contrast_scale_init,
                    fusion_refine_scale_init=fusion_refine_scale_init,
                )
                for stage, c in zip(stage_names, channels)
            ]
        )
        self.up3 = PatchExpandSkipBlock(channels[3], channels[2], channels[2])
        self.up2 = PatchExpandSkipBlock(channels[2], channels[1], channels[1])
        self.up1 = PatchExpandSkipBlock(channels[1], channels[0], channels[0])
        self.head = CompactMaskHead(channels[0])

    def forward(self, frames: torch.Tensor, return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError("TFGNetV6 expects input shape B x T x 1 x H x W.")

        if self.input_contrast_strength > 0.0:
            batch, num_frames, channels, height, width = frames.shape
            flat = frames.reshape(batch * num_frames, channels, height, width)
            local_mean = F.avg_pool2d(flat, kernel_size=9, stride=1, padding=4)
            residual = flat - local_mean
            local_var = F.avg_pool2d(
                residual.square(),
                kernel_size=9,
                stride=1,
                padding=4,
            )
            normalized = residual / torch.sqrt(local_var + 1e-4)
            flat = (flat + self.input_contrast_strength * normalized).clamp(0.0, 1.0)
            frames = flat.reshape(batch, num_frames, channels, height, width)

        center = frames[:, frames.shape[1] // 2]
        motion, center_feature = self.temporal_align(frames)
        appearance_features = self.appearance_encoder(center)
        if self.use_temporal:
            motion_features = self.motion_encoder(motion)
        else:
            motion_features = [torch.zeros_like(feature) for feature in appearance_features]

        fused = [fusion(a, m) for fusion, a, m in zip(self.fusions, appearance_features, motion_features)]
        x = self.up3(fused[3], fused[2])
        x = self.up2(x, fused[1])
        x = self.up1(x, fused[0])
        outputs = self.head(x, center_feature)
        return outputs if return_aux else outputs["logits"]


# ===== Source: modelv8.py =====

def _base_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def warp_feature(
    feature: torch.Tensor,
    flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp source features to reference coordinates.

    Flow is expressed in source pixels and maps each reference location to
    the location sampled from the source feature.
    """

    batch, _, height, width = feature.shape
    norm_x = flow[:, 0] * (2.0 / max(width - 1, 1))
    norm_y = flow[:, 1] * (2.0 / max(height - 1, 1))
    grid = _base_grid(batch, height, width, feature.device, feature.dtype)
    grid = grid + torch.stack([norm_x, norm_y], dim=-1)
    valid = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    ).unsqueeze(1)
    warped = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped, valid.to(feature.dtype)


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    old_height, old_width = flow.shape[-2:]
    new_height, new_width = size
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    resized[:, 0] *= new_width / max(old_width, 1)
    resized[:, 1] *= new_height / max(old_height, 1)
    return resized


class TemporalFrameEncoder(nn.Module):
    """Shared lightweight encoder for all frames at 1/2, 1/4 and 1/8 scale."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1, kernel_size=5, stride=2),
            ResidualBlock(c1),
        )
        self.s2 = nn.Sequential(
            ConvBNAct(c1, c2, kernel_size=3),
            ResidualBlock(c2),
        )
        self.s3 = nn.Sequential(
            ConvBNAct(c2, c3, kernel_size=3, stride=2),
            ResidualBlock(c3),
        )
        self.coarse = nn.Sequential(
            ConvBNAct(c3, c3, kernel_size=3, stride=2),
            ResidualBlock(c3),
        )

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_frames, channels, height, width = frames.shape
        flat = frames.reshape(batch * num_frames, channels, height, width)
        s2 = self.s2(self.stem(flat))
        s3 = self.s3(s2)
        coarse = self.coarse(s3)

        def restore(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch, num_frames, x.shape[1], x.shape[2], x.shape[3])

        return restore(s2), restore(s3), restore(coarse)


class LocalCorrelation(nn.Module):
    """Local matching volume used by the background-flow estimator."""

    def __init__(self, radius: int, temperature: float = 0.12) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("Correlation radius must be at least one.")
        self.radius = radius
        self.temperature = temperature
        offsets = [
            (float(dx), float(dy))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        self.register_buffer("offsets", torch.tensor(offsets).view(1, -1, 2, 1, 1))

    @property
    def channels(self) -> int:
        return (2 * self.radius + 1) ** 2

    def forward(
        self,
        reference: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = F.normalize(reference.float(), dim=1).to(reference.dtype)
        source = F.normalize(source.float(), dim=1).to(source.dtype)
        radius = self.radius
        height, width = reference.shape[-2:]
        padded = F.pad(source, (radius, radius, radius, radius), mode="replicate")
        correlations = []
        for dy in range(-radius, radius + 1):
            y0 = radius + dy
            for dx in range(-radius, radius + 1):
                x0 = radius + dx
                shifted = padded[..., y0 : y0 + height, x0 : x0 + width]
                correlations.append((reference * shifted).mean(dim=1, keepdim=True))
        volume = torch.cat(correlations, dim=1)
        probability = torch.softmax(volume / self.temperature, dim=1)
        expected_flow = (probability.unsqueeze(2) * self.offsets.to(probability)).sum(dim=1)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        confidence = 1.0 - entropy / math.log(float(self.channels))
        return volume, expected_flow, confidence.clamp(0.0, 1.0)


class CoarseToFineBackgroundFlow(nn.Module):
    """Two-level local-correlation flow dedicated to background alignment."""

    def __init__(
        self,
        fine_channels: int,
        coarse_channels: int,
        hidden_channels: int = 32,
        coarse_radius: int = 3,
        fine_radius: int = 2,
    ) -> None:
        super().__init__()
        self.fine_reduce = ConvBNAct(fine_channels, hidden_channels, kernel_size=1)
        self.coarse_reduce = ConvBNAct(coarse_channels, hidden_channels, kernel_size=1)
        self.coarse_correlation = LocalCorrelation(coarse_radius)
        self.fine_correlation = LocalCorrelation(fine_radius)
        self.coarse_match_scale = nn.Parameter(torch.tensor(0.0))
        self.fine_match_scale = nn.Parameter(torch.tensor(0.0))
        self.coarse_delta = nn.Sequential(
            ConvBNAct(hidden_channels * 2 + self.coarse_correlation.channels, hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )
        self.fine_delta = nn.Sequential(
            ConvBNAct(hidden_channels * 2 + self.fine_correlation.channels + 2, hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.coarse_delta[-1].weight)
        nn.init.zeros_(self.coarse_delta[-1].bias)
        nn.init.zeros_(self.fine_delta[-1].weight)
        nn.init.zeros_(self.fine_delta[-1].bias)

    def forward(
        self,
        reference_fine: torch.Tensor,
        source_fine: torch.Tensor,
        reference_coarse: torch.Tensor,
        source_coarse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_coarse = self.coarse_reduce(reference_coarse)
        source_coarse = self.coarse_reduce(source_coarse)
        coarse_volume, coarse_expected, _ = self.coarse_correlation(
            reference_coarse,
            source_coarse,
        )
        coarse_delta = 0.5 * torch.tanh(
            self.coarse_delta(
                torch.cat([reference_coarse, source_coarse, coarse_volume], dim=1)
            )
        )
        coarse_flow = torch.tanh(self.coarse_match_scale) * coarse_expected + coarse_delta

        reference_fine = self.fine_reduce(reference_fine)
        source_fine = self.fine_reduce(source_fine)
        initial_flow = resize_flow(coarse_flow, reference_fine.shape[-2:])
        warped_source, valid = warp_feature(source_fine, initial_flow)
        fine_volume, fine_expected, fine_confidence = self.fine_correlation(
            reference_fine,
            warped_source,
        )
        fine_delta = 0.5 * torch.tanh(
            self.fine_delta(
                torch.cat(
                    [
                        reference_fine,
                        warped_source,
                        fine_volume,
                        initial_flow,
                    ],
                    dim=1,
                )
            )
        )
        fine_flow = (
            initial_flow
            + torch.tanh(self.fine_match_scale) * fine_expected
            + fine_delta
        )
        return fine_flow, fine_confidence * valid


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros_like(x)
        reset, update = torch.sigmoid(self.gates(torch.cat([x, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class CenterBidirectionalConvGRU(nn.Module):
    """Reads past and future separately and summarizes them at the center frame."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.forward_cell = ConvGRUCell(channels)
        self.backward_cell = ConvGRUCell(channels)
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            ResidualBlock(channels),
        )

    def forward(self, sequence: torch.Tensor, center_index: int) -> torch.Tensor:
        forward_hidden = None
        for index in range(center_index + 1):
            forward_hidden = self.forward_cell(sequence[:, index], forward_hidden)
        backward_hidden = None
        for index in range(sequence.shape[1] - 1, center_index - 1, -1):
            backward_hidden = self.backward_cell(sequence[:, index], backward_hidden)
        return self.fuse(torch.cat([forward_hidden, backward_hidden], dim=1))


class TemporalDifferenceAggregator(nn.Module):
    """Temporal-difference convolution followed by frame-wise attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.mix = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.SiLU(inplace=True),
        )
        self.score = nn.Conv3d(channels, 1, kernel_size=1)
        self.out = ResidualBlock(channels)

    def forward(self, sequence: torch.Tensor, center_index: int) -> torch.Tensor:
        x = sequence.permute(0, 2, 1, 3, 4)
        x = self.mix(x + self.temporal(x))
        score = self.score(x)
        score[:, :, center_index] = -1e4
        weight = torch.softmax(score, dim=2)
        return self.out((weight * x).sum(dim=2))


class DifferenceProject(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            ResidualBlock(channels),
        )

    def forward(self, center: torch.Tensor, aligned: torch.Tensor) -> torch.Tensor:
        signed = center - aligned
        return self.project(torch.cat([signed, signed.abs()], dim=1))


def symmetric_pair_summary(sequence: torch.Tensor, center_index: int) -> torch.Tensor:
    pairs = []
    max_distance = min(center_index, sequence.shape[1] - center_index - 1)
    for distance in range(1, max_distance + 1):
        previous = sequence[:, center_index - distance]
        following = sequence[:, center_index + distance]
        pairs.append(0.5 * (previous + following) + 0.5 * (previous - following).abs())
    if not pairs:
        return torch.zeros_like(sequence[:, center_index])
    return torch.stack(pairs, dim=1).mean(dim=1)


class OpticalFlowDifferenceBranch(nn.Module):
    """Background flow, aligned differences, s2 TDC and s3 temporal persistence."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        consistency_temperature: float = 1.0,
        target_exclusion_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.consistency_temperature = consistency_temperature
        self.target_exclusion_kernel = target_exclusion_kernel
        self.detach_flow_for_motion = False
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.encoder = TemporalFrameEncoder(in_channels, base_channels)
        self.flow_estimator = CoarseToFineBackgroundFlow(
            fine_channels=c3,
            coarse_channels=c3,
        )
        self.s2_difference = DifferenceProject(c2)
        self.s3_difference = DifferenceProject(c3)
        self.s2_tdc = TemporalDifferenceAggregator(c2)
        self.s3_gru = CenterBidirectionalConvGRU(c3)
        self.s2_pair_scale = nn.Parameter(torch.tensor(0.10))
        self.s3_pair_scale = nn.Parameter(torch.tensor(0.10))
        self.s2_refine = ResidualBlock(c2)
        self.s3_refine = ResidualBlock(c3)

    @staticmethod
    def _local_normalize(image: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        variance = F.avg_pool2d((image - mean).square(), kernel_size=5, stride=1, padding=2)
        return (image - mean) / torch.sqrt(variance + 1e-4)

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (value * weight).sum() / (weight.sum() + 1e-6)

    def _flow_losses(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None,
        center_index: int,
        neighbour_indices: list[int],
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, _, _, _, _ = frames.shape
        num_neighbours = len(neighbour_indices)
        flow_height, flow_width = forward_flow.shape[-2:]
        center_image = frames[:, center_index]
        neighbours = frames[:, neighbour_indices].reshape(
            batch * num_neighbours,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )
        center_small = F.interpolate(
            center_image,
            size=(flow_height, flow_width),
            mode="bilinear",
            align_corners=True,
        ).repeat_interleave(num_neighbours, dim=0)
        neighbour_small = F.interpolate(
            neighbours,
            size=(flow_height, flow_width),
            mode="bilinear",
            align_corners=True,
        )
        aligned_neighbour, image_valid = warp_feature(neighbour_small, forward_flow)

        if target_mask is None:
            background = torch.ones_like(center_small)
        else:
            kernel = max(int(self.target_exclusion_kernel), 1)
            if kernel % 2 == 0:
                kernel += 1
            excluded = F.max_pool2d(
                target_mask.float(),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
            background = 1.0 - F.interpolate(
                excluded,
                size=(flow_height, flow_width),
                mode="nearest",
            )
            background = background.repeat_interleave(num_neighbours, dim=0)

        weight = background * valid * image_valid
        center_normalized = self._local_normalize(center_small)
        neighbour_normalized = self._local_normalize(neighbour_small)
        aligned_normalized = self._local_normalize(aligned_neighbour)
        before_error = (center_normalized - neighbour_normalized).abs()
        after_difference = center_normalized - aligned_normalized
        photometric = torch.sqrt(after_difference.square() + 1e-6) - 1e-3

        backward_warped, consistency_valid = warp_feature(backward_flow, forward_flow)
        consistency_error = torch.sqrt(
            (forward_flow + backward_warped).square().sum(dim=1, keepdim=True)
            + 1e-6
        )
        consistency_weight = weight * consistency_valid

        dx = forward_flow[..., :, 1:] - forward_flow[..., :, :-1]
        dy = forward_flow[..., 1:, :] - forward_flow[..., :-1, :]
        image_dx = center_small[..., :, 1:] - center_small[..., :, :-1]
        image_dy = center_small[..., 1:, :] - center_small[..., :-1, :]
        smoothness = (
            (dx.abs() * torch.exp(-10.0 * image_dx.abs())).mean()
            + (dy.abs() * torch.exp(-10.0 * image_dy.abs())).mean()
        )

        return {
            "flow_photometric_loss": self._weighted_mean(photometric, weight),
            "flow_consistency_loss": self._weighted_mean(
                consistency_error,
                consistency_weight,
            ),
            "flow_smoothness_loss": smoothness,
            "alignment_before": self._weighted_mean(before_error, weight),
            "alignment_after": self._weighted_mean(after_difference.abs(), weight),
            "mean_flow": torch.linalg.vector_norm(forward_flow, dim=1).mean(),
            "motion_confidence": confidence.mean(),
        }

    def _difference_sequence(
        self,
        stage_features: torch.Tensor,
        center_index: int,
        neighbour_indices: list[int],
        flow: torch.Tensor,
        confidence: torch.Tensor,
        projector: DifferenceProject,
    ) -> torch.Tensor:
        batch, _, channels, height, width = stage_features.shape
        num_neighbours = len(neighbour_indices)
        center = stage_features[:, center_index]
        neighbours = stage_features[:, neighbour_indices].reshape(
            batch * num_neighbours,
            channels,
            height,
            width,
        )
        stage_flow = resize_flow(flow, (height, width))
        aligned, valid = warp_feature(neighbours, stage_flow)
        center_flat = center.repeat_interleave(num_neighbours, dim=0)
        difference = projector(center_flat, aligned)
        stage_confidence = F.interpolate(
            confidence,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        difference = difference * stage_confidence * valid
        difference = difference.reshape(batch, num_neighbours, channels, height, width)

        sequence = []
        neighbour_position = {index: position for position, index in enumerate(neighbour_indices)}
        for index in range(stage_features.shape[1]):
            if index == center_index:
                sequence.append(torch.zeros_like(center))
            else:
                sequence.append(difference[:, neighbour_position[index]])
        return torch.stack(sequence, dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, num_frames, _, _, _ = frames.shape
        center_index = num_frames // 2
        if num_frames < 3:
            raise ValueError("V8 temporal branch requires at least three frames.")

        s2, s3, coarse = self.encoder(frames)
        neighbour_indices = [index for index in range(num_frames) if index != center_index]
        num_neighbours = len(neighbour_indices)

        def flatten_neighbours(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            center = feature[:, center_index].repeat_interleave(num_neighbours, dim=0)
            neighbours = feature[:, neighbour_indices].reshape(
                batch * num_neighbours,
                feature.shape[2],
                feature.shape[3],
                feature.shape[4],
            )
            return center, neighbours

        center_fine, neighbour_fine = flatten_neighbours(s3)
        center_coarse, neighbour_coarse = flatten_neighbours(coarse)
        reference_fine = torch.cat([center_fine, neighbour_fine], dim=0)
        source_fine = torch.cat([neighbour_fine, center_fine], dim=0)
        reference_coarse = torch.cat([center_coarse, neighbour_coarse], dim=0)
        source_coarse = torch.cat([neighbour_coarse, center_coarse], dim=0)
        bidirectional_flow, correlation_confidence = self.flow_estimator(
            reference_fine,
            source_fine,
            reference_coarse,
            source_coarse,
        )
        forward_flow, backward_flow = bidirectional_flow.chunk(2, dim=0)
        forward_correlation, backward_correlation = correlation_confidence.chunk(2, dim=0)

        backward_warped, consistency_valid = warp_feature(backward_flow, forward_flow)
        backward_correlation_warped, correlation_valid = warp_feature(
            backward_correlation,
            forward_flow,
        )
        consistency_error = torch.sqrt(
            (forward_flow + backward_warped).square().sum(dim=1, keepdim=True)
            + 1e-6
        )
        consistency_confidence = torch.exp(
            -consistency_error / max(self.consistency_temperature, 1e-3)
        )
        correlation_reliability = torch.sqrt(
            (forward_correlation * backward_correlation_warped).clamp_min(1e-6)
        )
        confidence = (
            (0.10 + 0.90 * correlation_reliability)
            * consistency_confidence
            * consistency_valid
            * correlation_valid
        ).clamp(0.0, 1.0)

        motion_flow = forward_flow.detach() if self.detach_flow_for_motion else forward_flow
        motion_confidence = confidence.detach() if self.detach_flow_for_motion else confidence
        s2_sequence = self._difference_sequence(
            s2,
            center_index,
            neighbour_indices,
            motion_flow,
            motion_confidence,
            self.s2_difference,
        )
        s3_sequence = self._difference_sequence(
            s3,
            center_index,
            neighbour_indices,
            motion_flow,
            motion_confidence,
            self.s3_difference,
        )

        s2_motion = self.s2_tdc(s2_sequence, center_index)
        s2_pair = symmetric_pair_summary(s2_sequence, center_index)
        s2_motion = self.s2_refine(
            s2_motion + torch.tanh(self.s2_pair_scale) * s2_pair
        )
        s3_motion = self.s3_gru(s3_sequence, center_index)
        s3_pair = symmetric_pair_summary(s3_sequence, center_index)
        s3_motion = self.s3_refine(
            s3_motion + torch.tanh(self.s3_pair_scale) * s3_pair
        )

        _, valid = warp_feature(neighbour_fine, forward_flow)
        diagnostics = self._flow_losses(
            frames,
            target_mask,
            center_index,
            neighbour_indices,
            forward_flow,
            backward_flow,
            confidence,
            valid,
        )
        return s2_motion, s3_motion, diagnostics


class ControlledTemporalFusion(nn.Module):
    """Adds temporal evidence without subtracting unrelated feature spaces."""

    def __init__(self, channels: int, use_frequency: bool = False) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.temporal_scale = nn.Parameter(torch.tensor(0.08))
        self.refine = ResidualBlock(channels)
        self.frequency = (
            TargetBackgroundFrequencyPurification(
                channels,
                use_frequency_contrast=False,
            )
            if use_frequency
            else None
        )

    def forward(self, appearance: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([appearance, temporal], dim=1))
        fused = self.refine(
            appearance + torch.tanh(self.temporal_scale) * gate * temporal
        )
        if self.frequency is not None:
            fused = self.frequency(fused, temporal)
        return fused


class BoundaryAwareMaskHead(nn.Module):
    """Boundary supervision refines the feature used by the final mask."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            MSFFBlock(channels * 2, channels),
            ResidualBlock(channels),
            ConvBNAct(channels, channels),
        )
        self.boundary_feature = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.SiLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.seg_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.center_head = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.objectness_head = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.boundary_scale = nn.Parameter(torch.tensor(0.05))
        self.center_scale = nn.Parameter(torch.tensor(0.08))

    def forward(self, x: torch.Tensor, low_level: torch.Tensor) -> dict[str, torch.Tensor]:
        if low_level.shape[-2:] != x.shape[-2:]:
            low_level = F.interpolate(
                low_level,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        feature = self.refine(torch.cat([x, low_level], dim=1))
        boundary_feature = self.boundary_feature(feature)
        boundary_logits = self.boundary_head(boundary_feature)
        boundary_attention = torch.sigmoid(boundary_logits)
        feature = feature + torch.tanh(self.boundary_scale) * boundary_attention * boundary_feature

        seg_logits = self.seg_head(feature)
        center_logits = self.center_head(feature)
        objectness_logits = self.objectness_head(feature)
        logits = seg_logits + torch.tanh(self.center_scale) * center_logits
        return {
            "logits": logits,
            "seg_logits": seg_logits,
            "center_logits": center_logits,
            "objectness_logits": objectness_logits,
            "boundary_logits": boundary_logits,
        }


class TFGNetV8(nn.Module):
    """V6 appearance path with background optical-flow differences at s2/s3."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        use_temporal: bool = True,
        use_frequency: bool = True,
        consistency_temperature: float = 1.0,
        target_exclusion_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.appearance_encoder = LVLowLevelEncoder(in_channels, base_channels)
        self.shallow_contrast = GatedContrastConv(channels[0], scale_init=0.08)
        self.temporal_branch = OpticalFlowDifferenceBranch(
            in_channels,
            base_channels,
            consistency_temperature=consistency_temperature,
            target_exclusion_kernel=target_exclusion_kernel,
        )
        self.s2_fusion = ControlledTemporalFusion(channels[1], use_frequency=False)
        self.s3_fusion = ControlledTemporalFusion(
            channels[2],
            use_frequency=use_frequency,
        )
        self.up3 = PatchExpandSkipBlock(channels[3], channels[2], channels[2])
        self.up2 = PatchExpandSkipBlock(channels[2], channels[1], channels[1])
        self.up1 = PatchExpandSkipBlock(channels[1], channels[0], channels[0])
        self.head = BoundaryAwareMaskHead(channels[0])

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError("TFGNetV8 expects input shape B x T x 1 x H x W.")

        center = frames[:, frames.shape[1] // 2]
        appearance = self.appearance_encoder(center)
        appearance[0] = self.shallow_contrast(appearance[0])

        if self.use_temporal:
            s2_motion, s3_motion, diagnostics = self.temporal_branch(
                frames,
                target_mask=target_mask,
            )
            appearance[1] = self.s2_fusion(appearance[1], s2_motion)
            appearance[2] = self.s3_fusion(appearance[2], s3_motion)
        else:
            zero = center.new_zeros(())
            diagnostics = {
                "flow_photometric_loss": zero,
                "flow_consistency_loss": zero,
                "flow_smoothness_loss": zero,
                "alignment_before": zero,
                "alignment_after": zero,
                "mean_flow": zero,
                "motion_confidence": zero,
            }

        x = self.up3(appearance[3], appearance[2])
        x = self.up2(x, appearance[1])
        x = self.up1(x, appearance[0])
        outputs = self.head(x, appearance[0])
        outputs.update(diagnostics)
        return outputs if return_aux else outputs["logits"]


# ===== Source: modelv7_2.py =====

class GlobalAffineCompensation(nn.Module):
    """Estimates one constrained global affine transform per neighboring frame."""

    def __init__(
        self,
        max_translation: float = 8.0,
        max_linear_delta: float = 0.06,
        alignment_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.max_translation = max_translation
        self.max_linear_delta = max_linear_delta
        self.alignment_size = alignment_size
        self.encoder = nn.Sequential(
            ConvBNAct(3, 16, kernel_size=5, stride=2),
            ConvBNAct(16, 24, kernel_size=3, stride=2),
            ConvBNAct(24, 32, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 6),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def _theta(self, center: torch.Tensor, neighbour: torch.Tensor) -> torch.Tensor:
        size = self.alignment_size
        center_small = F.interpolate(center, size=size, mode="bilinear", align_corners=False)
        neighbour_small = F.interpolate(neighbour, size=size, mode="bilinear", align_corners=False)
        pair = torch.cat([center_small, neighbour_small, (center_small - neighbour_small).abs()], dim=1)
        raw = torch.tanh(self.regressor(self.encoder(pair)))

        height, width = center.shape[-2:]
        max_tx = 2.0 * self.max_translation / max(width - 1, 1)
        max_ty = 2.0 * self.max_translation / max(height - 1, 1)
        theta = raw.new_zeros(raw.shape[0], 2, 3)
        theta[:, 0, 0] = 1.0 + self.max_linear_delta * raw[:, 0]
        theta[:, 0, 1] = self.max_linear_delta * raw[:, 1]
        theta[:, 0, 2] = max_tx * raw[:, 2]
        theta[:, 1, 0] = self.max_linear_delta * raw[:, 3]
        theta[:, 1, 1] = 1.0 + self.max_linear_delta * raw[:, 4]
        theta[:, 1, 2] = max_ty * raw[:, 5]
        return theta

    @staticmethod
    def _warp(image: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = F.affine_grid(theta, image.shape, align_corners=True)
        valid = (
            (grid[..., 0] >= -1.0)
            & (grid[..., 0] <= 1.0)
            & (grid[..., 1] >= -1.0)
            & (grid[..., 1] <= 1.0)
        ).unsqueeze(1)
        aligned = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return aligned, valid.to(image.dtype)

    @staticmethod
    def _trimmed_background_loss(
        error: torch.Tensor,
        weight: torch.Tensor,
        keep_ratio: float = 0.8,
    ) -> torch.Tensor:
        losses = []
        for sample_error, sample_weight in zip(error, weight):
            values = sample_error[sample_weight > 0.5]
            if values.numel() == 0:
                continue
            keep = max(int(values.numel() * keep_ratio), 1)
            losses.append(torch.topk(values, keep, largest=False).values.mean())
        if not losses:
            return error.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_frames, _, height, width = frames.shape
        center_index = num_frames // 2
        center = frames[:, center_index]
        if valid_mask is None:
            valid_mask = torch.ones_like(center)
        if target_mask is None:
            background_weight = valid_mask
        else:
            dilated_target = F.max_pool2d(target_mask, kernel_size=9, stride=1, padding=4)
            background_weight = valid_mask * (1.0 - dilated_target)

        aligned_frames = []
        alignment_losses = []
        regularization_losses = []
        identity = center.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0)

        for index in range(num_frames):
            if index == center_index:
                aligned_frames.append(center)
                continue
            neighbour = frames[:, index]
            theta = self._theta(center, neighbour)
            aligned, valid = self._warp(neighbour, theta)
            error = torch.sqrt((center - aligned).square() + 1e-6)
            alignment_losses.append(
                self._trimmed_background_loss(error, background_weight * valid)
            )
            regularization_losses.append((theta - identity).square().mean())
            aligned_frames.append(aligned)

        aligned = torch.stack(aligned_frames, dim=1)
        alignment_loss = (
            torch.stack(alignment_losses).mean()
            if alignment_losses
            else center.new_zeros(())
        )
        regularization_loss = (
            torch.stack(regularization_losses).mean()
            if regularization_losses
            else center.new_zeros(())
        )
        return aligned, alignment_loss, regularization_loss


class TemporalBackgroundResidual(nn.Module):
    """Extracts background-compensated residuals and models persistence at s2."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        channels_s1 = base_channels
        channels_s2 = base_channels * 2
        channels_s3 = base_channels * 4
        self.s1 = nn.Sequential(
            ConvBNAct(2, channels_s1),
            ResidualBlock(channels_s1),
        )
        self.down2 = SpaceToDepthDownsample(channels_s1, channels_s2)
        self.s2 = nn.Sequential(
            ResidualBlock(channels_s2),
            ChannelAttentionRefine(channels_s2, scale_init=0.05),
        )
        self.s2_gru = CenterBidirectionalConvGRU(channels_s2)
        self.down3 = SpaceToDepthDownsample(channels_s2, channels_s3)
        self.s3 = nn.Sequential(
            ResidualBlock(channels_s3),
            ChannelAttentionRefine(channels_s3, scale_init=0.05),
        )
        self.confidence = nn.Sequential(
            ConvBNAct(3, channels_s1, kernel_size=3),
            nn.Conv2d(channels_s1, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        aligned_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
        batch, num_frames, _, height, width = aligned_frames.shape
        center_index = num_frames // 2
        background = aligned_frames.median(dim=1).values
        signed = aligned_frames - background.unsqueeze(1)
        absolute = signed.abs()
        residual_input = torch.cat([signed, absolute], dim=2)

        flat = residual_input.reshape(batch * num_frames, 2, height, width)
        s1_sequence = self.s1(flat).reshape(batch, num_frames, -1, height, width)
        s2_flat = self.s2(self.down2(s1_sequence.reshape(batch * num_frames, -1, height, width)))
        s2_sequence = s2_flat.reshape(
            batch,
            num_frames,
            -1,
            s2_flat.shape[-2],
            s2_flat.shape[-1],
        )
        s2_temporal = self.s2_gru(s2_sequence, center_index)
        s3_temporal = self.s3(self.down3(s2_temporal))

        center_absolute = absolute[:, center_index]
        temporal_mean = absolute.mean(dim=1)
        temporal_std = absolute.std(dim=1, unbiased=False)
        confidence_s1 = self.confidence(
            torch.cat([center_absolute, temporal_mean, temporal_std], dim=1)
        )
        confidence_s2 = F.interpolate(
            confidence_s1,
            size=s2_temporal.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        confidence_s3 = F.interpolate(
            confidence_s1,
            size=s3_temporal.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return (
            s1_sequence[:, center_index],
            s2_temporal,
            s3_temporal,
            [confidence_s1, confidence_s2, confidence_s3],
            background,
        )


class TemporalLowLevelContrastEnhancement(nn.Module):
    """Adds a tightly bounded low-level temporal contrast correction."""

    def __init__(self, channels: int, scale_init: float = 0.02) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            ConvBNAct(channels * 2 + 1, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(
        self,
        appearance: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if temporal.shape[-2:] != appearance.shape[-2:]:
            temporal = F.interpolate(
                temporal,
                size=appearance.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        confidence = F.interpolate(
            confidence,
            size=appearance.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        local_contrast = temporal - F.avg_pool2d(temporal, kernel_size=5, stride=1, padding=2)
        delta = self.delta(torch.cat([appearance, temporal, local_contrast], dim=1))
        gate = self.gate(torch.cat([appearance, temporal, confidence], dim=1))
        return appearance + torch.tanh(self.scale) * gate * confidence * delta


class ControlledTemporalFusion(nn.Module):
    """Injects temporal evidence through a small residual path."""

    def __init__(
        self,
        channels: int,
        use_contrast: bool,
        use_frequency: bool,
        scale_init: float = 0.04,
    ) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            ConvBNAct(channels * 2 + 1, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = (
            GatedContrastConv(channels, scale_init=0.06)
            if use_contrast
            else ChannelAttentionRefine(channels, scale_init=0.06)
        )
        self.frequency = (
            TargetBackgroundFrequencyPurification(
                channels,
                use_frequency_contrast=False,
                contrast_scale_init=0.06,
            )
            if use_frequency
            else nn.Identity()
        )
        self.out = ResidualBlock(channels)
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(
        self,
        appearance: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if temporal.shape[-2:] != appearance.shape[-2:]:
            temporal = F.interpolate(
                temporal,
                size=appearance.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        confidence = F.interpolate(
            confidence,
            size=appearance.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        difference = (appearance - temporal).abs()
        delta = self.delta(torch.cat([appearance, temporal, difference], dim=1))
        gate = self.gate(torch.cat([appearance, temporal, confidence], dim=1))
        fused = appearance + torch.tanh(self.scale) * gate * confidence * delta
        fused = self.refine(fused)
        if isinstance(self.frequency, TargetBackgroundFrequencyPurification):
            fused = self.frequency(fused, temporal)
        else:
            fused = self.frequency(fused)
        return self.out(fused)


class AppearanceOnlyStage(nn.Module):
    def __init__(self, channels: int, use_contrast: bool, use_frequency: bool) -> None:
        super().__init__()
        self.local = (
            GatedContrastConv(channels, scale_init=0.06)
            if use_contrast
            else ChannelAttentionRefine(channels, scale_init=0.06)
        )
        self.msff = MSFFBlock(channels, channels) if use_contrast else ResidualBlock(channels)
        self.frequency = (
            TargetBackgroundFrequencyPurification(
                channels,
                use_frequency_contrast=False,
                contrast_scale_init=0.06,
            )
            if use_frequency
            else nn.Identity()
        )
        self.out = ResidualBlock(channels)

    def forward(self, appearance: torch.Tensor) -> torch.Tensor:
        feature = self.msff(self.local(appearance))
        if isinstance(self.frequency, TargetBackgroundFrequencyPurification):
            feature = self.frequency(feature, None)
        else:
            feature = self.frequency(feature)
        return self.out(feature)


class SignedBoundaryRefinement(nn.Module):
    """Predicts a signed correction that can add or remove boundary pixels."""

    def __init__(self, channels: int, scale_init: float = 0.05) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(
        self,
        decoder: torch.Tensor,
        appearance_s1: torch.Tensor,
        center_feature: torch.Tensor,
    ) -> torch.Tensor:
        target_size = decoder.shape[-2:]
        if appearance_s1.shape[-2:] != target_size:
            appearance_s1 = F.interpolate(
                appearance_s1,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if center_feature.shape[-2:] != target_size:
            center_feature = F.interpolate(
                center_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        correction = self.refine(torch.cat([decoder, appearance_s1, center_feature], dim=1))
        return torch.tanh(self.scale) * torch.tanh(correction)


class TFGNetV72(nn.Module):
    """V6 appearance backbone with global compensation and controlled s2/s3 temporal fusion."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        use_temporal: bool = True,
        use_frequency: bool = True,
        max_translation: float = 8.0,
        max_linear_delta: float = 0.06,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.appearance_encoder = LVLowLevelEncoder(in_channels, base_channels)
        self.center_stem = nn.Sequential(
            ConvBNAct(in_channels, base_channels),
            ResidualBlock(base_channels),
        )
        self.compensation = GlobalAffineCompensation(
            max_translation=max_translation,
            max_linear_delta=max_linear_delta,
        )
        self.temporal_residual = TemporalBackgroundResidual(base_channels)
        self.tlce = TemporalLowLevelContrastEnhancement(channels[0], scale_init=0.02)

        self.s1_stage = AppearanceOnlyStage(channels[0], use_contrast=True, use_frequency=False)
        self.s2_fusion = ControlledTemporalFusion(
            channels[1],
            use_contrast=True,
            use_frequency=False,
            scale_init=0.04,
        )
        self.s3_fusion = ControlledTemporalFusion(
            channels[2],
            use_contrast=False,
            use_frequency=use_frequency,
            scale_init=0.04,
        )
        self.s4_stage = AppearanceOnlyStage(
            channels[3],
            use_contrast=False,
            use_frequency=use_frequency,
        )

        self.up3 = PatchExpandSkipBlock(channels[3], channels[2], channels[2])
        self.up2 = PatchExpandSkipBlock(channels[2], channels[1], channels[1])
        self.up1 = PatchExpandSkipBlock(channels[1], channels[0], channels[0])
        self.head = CompactMaskHead(channels[0])
        self.boundary_refine = SignedBoundaryRefinement(channels[0], scale_init=0.05)

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError("TFGNetV72 expects input shape B x T x 1 x H x W.")

        center = frames[:, frames.shape[1] // 2]
        center_feature = self.center_stem(center)
        appearance = self.appearance_encoder(center)

        if self.use_temporal and frames.shape[1] > 1:
            aligned, alignment_loss, affine_reg_loss = self.compensation(
                frames,
                target_mask=target_mask,
                valid_mask=valid_mask,
            )
            temporal_s1, temporal_s2, temporal_s3, confidence, background = (
                self.temporal_residual(aligned)
            )
            s1_input = self.tlce(appearance[0], temporal_s1, confidence[0])
            fused_s1 = self.s1_stage(s1_input)
            fused_s2 = self.s2_fusion(appearance[1], temporal_s2, confidence[1])
            fused_s3 = self.s3_fusion(appearance[2], temporal_s3, confidence[2])
            temporal_confidence = torch.stack([item.mean() for item in confidence]).mean()
            background_residual = (center - background).abs().mean()
        else:
            fused_s1 = self.s1_stage(appearance[0])
            fused_s2 = appearance[1]
            fused_s3 = self.s3_fusion(
                appearance[2],
                torch.zeros_like(appearance[2]),
                torch.zeros_like(appearance[2][:, :1]),
            )
            alignment_loss = center.new_zeros(())
            affine_reg_loss = center.new_zeros(())
            temporal_confidence = center.new_zeros(())
            background_residual = center.new_zeros(())

        fused_s4 = self.s4_stage(appearance[3])
        x = self.up3(fused_s4, fused_s3)
        x = self.up2(x, fused_s2)
        x = self.up1(x, fused_s1)
        outputs = self.head(x, center_feature)
        boundary_correction = self.boundary_refine(x, appearance[0], center_feature)
        outputs["base_logits"] = outputs["logits"]
        outputs["boundary_correction"] = boundary_correction
        outputs["logits"] = outputs["logits"] + boundary_correction
        outputs["affine_alignment_loss"] = alignment_loss
        outputs["affine_reg_loss"] = affine_reg_loss
        outputs["temporal_confidence"] = temporal_confidence
        outputs["background_residual"] = background_residual
        return outputs if return_aux else outputs["logits"]


# ===== Source: modelv8_1.py =====

class SharedTemporalEncoder(nn.Module):
    """Shared full-resolution feature pyramid for all temporal frames."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.s1 = nn.Sequential(
            ConvBNAct(in_channels, channels[0], kernel_size=5),
            ResidualBlock(channels[0]),
        )
        self.down2 = SpaceToDepthDownsample(channels[0], channels[1])
        self.s2 = ResidualBlock(channels[1])
        self.down3 = SpaceToDepthDownsample(channels[1], channels[2])
        self.s3 = ResidualBlock(channels[2])
        self.down4 = SpaceToDepthDownsample(channels[2], channels[3])
        self.s4 = ResidualBlock(channels[3])

    def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
        batch, num_frames, channels, height, width = frames.shape
        flat = frames.reshape(batch * num_frames, channels, height, width)
        s1 = self.s1(flat)
        s2 = self.s2(self.down2(s1))
        s3 = self.s3(self.down3(s2))
        s4 = self.s4(self.down4(s3))

        def restore(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(
                batch,
                num_frames,
                x.shape[1],
                x.shape[2],
                x.shape[3],
            )

        return [restore(s1), restore(s2), restore(s3), restore(s4)]


class FrequencyAwareTemporalMemory(nn.Module):
    """Separates low/high wavelet cues before bidirectional temporal memory."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.low_project = ConvBNAct(channels, channels, kernel_size=1)
        self.high_project = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            ResidualBlock(channels),
        )
        self.high_gate = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.memory = CenterBidirectionalConvGRU(channels)
        self.refine = ResidualBlock(channels)
        self.high_scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))

    def forward(self, sequence: torch.Tensor, center_index: int) -> torch.Tensor:
        batch, num_frames, channels, height, width = sequence.shape
        flat = sequence.reshape(batch * num_frames, channels, height, width)
        ll, lh, hl, hh = haar_dwt(flat)
        low = self.low_project(ll)
        high = self.high_project(torch.cat([lh, hl, hh], dim=1))
        low = F.interpolate(
            low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        high = F.interpolate(
            high,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        gate = self.high_gate(torch.cat([low, high], dim=1))
        frequency_feature = (
            low + torch.tanh(self.high_scale) * gate * high
        ).reshape(batch, num_frames, channels, height, width)
        return self.refine(self.memory(frequency_feature, center_index))


class ResidualTemporalFusion(nn.Module):
    """Adds reliable temporal evidence through a bounded residual correction."""

    def __init__(
        self,
        channels: int,
        use_contrast: bool,
        scale_init: float,
    ) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            ConvBNAct(channels * 2 + 1, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = (
            GatedContrastConv(channels, scale_init=0.06)
            if use_contrast
            else ChannelAttentionRefine(channels, scale_init=0.06)
        )
        self.out = ResidualBlock(channels)
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        self.reliability_floor_logit = nn.Parameter(torch.tensor(-1.10))

    def forward(
        self,
        appearance: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if temporal.shape[-2:] != appearance.shape[-2:]:
            temporal = F.interpolate(
                temporal,
                size=appearance.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        confidence = F.interpolate(
            confidence,
            size=appearance.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        floor = torch.sigmoid(self.reliability_floor_logit)
        reliability = floor + (1.0 - floor) * confidence.clamp(0.0, 1.0)
        difference = (appearance - temporal).abs()
        delta = self.delta(torch.cat([appearance, temporal, difference], dim=1))
        gate = self.gate(torch.cat([appearance, temporal, reliability], dim=1))
        fused = appearance + torch.tanh(self.scale) * gate * reliability * delta
        return self.out(self.refine(fused))


class AppearanceStage(nn.Module):
    def __init__(self, channels: int, use_contrast: bool) -> None:
        super().__init__()
        self.local = (
            GatedContrastConv(channels, scale_init=0.06)
            if use_contrast
            else ChannelAttentionRefine(channels, scale_init=0.06)
        )
        self.context = (
            MSFFBlock(channels, channels)
            if use_contrast
            else ResidualBlock(channels)
        )
        self.out = ResidualBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.context(self.local(x)))


class BackgroundFlowDifferenceBranch(nn.Module):
    """Global compensation, background flow, ordered differences and memory."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        consistency_temperature: float,
        target_exclusion_kernel: int,
        max_translation: float,
        max_linear_delta: float,
    ) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.consistency_temperature = consistency_temperature
        self.target_exclusion_kernel = target_exclusion_kernel
        self.detach_alignment_for_motion = False
        self.affine = GlobalAffineCompensation(
            max_translation=max_translation,
            max_linear_delta=max_linear_delta,
        )
        self.encoder = SharedTemporalEncoder(in_channels, base_channels)
        self.flow_estimator = CoarseToFineBackgroundFlow(
            fine_channels=channels[2],
            coarse_channels=channels[3],
        )
        self.difference_projects = nn.ModuleList(
            [
                DifferenceProject(channels[0]),
                DifferenceProject(channels[1]),
                DifferenceProject(channels[2]),
            ]
        )
        self.s1_tdem = TemporalDifferenceAggregator(channels[0])
        self.s2_tdem = TemporalDifferenceAggregator(channels[1])
        self.s3_memory = FrequencyAwareTemporalMemory(channels[2])
        self.s1_refine = ResidualBlock(channels[0])
        self.s2_refine = ResidualBlock(channels[1])
        self.s3_refine = ResidualBlock(channels[2])

    @staticmethod
    def _local_normalize(image: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        variance = F.avg_pool2d(
            (image - mean).square(),
            kernel_size=5,
            stride=1,
            padding=2,
        )
        return (image - mean) / torch.sqrt(variance + 1e-4)

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (value * weight).sum() / (weight.sum() + 1e-6)

    def alignment_parameters(self) -> list[nn.Parameter]:
        return [
            *self.affine.parameters(),
            *self.encoder.parameters(),
            *self.flow_estimator.parameters(),
        ]

    def temporal_parameters(self) -> list[nn.Parameter]:
        alignment_ids = {id(parameter) for parameter in self.alignment_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in alignment_ids
        ]

    def _background_weight(
        self,
        target_mask: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        size: tuple[int, int],
        repeats: int,
    ) -> torch.Tensor:
        if target_mask is None:
            background = torch.ones(
                1,
                1,
                *size,
                device=next(self.parameters()).device,
            )
        else:
            kernel = max(int(self.target_exclusion_kernel), 1)
            if kernel % 2 == 0:
                kernel += 1
            excluded = F.max_pool2d(
                target_mask.float(),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
            background = 1.0 - F.interpolate(
                excluded,
                size=size,
                mode="nearest",
            )
        if valid_mask is not None:
            background = background * F.interpolate(
                valid_mask.float(),
                size=size,
                mode="nearest",
            )
        return background.repeat_interleave(repeats, dim=0)

    def _difference_sequence(
        self,
        stage_features: torch.Tensor,
        center_index: int,
        neighbour_indices: list[int],
        flow: torch.Tensor,
        confidence: torch.Tensor,
        projector: DifferenceProject,
    ) -> torch.Tensor:
        batch, _, channels, height, width = stage_features.shape
        num_neighbours = len(neighbour_indices)
        center = stage_features[:, center_index]
        neighbours = stage_features[:, neighbour_indices].reshape(
            batch * num_neighbours,
            channels,
            height,
            width,
        )
        stage_flow = resize_flow(flow, (height, width))
        aligned, valid = warp_feature(neighbours, stage_flow)
        center_flat = center.repeat_interleave(num_neighbours, dim=0)
        difference = projector(center_flat, aligned)
        stage_confidence = F.interpolate(
            confidence,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        reliability = 0.25 + 0.75 * stage_confidence
        difference = difference * reliability * valid
        difference = difference.reshape(
            batch,
            num_neighbours,
            channels,
            height,
            width,
        )

        neighbour_position = {
            index: position for position, index in enumerate(neighbour_indices)
        }
        ordered = []
        for index in range(stage_features.shape[1]):
            if index == center_index:
                ordered.append(torch.zeros_like(center))
            else:
                ordered.append(difference[:, neighbour_position[index]])
        return torch.stack(ordered, dim=1)

    def _diagnostics(
        self,
        raw_frames: torch.Tensor,
        affine_frames: torch.Tensor,
        target_mask: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        center_index: int,
        neighbour_indices: list[int],
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = raw_frames.shape[0]
        num_neighbours = len(neighbour_indices)
        flow_size = forward_flow.shape[-2:]

        def flatten_images(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            center = F.interpolate(
                frames[:, center_index],
                size=flow_size,
                mode="bilinear",
                align_corners=True,
            ).repeat_interleave(num_neighbours, dim=0)
            neighbours = F.interpolate(
                frames[:, neighbour_indices].reshape(
                    batch * num_neighbours,
                    frames.shape[2],
                    frames.shape[3],
                    frames.shape[4],
                ),
                size=flow_size,
                mode="bilinear",
                align_corners=True,
            )
            return center, neighbours

        raw_center, raw_neighbours = flatten_images(raw_frames)
        affine_center, affine_neighbours = flatten_images(affine_frames)
        flow_aligned, image_valid = warp_feature(affine_neighbours, forward_flow)
        background = self._background_weight(
            target_mask,
            valid_mask,
            flow_size,
            num_neighbours,
        )
        weight = background * image_valid

        raw_error = (
            self._local_normalize(raw_center)
            - self._local_normalize(raw_neighbours)
        ).abs()
        affine_error = (
            self._local_normalize(affine_center)
            - self._local_normalize(affine_neighbours)
        ).abs()
        flow_difference = (
            self._local_normalize(affine_center)
            - self._local_normalize(flow_aligned)
        )
        photometric = torch.sqrt(flow_difference.square() + 1e-6) - 1e-3

        backward_warped, consistency_valid = warp_feature(
            backward_flow,
            forward_flow,
        )
        consistency_error = torch.sqrt(
            (forward_flow + backward_warped).square().sum(dim=1, keepdim=True)
            + 1e-6
        )
        consistency_weight = weight * consistency_valid

        dx = forward_flow[..., :, 1:] - forward_flow[..., :, :-1]
        dy = forward_flow[..., 1:, :] - forward_flow[..., :-1, :]
        image_dx = affine_center[..., :, 1:] - affine_center[..., :, :-1]
        image_dy = affine_center[..., 1:, :] - affine_center[..., :-1, :]
        smoothness = (
            (dx.abs() * torch.exp(-10.0 * image_dx.abs())).mean()
            + (dy.abs() * torch.exp(-10.0 * image_dy.abs())).mean()
        )

        raw_alignment = self._weighted_mean(raw_error, background)
        affine_alignment = self._weighted_mean(affine_error, background)
        flow_alignment = self._weighted_mean(flow_difference.abs(), weight)
        return {
            "flow_photometric_loss": self._weighted_mean(photometric, weight),
            "flow_consistency_loss": self._weighted_mean(
                consistency_error,
                consistency_weight,
            ),
            "flow_smoothness_loss": smoothness,
            "alignment_raw": raw_alignment,
            "alignment_affine": affine_alignment,
            "alignment_flow": flow_alignment,
            "affine_gain": raw_alignment - affine_alignment,
            "flow_gain": affine_alignment - flow_alignment,
            "total_alignment_gain": raw_alignment - flow_alignment,
            "mean_flow": torch.linalg.vector_norm(forward_flow, dim=1).mean(),
            "motion_confidence": confidence.mean(),
        }

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
        alignment_only: bool = False,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        batch, num_frames = frames.shape[:2]
        center_index = num_frames // 2
        neighbour_indices = [
            index for index in range(num_frames) if index != center_index
        ]
        num_neighbours = len(neighbour_indices)

        affine_frames, affine_loss, affine_reg_loss = self.affine(
            frames,
            target_mask=target_mask,
            valid_mask=valid_mask,
        )
        pyramid = self.encoder(affine_frames)

        def flatten_neighbours(
            feature: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            center = feature[:, center_index].repeat_interleave(
                num_neighbours,
                dim=0,
            )
            neighbours = feature[:, neighbour_indices].reshape(
                batch * num_neighbours,
                feature.shape[2],
                feature.shape[3],
                feature.shape[4],
            )
            return center, neighbours

        center_fine, neighbour_fine = flatten_neighbours(pyramid[2])
        center_coarse, neighbour_coarse = flatten_neighbours(pyramid[3])
        bidirectional_flow, correlation_confidence = self.flow_estimator(
            torch.cat([center_fine, neighbour_fine], dim=0),
            torch.cat([neighbour_fine, center_fine], dim=0),
            torch.cat([center_coarse, neighbour_coarse], dim=0),
            torch.cat([neighbour_coarse, center_coarse], dim=0),
        )
        forward_flow, backward_flow = bidirectional_flow.chunk(2, dim=0)
        forward_correlation, backward_correlation = (
            correlation_confidence.chunk(2, dim=0)
        )

        backward_warped, consistency_valid = warp_feature(
            backward_flow,
            forward_flow,
        )
        backward_correlation_warped, correlation_valid = warp_feature(
            backward_correlation,
            forward_flow,
        )
        consistency_error = torch.sqrt(
            (forward_flow + backward_warped).square().sum(dim=1, keepdim=True)
            + 1e-6
        )
        consistency_confidence = torch.exp(
            -consistency_error / max(self.consistency_temperature, 1e-3)
        )
        correlation_reliability = torch.sqrt(
            (
                forward_correlation * backward_correlation_warped
            ).clamp_min(1e-6)
        )
        confidence = (
            correlation_reliability
            * consistency_confidence
            * consistency_valid
            * correlation_valid
        ).clamp(0.0, 1.0)

        diagnostics = self._diagnostics(
            frames,
            affine_frames,
            target_mask,
            valid_mask,
            center_index,
            neighbour_indices,
            forward_flow,
            backward_flow,
            confidence,
        )
        diagnostics["affine_alignment_loss"] = affine_loss
        diagnostics["affine_reg_loss"] = affine_reg_loss
        if alignment_only:
            return None, None, None, [], diagnostics

        motion_flow = (
            forward_flow.detach()
            if self.detach_alignment_for_motion
            else forward_flow
        )
        motion_confidence = (
            confidence.detach()
            if self.detach_alignment_for_motion
            else confidence
        )
        sequences = [
            self._difference_sequence(
                pyramid[stage_index],
                center_index,
                neighbour_indices,
                motion_flow,
                motion_confidence,
                self.difference_projects[stage_index],
            )
            for stage_index in range(3)
        ]
        s1_motion = self.s1_refine(
            self.s1_tdem(sequences[0], center_index)
        )
        s2_motion = self.s2_refine(
            self.s2_tdem(sequences[1], center_index)
        )
        s3_motion = self.s3_refine(
            self.s3_memory(sequences[2], center_index)
        )

        confidence_maps = [
            F.interpolate(
                confidence.reshape(
                    batch,
                    num_neighbours,
                    1,
                    *confidence.shape[-2:],
                ).mean(dim=1),
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for index in range(3)
        ]
        return s1_motion, s2_motion, s3_motion, confidence_maps, diagnostics


class TFGNetV81(nn.Module):
    """V6 appearance backbone with separated background-flow temporal roles."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        use_temporal: bool = True,
        consistency_temperature: float = 1.0,
        target_exclusion_kernel: int = 9,
        max_translation: float = 8.0,
        max_linear_delta: float = 0.06,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.appearance_encoder = LVLowLevelEncoder(in_channels, base_channels)
        self.center_stem = nn.Sequential(
            ConvBNAct(in_channels, channels[0]),
            ResidualBlock(channels[0]),
        )
        self.temporal_branch = BackgroundFlowDifferenceBranch(
            in_channels,
            base_channels,
            consistency_temperature=consistency_temperature,
            target_exclusion_kernel=target_exclusion_kernel,
            max_translation=max_translation,
            max_linear_delta=max_linear_delta,
        )
        self.s1_fusion = ResidualTemporalFusion(
            channels[0],
            use_contrast=True,
            scale_init=0.025,
        )
        self.s2_fusion = ResidualTemporalFusion(
            channels[1],
            use_contrast=True,
            scale_init=0.04,
        )
        self.s3_fusion = ResidualTemporalFusion(
            channels[2],
            use_contrast=False,
            scale_init=0.04,
        )
        self.s4_stage = AppearanceStage(channels[3], use_contrast=False)
        self.up3 = PatchExpandSkipBlock(channels[3], channels[2], channels[2])
        self.up2 = PatchExpandSkipBlock(channels[2], channels[1], channels[1])
        self.up1 = PatchExpandSkipBlock(channels[1], channels[0], channels[0])
        self.head = CompactMaskHead(channels[0])

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError("TFGNetV81 expects B x T x 1 x H x W input.")

        center = frames[:, frames.shape[1] // 2]
        appearance = self.appearance_encoder(center)
        center_feature = self.center_stem(center)

        if self.use_temporal and frames.shape[1] > 1:
            motion_s1, motion_s2, motion_s3, confidence, diagnostics = (
                self.temporal_branch(
                    frames,
                    target_mask=target_mask,
                    valid_mask=valid_mask,
                )
            )
            fused_s1 = self.s1_fusion(
                appearance[0],
                motion_s1,
                confidence[0],
            )
            fused_s2 = self.s2_fusion(
                appearance[1],
                motion_s2,
                confidence[1],
            )
            fused_s3 = self.s3_fusion(
                appearance[2],
                motion_s3,
                confidence[2],
            )
        else:
            zero = center.new_zeros(())
            fused_s1 = appearance[0]
            fused_s2 = appearance[1]
            fused_s3 = appearance[2]
            diagnostics = {
                key: zero
                for key in (
                    "flow_photometric_loss",
                    "flow_consistency_loss",
                    "flow_smoothness_loss",
                    "affine_alignment_loss",
                    "affine_reg_loss",
                    "alignment_raw",
                    "alignment_affine",
                    "alignment_flow",
                    "affine_gain",
                    "flow_gain",
                    "total_alignment_gain",
                    "mean_flow",
                    "motion_confidence",
                )
            }

        fused_s4 = self.s4_stage(appearance[3])
        x = self.up3(fused_s4, fused_s3)
        x = self.up2(x, fused_s2)
        x = self.up1(x, fused_s1)
        outputs = self.head(x, center_feature)
        outputs.update(diagnostics)
        return outputs if return_aux else outputs["logits"]


# ===== Source: modelv9.py =====

class OrderedTemporalAnomalyEncoder(nn.Module):
    """Encodes ordered temporal anomalies without geometric warping."""

    def __init__(self, base_channels: int, use_frequency_memory: bool = True) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
        ]
        self.encoder = SharedTemporalEncoder(4, base_channels)
        self.s1_tdem = TemporalDifferenceAggregator(channels[0])
        self.s2_tdem = TemporalDifferenceAggregator(channels[1])
        self.s3_memory = (
            FrequencyAwareTemporalMemory(channels[2])
            if use_frequency_memory
            else TemporalDifferenceAggregator(channels[2])
        )
        self.s1_refine = ResidualBlock(channels[0])
        self.s2_refine = ResidualBlock(channels[1])
        self.s3_refine = ResidualBlock(channels[2])
        self.reliability = nn.ModuleList(
            [TemporalReliabilityEstimator(channel) for channel in channels]
        )

    @staticmethod
    def _local_normalize(frames: torch.Tensor) -> torch.Tensor:
        batch, num_frames, channels, height, width = frames.shape
        flat = frames.reshape(batch * num_frames, channels, height, width)
        mean = F.avg_pool2d(flat, kernel_size=5, stride=1, padding=2)
        variance = F.avg_pool2d(
            (flat - mean).square(),
            kernel_size=5,
            stride=1,
            padding=2,
        )
        normalized = (flat - mean) / torch.sqrt(variance + 1e-4)
        return normalized.reshape(batch, num_frames, channels, height, width)

    @staticmethod
    def _ordered_cues(frames: torch.Tensor) -> torch.Tensor:
        center_index = frames.shape[1] // 2
        center = frames[:, center_index : center_index + 1]
        temporal_background = frames.median(dim=1, keepdim=True).values
        signed_center = center - frames
        absolute_center = signed_center.abs()
        background_residual = (frames - temporal_background).abs()

        previous = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
        following = torch.cat([frames[:, 1:], frames[:, -1:]], dim=1)
        adjacent_change = 0.5 * (
            (frames - previous).abs() + (following - frames).abs()
        )
        return torch.cat(
            [
                signed_center,
                absolute_center,
                background_residual,
                adjacent_change,
            ],
            dim=2,
        )

    def forward(
        self,
        frames: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
        center_index = frames.shape[1] // 2
        normalized = self._local_normalize(frames)
        cues = self._ordered_cues(normalized)
        pyramid = self.encoder(cues)

        s1 = self.s1_refine(self.s1_tdem(pyramid[0], center_index))
        s2 = self.s2_refine(self.s2_tdem(pyramid[1], center_index))
        s3 = self.s3_refine(self.s3_memory(pyramid[2], center_index))
        motion = [s1, s2, s3]
        confidence = [
            estimator(sequence, center_index)
            for estimator, sequence in zip(self.reliability, pyramid[:3])
        ]
        diagnostics = {
            "temporal_confidence": torch.stack(
                [item.mean() for item in confidence]
            ).mean(),
            "temporal_anomaly_energy": cues[:, :, 1:].abs().mean(),
            "temporal_center_residual": cues[:, center_index, 2:3].abs().mean(),
        }
        return motion, confidence, diagnostics


class TemporalReliabilityEstimator(nn.Module):
    """Estimates whether temporal evidence is persistent and symmetric."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.predict = nn.Sequential(
            ConvBNAct(3, hidden, kernel_size=3),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        center_index: int,
    ) -> torch.Tensor:
        energy = sequence.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        variance = sequence.float().var(dim=1, unbiased=False).mean(
            dim=1,
            keepdim=True,
        )
        pair_errors = []
        max_distance = min(
            center_index,
            sequence.shape[1] - center_index - 1,
        )
        for distance in range(1, max_distance + 1):
            previous = sequence[:, center_index - distance]
            following = sequence[:, center_index + distance]
            pair_errors.append(
                (previous - following).abs().mean(dim=1, keepdim=True)
            )
        if pair_errors:
            pair_error = torch.stack(pair_errors, dim=1).mean(dim=1)
        else:
            pair_error = torch.zeros_like(energy)
        symmetry = torch.exp(-pair_error / (energy.detach() + 1e-3))
        # Static infrared backgrounds often have exactly zero temporal variance.
        # sqrt(0) is finite in the forward pass but has an infinite derivative.
        stable_std = torch.sqrt(variance.clamp_min(0.0) + 1e-4)
        return self.predict(torch.cat([energy, stable_std, symmetry], dim=1))


class AppearanceRefinement(nn.Module):
    """Uses local contrast only at shallow appearance stages."""

    def __init__(self, channels: int, use_contrast: bool) -> None:
        super().__init__()
        self.local = (
            GatedContrastConv(channels, scale_init=0.06)
            if use_contrast
            else nn.Identity()
        )
        self.out = ResidualBlock(channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.out(self.local(feature))


class ReliableTemporalFusion(nn.Module):
    """Injects temporal evidence as a small confidence-gated residual."""

    def __init__(
        self,
        channels: int,
        scale_init: float,
        use_reliability: bool = True,
    ) -> None:
        super().__init__()
        self.use_reliability = use_reliability
        self.delta = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            ConvBNAct(channels * 2 + 1, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = ResidualBlock(channels)
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(
        self,
        appearance: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if temporal.shape[-2:] != appearance.shape[-2:]:
            temporal = F.interpolate(
                temporal,
                size=appearance.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        confidence = F.interpolate(
            confidence,
            size=appearance.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        if not self.use_reliability:
            confidence = torch.ones_like(confidence)
        difference = (appearance - temporal).abs()
        delta = self.delta(torch.cat([appearance, temporal, difference], dim=1))
        gate = self.gate(torch.cat([appearance, temporal, confidence], dim=1))
        fused = (
            appearance
            + torch.tanh(self.scale) * gate * confidence * delta
        )
        return self.out(fused)


class TemporalBoundaryMaskHead(nn.Module):
    """Refines the compact V6 mask with shallow temporal boundary evidence."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            MSFFBlock(channels * 2, channels),
            ResidualBlock(channels),
            ConvBNAct(channels, channels),
        )
        self.seg_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.center_head = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.objectness_head = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.boundary_head = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.temporal_correction = nn.Sequential(
            ConvBNAct(channels * 2 + 1, channels, kernel_size=1),
            ResidualBlock(channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.center_scale = nn.Parameter(torch.tensor(0.08))
        self.correction_scale = nn.Parameter(torch.tensor(0.02))

    def forward(
        self,
        x: torch.Tensor,
        low_level: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = x.shape[-2:]
        low_level = F.interpolate(
            low_level,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        temporal = F.interpolate(
            temporal,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        confidence = F.interpolate(
            confidence,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        feature = self.refine(torch.cat([x, low_level], dim=1))
        seg_logits = self.seg_head(feature)
        center_logits = self.center_head(feature)
        objectness_logits = self.objectness_head(feature)
        boundary_logits = self.boundary_head(feature)
        correction = self.temporal_correction(
            torch.cat([feature, temporal, confidence], dim=1)
        )
        logits = (
            seg_logits
            + torch.tanh(self.center_scale) * center_logits
            + torch.tanh(self.correction_scale) * correction
        )
        return {
            "logits": logits,
            "seg_logits": seg_logits,
            "center_logits": center_logits,
            "objectness_logits": objectness_logits,
            "boundary_logits": boundary_logits,
            "temporal_correction_logits": correction,
        }


class TFGNetV9(nn.Module):
    """V6 appearance backbone with ordered, flow-free temporal anomalies."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        use_temporal: bool = True,
        use_frequency_memory: bool = True,
        use_reliability: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.appearance_encoder = LVLowLevelEncoder(in_channels, base_channels)
        self.appearance_refine = nn.ModuleList(
            [
                AppearanceRefinement(channels[0], use_contrast=True),
                AppearanceRefinement(channels[1], use_contrast=True),
                AppearanceRefinement(channels[2], use_contrast=False),
                AppearanceRefinement(channels[3], use_contrast=False),
            ]
        )
        self.temporal_branch = OrderedTemporalAnomalyEncoder(
            base_channels,
            use_frequency_memory=use_frequency_memory,
        )
        self.fusions = nn.ModuleList(
            [
                ReliableTemporalFusion(
                    channels[0],
                    scale_init=0.02,
                    use_reliability=use_reliability,
                ),
                ReliableTemporalFusion(
                    channels[1],
                    scale_init=0.03,
                    use_reliability=use_reliability,
                ),
                ReliableTemporalFusion(
                    channels[2],
                    scale_init=0.03,
                    use_reliability=use_reliability,
                ),
            ]
        )
        self.up3 = PatchExpandSkipBlock(channels[3], channels[2], channels[2])
        self.up2 = PatchExpandSkipBlock(channels[2], channels[1], channels[1])
        self.up1 = PatchExpandSkipBlock(channels[1], channels[0], channels[0])
        self.head = TemporalBoundaryMaskHead(channels[0])

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del target_mask, valid_mask
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError("TFGNetV9 expects B x T x 1 x H x W input.")

        center = frames[:, frames.shape[1] // 2]
        appearance = [
            refine(feature)
            for refine, feature in zip(
                self.appearance_refine,
                self.appearance_encoder(center),
            )
        ]
        if self.use_temporal and frames.shape[1] > 1:
            temporal, confidence, diagnostics = self.temporal_branch(frames)
            fused = [
                fusion(appearance[index], temporal[index], confidence[index])
                for index, fusion in enumerate(self.fusions)
            ]
        else:
            temporal = [torch.zeros_like(feature) for feature in appearance[:3]]
            confidence = [
                torch.zeros(
                    feature.shape[0],
                    1,
                    feature.shape[2],
                    feature.shape[3],
                    device=feature.device,
                    dtype=feature.dtype,
                )
                for feature in appearance[:3]
            ]
            fused = appearance[:3]
            zero = center.new_zeros(())
            diagnostics = {
                "temporal_confidence": zero,
                "temporal_anomaly_energy": zero,
                "temporal_center_residual": zero,
            }

        x = self.up3(appearance[3], fused[2])
        x = self.up2(x, fused[1])
        x = self.up1(x, fused[0])
        outputs = self.head(
            x,
            appearance[0],
            temporal[0],
            confidence[0],
        )
        outputs.update(diagnostics)
        return outputs if return_aux else outputs["logits"]


# ===== Source: modelv10_1.py =====

class DualErrorCorrectionHead(nn.Module):
    """Learns separate foreground-recovery and false-positive suppression maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        input_channels = channels * 3 + 8
        hidden = max(channels * 2, 48)
        self.shared = nn.Sequential(
            ConvBNAct(input_channels, hidden, kernel_size=1),
            ResidualBlock(hidden),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            ConvBNAct(hidden, channels, kernel_size=1),
        )
        self.recover_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.suppress_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.recover_log_strength = nn.Parameter(torch.tensor(2.0))
        self.suppress_log_strength = nn.Parameter(torch.tensor(2.0))

        nn.init.zeros_(self.recover_head.weight)
        nn.init.constant_(self.recover_head.bias, -12.0)
        nn.init.zeros_(self.suppress_head.weight)
        nn.init.constant_(self.suppress_head.bias, -12.0)

    @staticmethod
    def _resize(
        feature: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        return F.interpolate(
            feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        decoded: torch.Tensor,
        shallow: torch.Tensor,
        temporal: torch.Tensor,
        confidence: torch.Tensor,
        center: torch.Tensor,
        temporal_residual: torch.Tensor,
        coarse_logits: torch.Tensor,
        center_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = coarse_logits.shape[-2:]
        decoded = self._resize(decoded, target_size)
        shallow = self._resize(shallow, target_size)
        temporal = self._resize(temporal, target_size)
        confidence = self._resize(confidence, target_size)
        center = self._resize(center, target_size)
        temporal_residual = self._resize(temporal_residual, target_size)
        center_logits = self._resize(center_logits, target_size)
        boundary_logits = self._resize(boundary_logits, target_size)

        local_mean = F.avg_pool2d(
            center,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        local_contrast = center - local_mean
        coarse_probability = torch.sigmoid(coarse_logits.float())
        center_probability = torch.sigmoid(center_logits.float())
        boundary_probability = torch.sigmoid(boundary_logits.float())
        feature = self.shared(
            torch.cat(
                [
                    decoded,
                    shallow,
                    temporal,
                    confidence,
                    center,
                    local_contrast,
                    temporal_residual,
                    coarse_probability,
                    center_probability,
                    boundary_probability,
                    torch.tanh(coarse_logits.float() / 6.0),
                ],
                dim=1,
            )
        )
        recover_logits = self.recover_head(feature)
        suppress_logits = self.suppress_head(feature)
        recover_strength = torch.sigmoid(self.recover_log_strength)
        suppress_strength = torch.sigmoid(self.suppress_log_strength)
        recover_probability = (
            recover_strength * torch.sigmoid(recover_logits.float())
        )
        suppress_probability = (
            suppress_strength
            * torch.sigmoid(suppress_logits.float())
        )
        recovered_probability = (
            coarse_probability
            + (1.0 - coarse_probability) * recover_probability
        )
        corrected_probability = (
            recovered_probability * (1.0 - suppress_probability)
        ).clamp(1e-7, 1.0 - 1e-7)
        corrected_logits = torch.logit(corrected_probability)
        recover_delta = (
            corrected_probability - coarse_probability
        ).clamp_min(0.0)
        suppress_delta = (
            recovered_probability - corrected_probability
        ).clamp_min(0.0)
        return {
            "logits": corrected_logits,
            "recover_logits": recover_logits,
            "suppress_logits": suppress_logits,
            "recover_delta": recover_delta,
            "suppress_delta": suppress_delta,
            "recover_strength": recover_strength,
            "suppress_strength": suppress_strength,
        }


class TFGNetV101(TFGNetV9):
    """V9 with bounded, bidirectional correction instead of mask regeneration."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        use_temporal: bool = True,
        use_frequency_memory: bool = True,
        use_reliability: bool = True,
        use_correction: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            use_temporal=use_temporal,
            use_frequency_memory=use_frequency_memory,
            use_reliability=use_reliability,
        )
        self.use_correction = use_correction
        self.correction = DualErrorCorrectionHead(base_channels)
        self.training_stage = "warmup"
        self.set_training_stage("warmup")

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"warmup", "refine", "finetune"}:
            raise ValueError(f"Unsupported V10.1 training stage: {stage}")
        self.training_stage = stage
        refine_prefixes = ("up3.", "up2.", "up1.", "head.", "correction.")
        for name, parameter in self.named_parameters():
            if stage == "warmup":
                trainable = name.startswith("correction.")
            elif stage == "refine":
                trainable = name.startswith(refine_prefixes)
            else:
                trainable = True
            parameter.requires_grad_(trainable)

    def train(self, mode: bool = True) -> TFGNetV101:
        super().train(mode)
        if mode and self.training_stage == "warmup":
            for name, module in self.named_children():
                if name != "correction":
                    module.eval()
            self.correction.train()
        elif mode and self.training_stage == "refine":
            self.appearance_encoder.eval()
            self.appearance_refine.eval()
            self.temporal_branch.eval()
            self.fusions.eval()
            self.up3.train()
            self.up2.train()
            self.up1.train()
            self.head.train()
            self.correction.train()
        return self

    def forward(
        self,
        frames: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del target_mask, valid_mask
        if frames.ndim == 4:
            frames = frames.unsqueeze(2)
        if frames.ndim != 5:
            raise ValueError("TFGNetV101 expects B x T x 1 x H x W input.")

        center_index = frames.shape[1] // 2
        center = frames[:, center_index]
        temporal_background = frames.median(dim=1).values
        temporal_residual = (center - temporal_background).abs()
        appearance = [
            refine(feature)
            for refine, feature in zip(
                self.appearance_refine,
                self.appearance_encoder(center),
            )
        ]
        if self.use_temporal and frames.shape[1] > 1:
            temporal, confidence, diagnostics = self.temporal_branch(frames)
            fused = [
                fusion(appearance[index], temporal[index], confidence[index])
                for index, fusion in enumerate(self.fusions)
            ]
        else:
            temporal = [torch.zeros_like(feature) for feature in appearance[:3]]
            confidence = [
                torch.zeros(
                    feature.shape[0],
                    1,
                    feature.shape[2],
                    feature.shape[3],
                    device=feature.device,
                    dtype=feature.dtype,
                )
                for feature in appearance[:3]
            ]
            fused = appearance[:3]
            zero = center.new_zeros(())
            diagnostics = {
                "temporal_confidence": zero,
                "temporal_anomaly_energy": zero,
                "temporal_center_residual": zero,
            }

        decoded = self.up3(appearance[3], fused[2])
        decoded = self.up2(decoded, fused[1])
        decoded = self.up1(decoded, fused[0])
        coarse = self.head(
            decoded,
            appearance[0],
            temporal[0],
            confidence[0],
        )
        if self.use_correction:
            correction = self.correction(
                decoded,
                appearance[0],
                temporal[0],
                confidence[0],
                center,
                temporal_residual,
                coarse["logits"],
                coarse["center_logits"],
                coarse["boundary_logits"],
            )
        else:
            zero_map = torch.zeros_like(coarse["logits"])
            correction = {
                "logits": coarse["logits"],
                "recover_logits": zero_map,
                "suppress_logits": zero_map,
                "recover_delta": zero_map,
                "suppress_delta": zero_map,
                "recover_strength": center.new_zeros(()),
                "suppress_strength": center.new_zeros(()),
            }
        outputs = {
            **coarse,
            **correction,
            "coarse_logits": coarse["logits"],
            "logits": correction["logits"],
            "mean_recover_delta": correction["recover_delta"].mean(),
            "mean_suppress_delta": correction["suppress_delta"].mean(),
            **diagnostics,
        }
        return outputs if return_aux else outputs["logits"]
