"""
Model definitions for the project.

This file implements a very small, beginner-friendly ConvNeXt-like CNN
completely from scratch (no torchvision/timm). It follows the spirit of
ConvNeXt blocks (depthwise conv + pointwise MLP + residual) but keeps the
design tiny and easy to read for students.

Key ideas used here (kept simple):
- Depthwise 7x7 convolution for local mixing
- LayerNorm applied in channels-last style (wrapped for convenience)
- Pointwise 1x1 "MLP" (expand -> GELU -> project)
- Residual connection
- Three stages with downsampling between stages

This is intentionally minimal: few lines, lots of comments, and small parameter
counts so it can train on modest GPUs or CPU for quick tests.
"""

# ========= IMPORTS =========
from typing import List

import torch
import torch.nn as nn


# ========= SMALL BUILDING BLOCKS =========
class LayerNorm2d(nn.Module):
    """LayerNorm wrapper that works on NCHW tensors by switching to NHWC.

    Args:
      num_channels: number of channels C to normalize over
      eps: small epsilon for numerical stability
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [N, C, H, W]; LayerNorm expects last-dim features, so switch
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # [N, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # back to [N, C, H, W]
        return x


class ConvNeXtLikeBlock(nn.Module):
    """A tiny ConvNeXt-style block with depthwise conv and pointwise MLP.

    Structure:
      x -> DWConv(7x7, groups=C) -> LayerNorm -> 1x1 Conv (expand 4C)
        -> GELU -> 1x1 Conv (project C) -> gamma scale -> + residual
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Depthwise convolution mixes within each channel spatially.
        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        # LayerNorm improves stability; we use a simple wrapper for NCHW.
        self.norm = LayerNorm2d(channels)
        # Pointwise "MLP": expand -> activation -> project
        self.pointwise_expand = nn.Conv2d(channels, 4 * channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_project = nn.Conv2d(4 * channels, channels, kernel_size=1)
        # Learnable per-channel scale (initialized to 1)
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.pointwise_expand(x)
        x = self.activation(x)
        x = self.pointwise_project(x)
        # Apply per-channel scale with broadcasting
        x = x * self.gamma.view(1, -1, 1, 1)
        return residual + x


# ========= THE TINY NETWORK =========
class ConvNeXtLikeTiny(nn.Module):
    """A very small ConvNeXt-inspired network for 2D images.

    Defaults:
      - Three stages with channel dims [64, 128, 256]
      - Two blocks per stage (depth=2) to keep it very small
      - Stem: 4x4 stride-4 conv to reduce resolution early
      - Downsampling between stages via 2x2 stride-2 conv
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dims: List[int] = None,
        depths: List[int] = None,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if dims is None:
            dims = [64, 128, 256]
        if depths is None:
            depths = [2, 2, 2]
        classifier_dropout = float(max(0.0, min(classifier_dropout, 1.0)))

        # Early stem to shrink spatial size and increase channels a bit.
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)

        # Build stages: each is a sequence of ConvNeXt-like blocks.
        stages: List[nn.Module] = []
        for stage_index, (channels, depth) in enumerate(zip(dims, depths)):
            blocks = [ConvNeXtLikeBlock(channels) for _ in range(depth)]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        # Downsamplers between stages (except after the last one)
        downs: List[nn.Module] = []
        for i in range(len(dims) - 1):
            downs.append(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
        self.downsamples = nn.ModuleList(downs)

        # Final classifier head: global average pool -> linear
        self.head_norm = LayerNorm2d(dims[-1])
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(dims[-1], num_classes)

        # Initialize the linear classifier weights to small values.
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W]
        x = self.stem(x)
        # Stage 0
        x = self.stages[0](x)
        # Stage 1
        x = self.downsamples[0](x)
        x = self.stages[1](x)
        # Stage 2
        x = self.downsamples[1](x)
        x = self.stages[2](x)

        # Global average pooling over spatial dims (H, W)
        x = self.head_norm(x)
        x = x.mean(dim=(2, 3))  # [N, C]
        x = self.dropout(x)
        x = self.classifier(x)  # [N, num_classes]
        return x


# ========= PUBLIC FACTORY FUNCTION =========
def build_convnext_tiny(num_classes: int = 2, classifier_dropout: float = 0.3) -> nn.Module:
    """Create the minimal ConvNeXt-like Tiny model (from scratch).

    Args:
      num_classes: number of output classes for classification (2 for AD/CN)

    Returns:
      A torch.nn.Module that predicts logits of shape [batch_size, num_classes].
    """
    return ConvNeXtLikeTiny(in_channels=3, num_classes=num_classes, classifier_dropout=classifier_dropout)
