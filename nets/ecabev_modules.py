"""
ecabev_modules.py
==================

Core novel building blocks of **ECABEV** (Enhanced Channel Attention BEV).

This file implements the three main contributions described in the paper
"Enhanced BEV Scene Segmentation: De-Noise Channel Attention for
Resource-Constrained Environments":

1. :class:`DeNoiseEnhancedChannelAttention`
   A channel-attention module that combines *global average pooling* (GAP) and
   *global max pooling* (GMP) through a shared MLP followed by a sigmoid gate.
   GAP preserves global context while GMP preserves finer spatial peaks, so the
   module suppresses irrelevant camera noise while retaining discriminative
   features (paper, Sec. 3.4.1, Fig. 4).

2. :class:`BilinearInterpolationLayerNorm`
   Layer normalization with learned bilinear interpolation (paper, Sec. 3.5,
   Algorithm 1). Radar and image feature maps are resized to a common spatial
   resolution, channel-aligned with lightweight convolutions, layer-normalized,
   concatenated, and refined with depth-wise separable convolutions. This keeps
   spatial fidelity while reducing computational overhead relative to naive
   normalization.

3. :class:`ScalableCrossEntropyBEVLoss`
   A bucket-based scalable cross-entropy (SCE) loss for BEV object segmentation
   (paper, Sec. 3.6, Eqs. (2)-(3)). It partitions spatial locations into buckets
   and computes the loss only over the most impactful (hard-negative)
   representatives per bucket, which handles severe class imbalance on
   nuScenes with lower computational cost than dense cross-entropy.

All modules are self-contained and depend only on PyTorch, so they can be
imported and unit-tested independently of the rest of the pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DeNoiseEnhancedChannelAttention",
    "DepthwiseSeparableConv2d",
    "BilinearInterpolationLayerNorm",
    "ScalableCrossEntropyBEVLoss",
]


# --------------------------------------------------------------------------- #
#  Contribution 1:  De-Noise Enhanced Channel Attention                        #
# --------------------------------------------------------------------------- #
class DeNoiseEnhancedChannelAttention(nn.Module):
    r"""De-Noise Enhanced Channel Attention (paper Sec. 3.4.1, Fig. 4).

    The input feature map is squeezed with **two parallel** pooling operations:

    * Global Average Pooling (GAP) -> global context
    * Global Max Pooling  (GMP)    -> finer spatial peaks

    Both descriptors pass through a **shared** MLP (bottleneck reduction), their
    outputs are summed and passed through a sigmoid to produce channel-wise
    attention weights that re-calibrate the input, amplifying informative
    channels and suppressing noisy ones.

    The module accepts either

    * a 4-D tensor ``(B, C, H, W)`` (image/BEV feature map), or
    * a 3-D token tensor ``(B, N, C)`` (transformer BEV queries) together with
      the spatial shape ``(H, W)`` so pooling is well defined.

    Args:
        channels: number of feature channels ``C``.
        reduction: bottleneck reduction ratio for the shared MLP.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        # Shared MLP applied identically to the GAP and GMP descriptors.
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    # ---- 4-D path: (B, C, H, W) ------------------------------------------- #
    def _forward_map(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)   # GAP -> (B, C)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)    # GMP -> (B, C)
        attn = self.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * attn.view(b, c, 1, 1)

    # ---- 3-D path: (B, N, C) ---------------------------------------------- #
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # Pool over the token dimension N (== H*W flattened BEV grid).
        avg = x.mean(dim=1)                            # GAP -> (B, C)
        mx = x.max(dim=1).values                       # GMP -> (B, C)
        attn = self.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return x * attn.unsqueeze(1)                    # broadcast over N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: ``(B, C, H, W)`` feature map **or** ``(B, N, C)`` token tensor.

        Returns:
            Re-calibrated tensor with the same shape as ``x``.
        """
        if x.dim() == 4:
            return self._forward_map(x)
        elif x.dim() == 3:
            return self._forward_tokens(x)
        raise ValueError(
            f"DeNoiseEnhancedChannelAttention expects a 3-D or 4-D tensor, "
            f"got shape {tuple(x.shape)}"
        )


# --------------------------------------------------------------------------- #
#  Contribution 2:  Bilinear Interpolation Layer Normalization                 #
# --------------------------------------------------------------------------- #
class DepthwiseSeparableConv2d(nn.Module):
    """Lightweight depth-wise separable convolution used for feature enhancement.

    A depth-wise ``3x3`` convolution followed by a point-wise ``1x1`` convolution.
    This adds little computational cost while enriching the fused representation
    (paper Sec. 3.5).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class BilinearInterpolationLayerNorm(nn.Module):
    r"""Bilinear Interpolation Layer Normalization (paper Sec. 3.5, Algorithm 1).

    Steps (mirroring Algorithm 1 in the manuscript):

    1. Bilinearly interpolate the radar feature map to the desired output size
       ``(out_h, out_w)``.
    2. Align channels of the resized radar map and the image map with two
       lightweight ``1x1`` convolutions so both have ``fused_channels``.
    3. Layer-normalize both aligned feature maps.
    4. Concatenate along the channel dimension.
    5. Refine the concatenated map with a depth-wise separable feature-enhancement
       network and a final ``1x1`` convolution, producing a unified,
       scale-consistent feature map of ``fused_channels`` channels.

    Args:
        image_channels:  channels ``C_i`` of the image feature map.
        radar_channels:  channels ``C_r`` of the radar feature map.
        fused_channels:  desired number of channels ``C_f`` of the fused output.
        out_size: optional ``(H_o, W_o)``. If ``None``, the image map's spatial
            size is used at run time.
    """

    def __init__(
        self,
        image_channels: int,
        radar_channels: int,
        fused_channels: int,
        out_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.out_size = out_size
        self.fused_channels = fused_channels

        # Step 2: channel-alignment convolutions.
        self.align_radar = nn.Conv2d(radar_channels, fused_channels, kernel_size=1, bias=False)
        self.align_image = nn.Conv2d(image_channels, fused_channels, kernel_size=1, bias=False)

        # Step 3: layer norm over the channel dim (implemented via GroupNorm with
        # a single group == LayerNorm across channels, works on (B, C, H, W)).
        self.norm_radar = nn.GroupNorm(1, fused_channels)
        self.norm_image = nn.GroupNorm(1, fused_channels)

        # Step 5: feature enhancement + fuse concatenated (2*C_f) -> C_f.
        self.feature_enhancement = nn.Sequential(
            DepthwiseSeparableConv2d(2 * fused_channels, 2 * fused_channels),
            nn.Conv2d(2 * fused_channels, fused_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(fused_channels),
            nn.GELU(),
        )

    def forward(self, image_feat: torch.Tensor, radar_feat: torch.Tensor) -> torch.Tensor:
        """Fuse and normalize image and radar feature maps.

        Args:
            image_feat: ``(B, C_i, H_i, W_i)`` image BEV feature map.
            radar_feat: ``(B, C_r, H_r, W_r)`` radar BEV feature map.

        Returns:
            ``(B, C_f, H_o, W_o)`` fused, scale-consistent feature map.
        """
        out_size = self.out_size or image_feat.shape[-2:]

        # 1. resize radar features to the target resolution.
        radar_resized = F.interpolate(
            radar_feat, size=out_size, mode="bilinear", align_corners=False
        )
        # also make sure the image map matches the target resolution.
        if image_feat.shape[-2:] != tuple(out_size):
            image_feat = F.interpolate(
                image_feat, size=out_size, mode="bilinear", align_corners=False
            )

        # 2. channel alignment.
        radar_aligned = self.align_radar(radar_resized)
        image_aligned = self.align_image(image_feat)

        # 3. layer normalization.
        radar_norm = self.norm_radar(radar_aligned)
        image_norm = self.norm_image(image_aligned)

        # 4. concatenate.
        fused = torch.cat([radar_norm, image_norm], dim=1)

        # 5. feature enhancement.
        return self.feature_enhancement(fused)


# --------------------------------------------------------------------------- #
#  Contribution 3:  Scalable Cross-Entropy (SCE) BEV Loss                      #
# --------------------------------------------------------------------------- #
class ScalableCrossEntropyBEVLoss(nn.Module):
    r"""Scalable Cross-Entropy BEV loss (paper Sec. 3.6, Eqs. (2)-(3)).

    Dense binary cross-entropy over every BEV cell is expensive and dominated by
    the many easy background cells. The SCE BEV loss instead:

    * partitions the BEV grid into ``num_buckets`` spatial buckets;
    * for each bucket, keeps only the **hardest** representative locations
      (the ``max`` operator in Eq. (2)), i.e. positives that are poorly scored
      and hard negatives with the highest logits;
    * averages the per-representative binary cross-entropy over non-empty
      buckets.

    This yields balanced treatment of rare classes ("pedestrian crossing") vs.
    frequent classes ("drivable area") while computing far fewer logits.

    The loss operates on a single object channel (``vehicles``) exactly like the
    original object head, so it is a drop-in replacement for the BCE object loss.

    Args:
        num_buckets: number of spatial buckets ``|B|`` the grid is split into
            (per side); the grid is tiled into ``num_buckets x num_buckets``
            regions.
        top_k: number of hardest representatives kept per bucket.
        pos_weight: weight applied to positive samples in the BCE term to further
            counter class imbalance.
    """

    def __init__(self, num_buckets: int = 10, top_k: int = 4, pos_weight: float = 2.13):
        super().__init__()
        self.num_buckets = num_buckets
        self.top_k = top_k
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the SCE BEV loss.

        Args:
            pred_logits: ``(B, 1, H, W)`` raw object logits.
            target: ``(B, 1, H, W)`` binary ground truth.
            valid: optional ``(B, 1, H, W)`` mask of valid BEV cells.

        Returns:
            Scalar loss tensor.
        """
        b, c, h, w = pred_logits.shape
        assert c == 1, "SCE BEV loss expects a single object channel"

        if valid is None:
            valid = torch.ones_like(target)

        # Per-cell BCE without reduction, with positive weighting.
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction="none", pos_weight=self.pos_weight
        )
        bce = bce * valid  # zero-out invalid cells

        # Tile the grid into num_buckets x num_buckets buckets using average
        # pooling over the "hardness" (bce) values, then select the hardest
        # top_k representatives inside each bucket.
        nb = self.num_buckets
        # bucket size (ceil division so the whole grid is covered)
        bh = (h + nb - 1) // nb
        bw = (w + nb - 1) // nb

        # Pad so h, w are divisible by the bucket size.
        pad_h = bh * nb - h
        pad_w = bw * nb - w
        bce_p = F.pad(bce, (0, pad_w, 0, pad_h))
        valid_p = F.pad(valid, (0, pad_w, 0, pad_h))

        # Reshape into (B, nb*nb, bh*bw) so each bucket is one row.
        bce_buckets = (
            bce_p.view(b, 1, nb, bh, nb, bw)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, nb * nb, bh * bw)
        )
        valid_buckets = (
            valid_p.view(b, 1, nb, bh, nb, bw)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, nb * nb, bh * bw)
        )

        # Hardest top_k representatives per bucket (Eq. (2) max operator,
        # generalized to top_k for stability).
        k = min(self.top_k, bce_buckets.shape[-1])
        topk_vals, _ = bce_buckets.topk(k, dim=-1)          # (B, nb*nb, k)

        # A bucket is "non-empty" (L_k != empty) if it contains any valid cell.
        bucket_has_valid = valid_buckets.sum(dim=-1) > 0     # (B, nb*nb)

        # Mean over the kept representatives, then mean over non-empty buckets.
        per_bucket = topk_vals.mean(dim=-1)                  # (B, nb*nb)
        per_bucket = per_bucket * bucket_has_valid
        denom = bucket_has_valid.sum().clamp(min=1.0)
        loss = per_bucket.sum() / denom
        return loss
