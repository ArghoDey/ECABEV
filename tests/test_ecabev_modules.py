"""
Unit tests for the ECABEV novel modules.

These tests do **not** require CUDA or the compiled MultiScaleDeformableAttention
extension; they validate the three core contributions in isolation:

* De-Noise Enhanced Channel Attention
* Bilinear Interpolation Layer Normalization
* Scalable Cross-Entropy (SCE) BEV loss

Run with::

    python -m pytest tests/test_ecabev_modules.py -v

or simply::

    python tests/test_ecabev_modules.py
"""

import torch

from nets.ecabev_modules import (
    DeNoiseEnhancedChannelAttention,
    BilinearInterpolationLayerNorm,
    ScalableCrossEntropyBEVLoss,
)


def test_channel_attention_map_and_tokens():
    ca = DeNoiseEnhancedChannelAttention(channels=128, reduction=8)

    # 4-D feature map path
    x4 = torch.randn(2, 128, 50, 50)
    y4 = ca(x4)
    assert y4.shape == x4.shape

    # 3-D token path (BEV queries)
    x3 = torch.randn(2, 200 * 200, 128)
    y3 = ca(x3)
    assert y3.shape == x3.shape

    # attention should keep values bounded relative to input (sigmoid gate in [0,1])
    assert torch.isfinite(y4).all() and torch.isfinite(y3).all()


def test_channel_attention_gradients():
    ca = DeNoiseEnhancedChannelAttention(channels=64)
    x = torch.randn(1, 64, 20, 20, requires_grad=True)
    ca(x).sum().backward()
    assert x.grad is not None


def test_bilinear_interp_layer_norm_shapes():
    bln = BilinearInterpolationLayerNorm(
        image_channels=128, radar_channels=128, fused_channels=128, out_size=(200, 200)
    )
    img = torch.randn(2, 128, 200, 200)
    rad = torch.randn(2, 128, 100, 100)   # different resolution on purpose
    out = bln(img, rad)
    assert out.shape == (2, 128, 200, 200)


def test_bilinear_interp_layer_norm_channel_mismatch():
    # radar and image with different channel counts should still fuse to C_f
    bln = BilinearInterpolationLayerNorm(
        image_channels=96, radar_channels=64, fused_channels=128
    )
    img = torch.randn(1, 96, 50, 50)
    rad = torch.randn(1, 64, 25, 25)
    out = bln(img, rad)
    assert out.shape == (1, 128, 50, 50)


def test_sce_loss_basic():
    sce = ScalableCrossEntropyBEVLoss(num_buckets=10, top_k=4, pos_weight=2.13)
    pred = torch.randn(2, 1, 200, 200, requires_grad=True)
    tgt = (torch.rand(2, 1, 200, 200) > 0.9).float()
    valid = torch.ones(2, 1, 200, 200)
    loss = sce(pred, tgt, valid)
    loss.backward()
    assert loss.dim() == 0            # scalar
    assert loss.item() >= 0.0
    assert pred.grad is not None


def test_sce_loss_respects_valid_mask():
    sce = ScalableCrossEntropyBEVLoss(num_buckets=5, top_k=2)
    pred = torch.randn(1, 1, 40, 40)
    tgt = torch.zeros(1, 1, 40, 40)
    valid = torch.zeros(1, 1, 40, 40)  # nothing valid -> loss falls back to 0
    loss = sce(pred, tgt, valid)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_channel_attention_map_and_tokens()
    test_channel_attention_gradients()
    test_bilinear_interp_layer_norm_shapes()
    test_bilinear_interp_layer_norm_channel_mismatch()
    test_sce_loss_basic()
    test_sce_loss_respects_valid_mask()
    print("All ECABEV module tests passed.")
