"""
Feature extractors for Kinetics-400 pretrained torchvision video models.

Follows the same API as huggingface_models.py:
    load_{model}()  -> (model, None)          # processor is None (not needed)
    encode_{model}(frames_bgr, model, None, device) -> np.ndarray (1, D)

Unlike the image models in huggingface_models.py which return (T, D) — one vector per
frame — these video models return (1, D): a single clip-level embedding produced by the
backbone's spatial-temporal pooling before the classifier head.

Feature extraction: a forward hook on model._classifier_module captures the INPUT to
that layer (the penultimate feature vector). The classifier output is discarded.
This works for all head types (Linear, Conv3d) because _replace_classifier_head in
train_torchvision_rgb_probe.py always sets _classifier_module.

Supported models (same set as in train_torchvision_rgb_probe.py):
    r3d_18, mc3_18, r2plus1d_18, mvit_v2_s, s3d, swin3d_s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_torchvision_rgb_probe import build_model, normalize_rgb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s", "s3d", "swin3d_s"]


# Transformer-based models require 224×224 input due to fixed positional encodings.
# CNN models (r3d_18, mc3_18, r2plus1d_18, s3d) use global average pooling and
# can process any spatial resolution without issue.
_REQUIRES_224 = {"mvit_v2_s", "swin3d_s"}


@torch.no_grad()
def _encode_torchvision_video(
    frames_bgr: np.ndarray,
    model: nn.Module,
    model_name: str,
    device: torch.device,
) -> np.ndarray:
    """
    frames_bgr: (T, H, W, 3) uint8 BGR  — any spatial resolution
    Returns:    (1, D) float32 L2-normalised clip embedding
    """
    # BGR → RGB, scale to [0, 1]
    frames_rgb = frames_bgr[:, :, :, ::-1].copy()
    x = torch.from_numpy(frames_rgb).float() / 255.0          # (T, H, W, 3)
    x = x.permute(0, 3, 1, 2)                                 # (T, 3, H, W)

    # Transformer models have fixed positional encodings for 224×224 patches —
    # they crash at any other spatial resolution.
    if model_name in _REQUIRES_224 and (x.shape[2] != 224 or x.shape[3] != 224):
        import torchvision.transforms.functional as TF
        x = TF.resize(x, [224, 224], antialias=True)

    x = x.permute(1, 0, 2, 3).unsqueeze(0).to(device)        # (1, 3, T, H, W)
    x = normalize_rgb(x, model_name)

    captured: list[torch.Tensor] = []

    def _hook(module: nn.Module, inp, out) -> None:
        feat = inp[0] if isinstance(inp, tuple) else inp
        # flatten all dims after batch: handles Linear (B, D) and Conv3d (B, C, t, h, w)
        captured.append(feat.detach().view(feat.shape[0], -1))

    handle = model._classifier_module.register_forward_hook(_hook)
    try:
        model(x)
    finally:
        handle.remove()

    feat = captured[0]                                         # (1, D)
    feat = F.normalize(feat.float(), dim=-1)
    return feat.cpu().numpy()                                  # (1, D)


def _make_load(model_name: str):
    def _load():
        m = build_model(model_name, num_classes=400, pretrained=True).to(DEVICE).eval()
        return m, None
    _load.__name__ = f"load_{model_name}"
    return _load


def _make_encode(model_name: str):
    def _encode(frames_bgr: np.ndarray, model: nn.Module, _processor, device=DEVICE):
        return _encode_torchvision_video(frames_bgr, model, model_name, device)
    _encode.__name__ = f"encode_{model_name}"
    return _encode


# Expose load_{model} and encode_{model} for every supported model
for _name in _MODELS:
    globals()[f"load_{_name}"]   = _make_load(_name)
    globals()[f"encode_{_name}"] = _make_encode(_name)
