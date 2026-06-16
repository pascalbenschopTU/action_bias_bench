"""
Per-model feature extractors for the 6 downloaded foundation models.

Image models (CLIP, DINOv2, SigLIP, EVA-02, Hiera):
    load_<model>()  -> (model, processor)
    encode_<model>(frames_bgr, model, processor, device) -> np.ndarray (T, D)

    frames_bgr : (T, H, W, 3) uint8 BGR  — all frames encoded in one forward pass
    returns    : (T, D) float32 L2-normalised, one vector per frame

Video model (V-JEPA2):
    encode_vjepa2(clip_bgr, model, processor, device) -> np.ndarray (T//2, D)

    clip_bgr : (T, H, W, 3) uint8 BGR  — full clip, typically T=64
    returns  : (T//2, D) — one vector per 2-frame tubelet (temporal patch size = 2)
               e.g. 64 frames -> 32 temporal representations
               use for clip-level or coarse temporal analysis

All weights are loaded from the local HF snapshot; nothing is downloaded at runtime.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HF_ROOT = Path(__file__).resolve().parent / "huggingface"
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_rgb_list(frames_bgr: np.ndarray) -> list:
    """(T, H, W, 3) BGR uint8 -> list of T (H, W, 3) RGB uint8 arrays."""
    return [frames_bgr[t, :, :, ::-1].copy() for t in range(frames_bgr.shape[0])]


# ── CLIP ViT-L/14 ─────────────────────────────────────────────────────────────

def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    path = str(HF_ROOT / "openai__clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained(path)
    model = CLIPModel.from_pretrained(path).vision_model.eval().to(DEVICE)
    return model, processor


@torch.no_grad()
def encode_clip(frames_bgr: np.ndarray, model, processor, device=DEVICE) -> np.ndarray:
    """frames_bgr: (T, H, W, 3) -> (T, 768)"""
    inputs = processor(images=_to_rgb_list(frames_bgr), return_tensors="pt").to(device)
    emb = F.normalize(model(**inputs).pooler_output, dim=-1)   # (T, 768)
    return emb.cpu().float().numpy()


# ── DINOv2-L ──────────────────────────────────────────────────────────────────

def load_dinov2():
    from transformers import AutoModel, AutoProcessor
    path = str(HF_ROOT / "facebook__dinov2-large")
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(DEVICE)
    return model, processor


@torch.no_grad()
def encode_dinov2(frames_bgr: np.ndarray, model, processor, device=DEVICE) -> np.ndarray:
    """frames_bgr: (T, H, W, 3) -> (T, 1024)"""
    inputs = processor(images=_to_rgb_list(frames_bgr), return_tensors="pt").to(device)
    cls = model(**inputs).last_hidden_state[:, 0]              # CLS token, (T, 1024)
    return F.normalize(cls, dim=-1).cpu().float().numpy()


# ── SigLIP-so400m ─────────────────────────────────────────────────────────────

def load_siglip():
    from transformers import AutoModel, AutoImageProcessor
    path = str(HF_ROOT / "google__siglip-so400m-patch14-384")
    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModel.from_pretrained(path).vision_model.eval().to(DEVICE)
    return model, processor


@torch.no_grad()
def encode_siglip(frames_bgr: np.ndarray, model, processor, device=DEVICE) -> np.ndarray:
    """frames_bgr: (T, H, W, 3) -> (T, 1152)"""
    inputs = processor(images=_to_rgb_list(frames_bgr), return_tensors="pt").to(device)
    emb = F.normalize(model(**inputs).pooler_output, dim=-1)   # (T, 1152)
    return emb.cpu().float().numpy()


# ── DINOv3-L ──────────────────────────────────────────────────────────────────
# Gated model: accept terms at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

def load_dinov3():
    from transformers import AutoModel, AutoProcessor
    path = str(HF_ROOT / "facebook__dinov3-vitl16-pretrain-lvd1689m")
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(DEVICE)
    return model, processor


@torch.no_grad()
def encode_dinov3(frames_bgr: np.ndarray, model, processor, device=DEVICE) -> np.ndarray:
    """frames_bgr: (T, H, W, 3) -> (T, 1024)"""
    inputs = processor(images=_to_rgb_list(frames_bgr), return_tensors="pt").to(device)
    emb = F.normalize(model(**inputs).pooler_output, dim=-1)   # (T, 1024)
    return emb.cpu().float().numpy()


# ── EVA-02-L (timm) ───────────────────────────────────────────────────────────

def load_eva02():
    import timm
    from safetensors.torch import load_file
    path = HF_ROOT / "timm__eva02_large_patch14_448.mim_in22k_ft_in22k"
    model = timm.create_model("eva02_large_patch14_448.mim_in22k_ft_in22k",
                               pretrained=False, num_classes=0).eval().to(DEVICE)
    model.load_state_dict(load_file(path / "model.safetensors"), strict=False)
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    return model, transform


@torch.no_grad()
def encode_eva02(frames_bgr: np.ndarray, model, transform, device=DEVICE) -> np.ndarray:
    """frames_bgr: (T, H, W, 3) -> (T, 1024)"""
    from PIL import Image
    batch = torch.stack([
        transform(Image.fromarray(frames_bgr[t, :, :, ::-1]))
        for t in range(frames_bgr.shape[0])
    ]).to(device)                                               # (T, 3, 448, 448)
    emb = model(batch)                                          # avg pool over patches, (T, 1024)
    return F.normalize(emb, dim=-1).cpu().float().numpy()


# ── Hiera (image model) ───────────────────────────────────────────────────────
# Note: the HF transformers Hiera checkpoint is image-only (no temporal dimension).
# Encode frames independently in a batch, same as the other image models.

def load_hiera():
    from transformers import AutoModel, AutoProcessor
    path = str(HF_ROOT / "facebook__hiera-base-224-hf")
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(DEVICE)
    return model, processor


@torch.no_grad()
def encode_hiera(frames_bgr: np.ndarray, model, processor, device=DEVICE) -> np.ndarray:
    """frames_bgr: (T, H, W, 3) -> (T, 768)"""
    inputs = processor(images=_to_rgb_list(frames_bgr), return_tensors="pt").to(device)
    emb = model(**inputs).last_hidden_state.mean(dim=1)         # avg pool over patches, (T, 768)
    return F.normalize(emb, dim=-1).cpu().float().numpy()


# ── V-JEPA 2 (video model) ────────────────────────────────────────────────────
# Tubelet size = 2, so T frames -> T//2 temporal tokens after spatial pooling.
# e.g. 64 frames -> 32 temporal vectors, each summarising a 2-frame window.
# Use for clip-level or coarse temporal analysis; not suitable for per-frame SSM.

def load_vjepa2():
    from transformers import AutoModel, AutoVideoProcessor
    path = str(HF_ROOT / "facebook__vjepa2-vitl-fpc64-256")
    processor = AutoVideoProcessor.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(DEVICE)
    return model, processor


@torch.no_grad()
def encode_vjepa2(clip_bgr: np.ndarray, model, processor, device=DEVICE) -> np.ndarray:
    """
    clip_bgr: (T, H, W, 3) uint8 BGR.
    Returns (T//2, D) — one L2-normalised vector per 2-frame tubelet.
    For T=64: output shape is (32, 1024).
    """
    clip_rgb = clip_bgr[:, :, :, ::-1].copy()                  # BGR -> RGB
    inputs = processor(videos=list(clip_rgb), return_tensors="pt").to(device)
    tokens = model(**inputs).last_hidden_state                  # (1, N, D) — no CLS token
    # N = (T//2) * (H//patch) * (W//patch); reshape to pool over spatial dims
    T_half = clip_bgr.shape[0] // 2
    D = tokens.shape[-1]
    temporal = tokens.view(1, T_half, -1, D).mean(dim=2)       # (1, T//2, D)
    emb = F.normalize(temporal.squeeze(0), dim=-1)              # (T//2, D)
    return emb.cpu().float().numpy()
