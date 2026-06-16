#!/usr/bin/env python
"""Download foundation-model weights from the HuggingFace Hub into a local folder.

Each model is fetched into its own subfolder under DEST (default:
models/huggingface/), as plain files (no symlink cache). All incidental HF /
torch cache is routed to the project-local .cache/ so nothing lands in $HOME.

Usage:
    python scripts/download_foundation_models.py                 # core models
    python scripts/download_foundation_models.py --list          # show models, no download
    python scripts/download_foundation_models.py --only clip-vit-l dinov2-l
    python scripts/download_foundation_models.py --include-gated # also ViCLIP / InternVideo2
    python scripts/download_foundation_models.py --dest /some/other/dir

Gated repos (ViCLIP, InternVideo2) require accepting their terms on the hub and
a token: pass --token <hf_token> or set HF_TOKEN in the environment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# ── Resolve project paths and route ALL caches to project-local .cache ──────────
# (must happen before importing huggingface_hub so it reads the right defaults)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent                      # .../ActionBiasBench
PROJECT_CACHE = ROOT_DIR / ".cache"
DEFAULT_DEST = ROOT_DIR / "models" / "huggingface"

os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_CACHE / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))

from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError  # noqa: E402

# ── Model registry ──────────────────────────────────────────────────────────────
# name -> (repo_id, kind, gated)
CORE_MODELS: dict[str, tuple[str, str]] = {
    "clip-vit-l":     ("openai/clip-vit-large-patch14",                   "image"),
    "dinov2-l":       ("facebook/dinov2-large",                           "image"),
    "siglip-so400m":  ("google/siglip-so400m-patch14-384",               "image"),
    "eva02-l":        ("timm/eva02_large_patch14_448.mim_in22k_ft_in22k", "image"),
    "vjepa2-l":       ("facebook/vjepa2-vitl-fpc64-256",                  "video"),
    "hiera-b":        ("facebook/hiera-base-224-hf",                      "video"),
}

GATED_MODELS: dict[str, tuple[str, str]] = {
    "dinov3-l":       ("facebook/dinov3-vitl16-pretrain-lvd1689m", "image"),
    "viclip":         ("OpenGVLab/ViCLIP",                         "video"),
    "internvideo2-1b":("OpenGVLab/InternVideo2-CLIP-1B-224p-f8",   "video"),
}

# Skip framework duplicates we never use (PyTorch-only), to save disk.
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ckpt", "*tf_model*", "*flax*", "*.onnx", "*.tflite"]


def folder_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}"
        f /= 1024


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"download root (default: {DEFAULT_DEST})")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="download only these model names")
    ap.add_argument("--include-gated", action="store_true", help="also fetch gated models (ViCLIP, InternVideo2)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token for gated repos (or set HF_TOKEN)")
    ap.add_argument("--list", action="store_true", help="list models and exit (no download)")
    args = ap.parse_args()

    registry = dict(CORE_MODELS)
    if args.include_gated or (args.only and any(n in GATED_MODELS for n in args.only)):
        registry.update(GATED_MODELS)

    if args.only:
        unknown = [n for n in args.only if n not in registry]
        if unknown:
            ap.error(f"unknown model name(s): {unknown}. Available: {sorted(registry)}")
        selected = {n: registry[n] for n in args.only}
    else:
        selected = dict(CORE_MODELS)

    if args.list:
        print(f"dest: {args.dest}")
        print(f"cache (HF_HOME): {os.environ['HF_HOME']}")
        print("\nselected models:")
        for name, (repo, kind) in selected.items():
            gated = " [GATED]" if name in GATED_MODELS else ""
            print(f"  {name:16s} {kind:6s} {repo}{gated}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(selected)} model(s) -> {args.dest}")
    print(f"Cache routed to: {os.environ['HF_HOME']}\n")

    ok, failed = [], []
    for name, (repo, kind) in selected.items():
        target = args.dest / folder_name(repo)
        print(f"── {name} ({kind}) :: {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(target),
                ignore_patterns=IGNORE_PATTERNS,
                token=args.token,
            )
            print(f"   done -> {target}  ({human(dir_size(target))})\n")
            ok.append(name)
        except GatedRepoError:
            print(f"   GATED: accept terms at https://huggingface.co/{repo} and pass --token\n")
            failed.append(name)
        except RepositoryNotFoundError:
            print(f"   NOT FOUND (or needs auth): {repo}\n")
            failed.append(name)
        except Exception as e:  # keep going on any single-model failure
            print(f"   FAILED: {type(e).__name__}: {e}\n")
            failed.append(name)

    print("=" * 60)
    print(f"ok:     {ok}")
    if failed:
        print(f"failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
