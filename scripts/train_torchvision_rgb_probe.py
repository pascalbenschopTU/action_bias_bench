from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import build_warmup_cosine_scheduler


VIDEO_MODEL_SPECS: List[Tuple[str, str, str]] = [
    ("r3d_18", "r3d_18", "R3D_18_Weights"),
    ("mc3_18", "mc3_18", "MC3_18_Weights"),
    ("r2plus1d_18", "r2plus1d_18", "R2Plus1D_18_Weights"),
    ("mvit_v1_b", "mvit_v1_b", "MViT_V1_B_Weights"),
    ("mvit_v2_s", "mvit_v2_s", "MViT_V2_S_Weights"),
    ("s3d", "s3d", "S3D_Weights"),
    ("swin3d_t", "swin3d_t", "Swin3D_T_Weights"),
    ("swin3d_s", "swin3d_s", "Swin3D_S_Weights"),
    ("swin3d_b", "swin3d_b", "Swin3D_B_Weights"),
]


def mode_name_for_model(model_name: str) -> str:
    return f"rgb_{str(model_name).lower()}_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a Kinetics-pretrained torchvision RGB-only action model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_models = subparsers.add_parser("list_models")
    list_models.add_argument("--json", action="store_true")

    train = subparsers.add_parser("train")
    add_common_dataset_args(train)
    train.add_argument("--out_dir", type=str, required=True)
    train.add_argument("--model", type=str, default="r3d_18")
    train.add_argument("--pretrained", action="store_true", default=True)
    train.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    train.add_argument("--batch_size", type=int, default=4)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--warmup_steps", type=int, default=0)
    train.add_argument("--min_lr", type=float, default=0.0)
    train.add_argument("--label_smoothing", type=float, default=0.0)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log_every", type=int, default=10)
    train.add_argument("--checkpoint_name", type=str, default="checkpoint_latest.pt")
    train.add_argument("--checkpoint_mode", type=str, default="latest", choices=["latest", "best", "final"])
    train.add_argument("--val_every", type=int, default=0)
    train.add_argument("--amp", action="store_true", default=True)
    train.add_argument("--no_amp", dest="amp", action="store_false")
    train.add_argument("--color_jitter", type=float, default=0.0,
                        help="Probability of applying ColorJitter to RGB frames during training.")
    train.add_argument("--p_hflip", type=float, default=0.0,
                        help="Probability of applying horizontal flip during RGB training.")
    train.add_argument("--mixup_prob", type=float, default=0.0)
    train.add_argument("--mixup_alpha", type=float, default=0.2)
    train.add_argument("--freeze_backbone", action="store_true", default=False)
    train.add_argument("--freeze_bn_stats", action="store_true", default=False)
    train.add_argument("--val_root_dir", type=str, default="")
    train.add_argument("--val_manifest", type=str, default="")
    train.add_argument("--val_class_id_to_label_csv", type=str, default="")
    train.add_argument("--resume_ckpt", type=str, default="")

    ev = subparsers.add_parser("eval")
    add_common_dataset_args(ev)
    ev.add_argument("--ckpt", type=str, required=True)
    ev.add_argument("--out_dir", type=str, required=True)
    ev.add_argument("--split_name", type=str, default="eval")
    ev.add_argument("--model", type=str, default="r3d_18")
    ev.add_argument("--batch_size", type=int, default=4)
    ev.add_argument("--num_workers", type=int, default=4)
    ev.add_argument("--device", type=str, default="cuda")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--summary_only", action="store_true")

    ag = subparsers.add_parser("aggregate")
    ag.add_argument("--out_dir", type=str, required=True)
    ag.add_argument("--model", type=str, required=True)

    return parser.parse_args()


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--class_id_to_label_csv", type=str, required=True)
    parser.add_argument("--rgb_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--rgb_sampling", type=str, default="uniform", choices=["uniform", "center", "random"])


def load_rgb_dataset_api():
    from dataset import RGBVideoClipDataset, collate_rgb_clip

    return RGBVideoClipDataset, collate_rgb_clip


def load_augment_api():
    from augment import mixup_batch, soft_target_cross_entropy

    return mixup_batch, soft_target_cross_entropy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _default_mean_std_for_model(model_name: str) -> Tuple[List[float], List[float]]:
    model_name = str(model_name).lower()
    if model_name.startswith(("mvit_", "swin3d_")):
        return [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
    return [0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]


def _extract_weights_mean_std(weights: Any, model_name: str) -> Tuple[List[float], List[float]]:
    default_mean, default_std = _default_mean_std_for_model(model_name)
    if weights is None:
        return default_mean, default_std
    try:
        transforms = weights.transforms()
    except Exception:
        return default_mean, default_std
    mean = getattr(transforms, "mean", None)
    std = getattr(transforms, "std", None)
    if mean is None or std is None:
        return default_mean, default_std
    return [float(x) for x in mean], [float(x) for x in std]


def get_torchvision_video_registry() -> Dict[str, Dict[str, Any]]:
    try:
        import torchvision.models.video as tv_video
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required for the rgb_k400 baseline. Install torchvision in the active environment."
        ) from exc

    registry: Dict[str, Dict[str, Any]] = {}
    for model_name, builder_name, enum_name in VIDEO_MODEL_SPECS:
        builder = getattr(tv_video, builder_name, None)
        if not callable(builder):
            continue
        weight_enum = getattr(tv_video, enum_name, None)
        default_weights = None
        if weight_enum is not None:
            default_weights = getattr(weight_enum, "DEFAULT", getattr(weight_enum, "KINETICS400_V1", None))
        mean, std = _extract_weights_mean_std(default_weights, model_name)
        registry[model_name] = {
            "builder": builder,
            "default_weights": default_weights,
            "mean": mean,
            "std": std,
        }

    # Best-effort discovery for future torchvision versions that add
    # video builders with a matching weights enum we do not explicitly know yet.
    for attr_name in dir(tv_video):
        if attr_name.startswith("_") or attr_name in registry:
            continue
        candidate = getattr(tv_video, attr_name)
        if not callable(candidate) or not inspect.isfunction(candidate):
            continue
        enum_prefix = attr_name.replace("_", "").upper()
        matching_enums = [
            enum_name
            for enum_name in dir(tv_video)
            if enum_name.endswith("_Weights") and enum_name.replace("_", "").upper().startswith(enum_prefix)
        ]
        if not matching_enums:
            continue
        weight_enum = getattr(tv_video, matching_enums[0], None)
        default_weights = getattr(weight_enum, "DEFAULT", getattr(weight_enum, "KINETICS400_V1", None))
        mean, std = _extract_weights_mean_std(default_weights, attr_name)
        registry[attr_name] = {
            "builder": candidate,
            "default_weights": default_weights,
            "mean": mean,
            "std": std,
        }

    return dict(sorted(registry.items()))


def list_available_models() -> List[str]:
    return sorted(get_torchvision_video_registry())


def _replace_linear_layer(layer: nn.Linear, num_classes: int) -> nn.Linear:
    return nn.Linear(int(layer.in_features), int(num_classes))


def _replace_conv3d_layer(layer: nn.Conv3d, num_classes: int) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels=int(layer.in_channels),
        out_channels=int(num_classes),
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=int(layer.groups),
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
    )


def _replace_classifier_layer(layer: nn.Module, num_classes: int) -> nn.Module:
    if isinstance(layer, nn.Linear):
        return _replace_linear_layer(layer, num_classes)
    if isinstance(layer, nn.Conv3d):
        return _replace_conv3d_layer(layer, num_classes)
    raise RuntimeError(
        f"Expected classifier projection to be Linear or Conv3d, got: {type(layer).__name__}"
    )


def _set_nested_attr(module: nn.Module, path: str, new_layer: nn.Module) -> None:
    parts = path.split(".")
    target = module
    for part in parts[:-1]:
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part)
    last = parts[-1]
    if last.isdigit():
        target[int(last)] = new_layer
    else:
        setattr(target, last, new_layer)


def _replace_classifier_head(module: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(module, "fc") and isinstance(module.fc, nn.Linear):
        module.fc = _replace_linear_layer(module.fc, num_classes)
        module._classifier_module = module.fc
        return module

    if hasattr(module, "head") and isinstance(module.head, (nn.Linear, nn.Conv3d)):
        module.head = _replace_classifier_layer(module.head, num_classes)
        module._classifier_module = module.head
        return module

    if hasattr(module, "head") and isinstance(module.head, nn.Sequential) and len(module.head) > 0:
        final_layer = module.head[-1]
        if isinstance(final_layer, (nn.Linear, nn.Conv3d)):
            module.head[-1] = _replace_classifier_layer(final_layer, num_classes)
            module._classifier_module = module.head[-1]
            return module

    if hasattr(module, "classifier"):
        classifier = module.classifier
        if isinstance(classifier, nn.Linear):
            module.classifier = _replace_linear_layer(classifier, num_classes)
            module._classifier_module = module.classifier
            return module
        if isinstance(classifier, nn.Conv3d):
            module.classifier = _replace_conv3d_layer(classifier, num_classes)
            module._classifier_module = module.classifier
            return module
        if isinstance(classifier, nn.Sequential) and len(classifier) > 0 and isinstance(classifier[-1], (nn.Linear, nn.Conv3d)):
            classifier[-1] = _replace_classifier_layer(classifier[-1], num_classes)
            module.classifier = classifier
            module._classifier_module = classifier[-1]
            return module

    keyword_candidates = []
    for name, layer in module.named_modules():
        if not name:
            continue
        if not isinstance(layer, (nn.Linear, nn.Conv3d)):
            continue
        lower_name = name.lower()
        if any(keyword in lower_name for keyword in ("head", "classifier", "fc", "logits", "proj")):
            keyword_candidates.append((name, layer))

    if keyword_candidates:
        name, layer = keyword_candidates[-1]
        new_layer = _replace_classifier_layer(layer, num_classes)
        _set_nested_attr(module, name, new_layer)
        module._classifier_module = new_layer
        return module

    raise RuntimeError(
        f"Expected torchvision video model with a replaceable classifier head, got: {type(module).__name__}"
    )


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    registry = get_torchvision_video_registry()
    model_key = str(model_name).lower()
    if model_key not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported torchvision video model: {model_name}. Available: {available}")
    spec = registry[model_key]
    weights = spec["default_weights"] if pretrained else None
    model = spec["builder"](weights=weights)
    model = _replace_classifier_head(model, num_classes)
    model._benchmark_model_name = model_key
    model._benchmark_rgb_mean = torch.tensor(spec["mean"], dtype=torch.float32).view(1, 3, 1, 1, 1)
    model._benchmark_rgb_std = torch.tensor(spec["std"], dtype=torch.float32).view(1, 3, 1, 1, 1)
    return model


def build_dataset(args: argparse.Namespace, training: bool) -> RGBVideoClipDataset:
    RGBVideoClipDataset, _collate_rgb_clip = load_rgb_dataset_api()
    return RGBVideoClipDataset(
        root_dir=args.root_dir if training or not getattr(args, "val_root_dir", "") else args.val_root_dir,
        rgb_frames=args.rgb_frames,
        img_size=args.img_size,
        sampling_mode=args.rgb_sampling if training else "uniform",
        dataset_split_txt=args.manifest if training or not getattr(args, "val_manifest", "") else args.val_manifest,
        class_id_to_label_csv=(
            args.class_id_to_label_csv
            if training or not getattr(args, "val_class_id_to_label_csv", "")
            else args.val_class_id_to_label_csv
        ),
        rgb_norm="none",
        seed=args.seed,
        color_jitter_prob=getattr(args, "color_jitter", 0.0) if training else 0.0,
        p_hflip=getattr(args, "p_hflip", 0.0) if training else 0.0,
    )


def freeze_backbone_parameters(model: nn.Module) -> None:
    classifier = getattr(model, "_classifier_module", None)
    if classifier is None:
        raise RuntimeError("Expected model to expose _classifier_module for backbone freezing.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)


def freeze_backbone_bn_stats(model: nn.Module) -> None:
    classifier = getattr(model, "_classifier_module", None)
    classifier_modules = set(classifier.modules()) if classifier is not None else set()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            if module in classifier_modules:
                continue
            module.eval()


def save_checkpoint(payload: Dict[str, object], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    print(f"[CKPT] saved {save_path.as_posix()}", flush=True)


def normalize_rgb(x: torch.Tensor, model_name: str) -> torch.Tensor:
    registry = get_torchvision_video_registry()
    model_key = str(model_name).lower()
    spec = registry.get(model_key)
    if spec is None:
        mean_values, std_values = _default_mean_std_for_model(model_key)
        mean = torch.tensor(mean_values, dtype=torch.float32).view(1, 3, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        std = torch.tensor(std_values, dtype=torch.float32).view(1, 3, 1, 1, 1).to(device=x.device, dtype=x.dtype)
    else:
        mean = torch.tensor(spec["mean"], dtype=torch.float32).view(1, 3, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        std = torch.tensor(spec["std"], dtype=torch.float32).view(1, 3, 1, 1, 1).to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    k = min(k, int(logits.shape[1]))
    _, pred = logits.topk(k, dim=1)
    return int(pred.eq(y.view(-1, 1)).any(dim=1).sum().item())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def prf_from_cm(cm: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    precision = tp / (pred_sum + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1, support


def macro_weighted(values: np.ndarray, support: np.ndarray) -> Tuple[float, float]:
    macro = float(np.nanmean(values))
    weighted = float(np.nansum(values * support) / (np.sum(support) + 1e-12))
    return macro, weighted


# ---------------------------------------------------------------------------
# Per skin-tone variant helpers
# ---------------------------------------------------------------------------

_DARK_VARIANTS: frozenset = frozenset({"african", "indian"})
_LIGHT_VARIANTS: frozenset = frozenset({"white", "asian"})
_VARIANT_RE = re.compile(r"_modified_([^/_]+?)(?:\.mp4|\.avi|\.zst|$)", re.IGNORECASE)


def _extract_variant(path: str) -> str:
    """Return the skin-tone variant name embedded in a synthetic video filename."""
    m = _VARIANT_RE.search(str(path))
    if m:
        return m.group(1).lower()
    if "_initial." in str(path).lower():
        return "initial"
    return "unknown"


def _compute_per_variant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: List[str],
    classnames: List[str],
) -> Dict[str, object]:
    """Compute per skin-tone variant accuracy and tone-group gap.

    Returns a dict with keys ``per_variant`` and ``per_tone_group``.
    """
    variants = [_extract_variant(p) for p in paths]
    unique_variants = sorted({v for v in variants if v != "unknown"})
    n_classes = len(classnames)

    per_variant: Dict[str, object] = {}
    for variant in unique_variants:
        mask = np.array([v == variant for v in variants], dtype=bool)
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            continue
        cm = confusion_matrix(yt, yp, n_classes)
        prec, rec, f1, sup = prf_from_cm(cm)
        f1_macro, _ = macro_weighted(f1, sup)
        top1 = float((yt == yp).sum()) / float(len(yt))
        per_variant[variant] = {
            "count": int(len(yt)),
            "top1": float(top1),
            "f1_macro": float(f1_macro),
        }

    def _group_top1(group: frozenset) -> float:
        total = sum(per_variant[v]["count"] for v in group if v in per_variant)  # type: ignore[index]
        if total == 0:
            return float("nan")
        return sum(per_variant[v]["top1"] * per_variant[v]["count"] for v in group if v in per_variant) / total  # type: ignore[index]

    dark_top1 = _group_top1(_DARK_VARIANTS)
    light_top1 = _group_top1(_LIGHT_VARIANTS)
    gap = (light_top1 - dark_top1) if (dark_top1 == dark_top1 and light_top1 == light_top1) else float("nan")

    per_tone_group: Dict[str, object] = {
        "dark": {
            "count": sum(per_variant[v]["count"] for v in _DARK_VARIANTS if v in per_variant),  # type: ignore[index]
            "top1": dark_top1,
        },
        "light": {
            "count": sum(per_variant[v]["count"] for v in _LIGHT_VARIANTS if v in per_variant),  # type: ignore[index]
            "top1": light_top1,
        },
        "gap_light_minus_dark": gap,
    }

    return {"per_variant": per_variant, "per_tone_group": per_tone_group}


def save_cm_csv(cm: np.ndarray, classnames: List[str], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + classnames)
        for idx, class_name in enumerate(classnames):
            writer.writerow([class_name] + cm[idx].tolist())


def save_per_class_csv(classnames: List[str], precision: np.ndarray, recall: np.ndarray, f1: np.ndarray, support: np.ndarray, top1_acc: np.ndarray, out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "support", "precision", "recall", "f1", "top1_acc"])
        for class_name, sup, prec, rec, f1_val, acc in zip(classnames, support, precision, recall, f1, top1_acc):
            writer.writerow([class_name, int(sup), float(prec), float(rec), float(f1_val), float(acc)])


def build_summary(
    *,
    mode_name: str,
    split_name: str,
    metrics: Dict[str, float],
    per_variant_data: "Dict[str, object] | None" = None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "mode": mode_name,
        "num_splits": 1,
        "splits": {split_name: metrics},
        "aggregate": {key: {"mean": float(value), "std": 0.0} for key, value in metrics.items()},
    }
    if per_variant_data:
        summary["per_variant_splits"] = {split_name: per_variant_data}
    return summary


def aggregate_split_summaries(out_dir: Path, model_name: str) -> Dict[str, object]:
    mode_name = mode_name_for_model(model_name)
    split_summaries: Dict[str, Dict[str, float]] = {}
    per_variant_by_split: Dict[str, object] = {}
    for split_dir in sorted(path for path in out_dir.iterdir() if path.is_dir() and path.name.startswith("eval_")):
        summary_path = split_dir / f"summary_{mode_name}.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        split_metrics = payload.get("splits", {}).get(split_dir.name)
        if isinstance(split_metrics, dict):
            split_summaries[split_dir.name] = {str(k): float(v) for k, v in split_metrics.items()}
        per_variant = payload.get("per_variant_splits", {}).get(split_dir.name)
        if per_variant:
            per_variant_by_split[split_dir.name] = per_variant

    metric_names = sorted({metric for metrics in split_summaries.values() for metric in metrics})
    aggregate: Dict[str, Dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(metrics[metric_name]) for metrics in split_summaries.values() if metric_name in metrics]
        if not values:
            continue
        aggregate[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    result: Dict[str, object] = {
        "mode": mode_name,
        "model_name": str(model_name),
        "num_splits": len(split_summaries),
        "splits": split_summaries,
        "aggregate": aggregate,
    }
    if per_variant_by_split:
        result["per_variant_splits"] = per_variant_by_split
    return result


def evaluate_model(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    classnames: List[str],
    device: torch.device,
    split_name: str,
    out_dir: Path,
    summary_only: bool,
) -> Dict[str, float]:
    model.eval()
    mode_name = mode_name_for_model(getattr(model, "_benchmark_model_name", "r2plus1d_18"))
    y_true: List[int] = []
    y_pred: List[int] = []
    all_paths: List[str] = []
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for rgb, _dummy_second, labels, paths in dataloader:
            rgb = normalize_rgb(rgb.to(device, non_blocking=True), getattr(model, "_benchmark_model_name", "r3d_18"))
            labels = labels.to(device, non_blocking=True)
            logits = model(rgb)
            preds = logits.argmax(dim=1)
            top1_correct += int((preds == labels).sum().item())
            top5_correct += topk_correct(logits, labels, 5)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            all_paths.extend(list(paths))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true_np, y_pred_np, len(classnames))
    precision, recall, f1, support = prf_from_cm(cm)
    top1_acc = np.divide(np.diag(cm).astype(np.float64), support + 1e-12)
    acc1 = float(top1_correct / max(1, len(y_true_np)))
    acc5 = float(top5_correct / max(1, len(y_true_np)))
    mean_class_acc = float(np.nanmean(recall))
    f1_macro, f1_weighted = macro_weighted(f1, support)
    p_macro, p_weighted = macro_weighted(precision, support)
    r_macro, r_weighted = macro_weighted(recall, support)
    metrics = {
        "top1": acc1,
        "top5": acc5,
        "mean_class_acc": mean_class_acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }

    per_variant_data = _compute_per_variant_metrics(y_true_np, y_pred_np, all_paths, classnames)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary_only:
        (out_dir / f"metrics_{mode_name}.json").write_text(
            json.dumps({"mode": mode_name, "split": split_name, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        save_cm_csv(cm, classnames, out_dir / f"confusion_{mode_name}.csv")
        save_per_class_csv(classnames, precision, recall, f1, support, top1_acc, out_dir / f"per_class_{mode_name}.csv")

    summary = build_summary(
        mode_name=mode_name,
        split_name=split_name,
        metrics=metrics,
        per_variant_data=per_variant_data,
    )
    (out_dir / f"summary_{mode_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics


def evaluate_loader(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    classnames: List[str],
    device: torch.device,
    split_name: str,
    out_dir: Path,
) -> Dict[str, float]:
    return evaluate_model(
        model=model,
        dataloader=dataloader,
        classnames=classnames,
        device=device,
        split_name=split_name,
        out_dir=out_dir,
        summary_only=True,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _RGBVideoClipDataset, collate_rgb_clip = load_rgb_dataset_api()
    mixup_batch, soft_target_cross_entropy = load_augment_api()

    train_dataset = build_dataset(args, training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgb_clip,
        drop_last=False,
    )

    model = build_model(args.model, num_classes=len(train_dataset.classnames), pretrained=bool(args.pretrained)).to(device)
    model._benchmark_model_name = str(args.model)
    if bool(args.freeze_backbone):
        freeze_backbone_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = max(1, int(args.epochs) * max(1, len(train_loader)))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=float(args.lr),
        min_lr=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        total_steps=total_steps,
    )
    val_dataset = None
    val_loader = None
    if args.val_root_dir and args.val_manifest:
        val_dataset = build_dataset(args, training=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_rgb_clip,
            drop_last=False,
        )
    best_top1 = -float("inf")
    best_loss = float("inf")
    start_epoch = 0
    global_step = 0

    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device)
        resume_model_name = str(resume_payload.get("model_name", args.model))
        if resume_model_name != str(args.model):
            raise ValueError(
                f"Resume checkpoint model mismatch: ckpt={resume_model_name} requested={args.model}"
            )
        model.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        scaler_state = resume_payload.get("scaler_state")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        global_step = int(resume_payload.get("global_step", 0))
        best_top1 = float(resume_payload.get("best_top1", best_top1))
        best_loss = float(resume_payload.get("best_loss", best_loss))

    print(
        f"[CONFIG] model={args.model} pretrained={args.pretrained} rgb_frames={args.rgb_frames} img_size={args.img_size} "
        f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} manifest={args.manifest} "
        f"freeze_backbone={bool(args.freeze_backbone)} freeze_bn_stats={bool(args.freeze_bn_stats)} "
        f"mixup_prob={float(args.mixup_prob):.3f} mixup_alpha={float(args.mixup_alpha):.3f} "
        f"p_hflip={float(args.p_hflip):.3f} warmup_steps={int(args.warmup_steps)} min_lr={float(args.min_lr):.6g} "
        f"val_every={int(args.val_every)} checkpoint_mode={args.checkpoint_mode} "
        f"resume_ckpt={args.resume_ckpt or 'none'} start_epoch={start_epoch + 1}",
        flush=True,
    )
    if start_epoch >= int(args.epochs):
        print(
            f"[RESUME] checkpoint already reached epoch {start_epoch}; nothing to do for epochs={args.epochs}",
            flush=True,
        )
        return
    if args.resume_ckpt:
        print(
            f"[RESUME] loaded {args.resume_ckpt} (next_epoch={start_epoch + 1:03d}, global_step={global_step})",
            flush=True,
        )

    for epoch in range(start_epoch, int(args.epochs)):
        train_dataset.set_epoch(epoch)
        model.train()
        if bool(args.freeze_bn_stats):
            freeze_backbone_bn_stats(model)
        running_loss = 0.0
        num_batches = 0
        start_time = time.time()
        for step, (rgb, _dummy_second, labels, _paths) in enumerate(train_loader, start=1):
            rgb = normalize_rgb(rgb.to(device, non_blocking=True), args.model)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            labels_soft = None
            if float(args.mixup_prob) > 0 and float(args.mixup_alpha) > 0 and labels.size(0) > 1:
                if random.random() < float(args.mixup_prob):
                    dummy_second = torch.zeros(
                        (rgb.shape[0], 1, 1, 1, 1),
                        device=rgb.device,
                        dtype=rgb.dtype,
                    )
                    rgb, _dummy_second_mix, labels_soft = mixup_batch(
                        rgb,
                        dummy_second,
                        labels,
                        num_classes=len(train_dataset.classnames),
                        alpha=float(args.mixup_alpha),
                        label_smoothing=float(args.label_smoothing),
                    )
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(rgb)
                if labels_soft is not None:
                    loss = soft_target_cross_entropy(logits, labels_soft)
                else:
                    loss = F.cross_entropy(logits, labels, label_smoothing=float(args.label_smoothing))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach().item())
            num_batches += 1
            global_step += 1
            if args.log_every > 0 and step % args.log_every == 0:
                step_elapsed = time.time() - start_time
                print(
                    f"[STEP {epoch+1:03d}:{step:04d}] loss={loss.detach().item():.4f} "
                    f"lr={float(optimizer.param_groups[0]['lr']):.6g} "
                    f"elapsed={step_elapsed:.1f}s",
                    flush=True,
                )

        epoch_loss = running_loss / max(1, num_batches)
        elapsed = time.time() - start_time
        print(f"[EPOCH {epoch+1:03d}] loss={epoch_loss:.4f} time={elapsed:.1f}s", flush=True)

        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_name": args.model,
            "num_classes": len(train_dataset.classnames),
            "classnames": list(train_dataset.classnames),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_top1": best_top1,
            "best_loss": best_loss,
            "args": vars(args),
        }
        checkpoint_mode = str(args.checkpoint_mode).lower()
        epochs_completed = epoch + 1
        is_final_epoch = epochs_completed >= int(args.epochs)
        do_val = val_loader is not None and int(args.val_every) > 0 and epochs_completed % int(args.val_every) == 0

        if do_val:
            metrics = evaluate_loader(
                model=model,
                dataloader=val_loader,
                classnames=list(val_dataset.classnames),
                device=device,
                split_name="validation",
                out_dir=out_dir / "eval_validation",
            )
            top1 = float(metrics["top1"])
            improved = top1 > best_top1
            print(
                f"[VAL] top1={top1:.6f} best={best_top1 if best_top1 > -1e8 else float('nan'):.6f} improved={improved}",
                flush=True,
            )
            should_save = (
                checkpoint_mode == "latest"
                or (checkpoint_mode == "final" and is_final_epoch)
                or (checkpoint_mode == "best" and improved)
            )
            if should_save:
                if checkpoint_mode == "latest":
                    save_path = ckpt_dir / args.checkpoint_name
                elif checkpoint_mode == "final":
                    save_path = ckpt_dir / "checkpoint_final.pt"
                else:
                    save_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}_top1_{top1:.4f}.pt"
                save_checkpoint(payload, save_path)
            if improved:
                best_top1 = top1
        else:
            should_save = (
                checkpoint_mode == "latest"
                or (checkpoint_mode == "final" and is_final_epoch)
                or (checkpoint_mode == "best" and epoch_loss < best_loss)
            )
            if should_save:
                if checkpoint_mode == "latest":
                    save_path = ckpt_dir / args.checkpoint_name
                elif checkpoint_mode == "final":
                    save_path = ckpt_dir / "checkpoint_final.pt"
                else:
                    save_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}_loss_{epoch_loss:.4f}.pt"
                save_checkpoint(payload, save_path)
            if epoch_loss < best_loss:
                best_loss = epoch_loss


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = str(ckpt.get("model_name", args.model))
    _RGBVideoClipDataset, collate_rgb_clip = load_rgb_dataset_api()

    eval_dataset = build_dataset(args, training=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgb_clip,
        drop_last=False,
    )

    model = build_model(model_name, num_classes=len(eval_dataset.classnames), pretrained=False).to(device)
    model._benchmark_model_name = model_name
    model.load_state_dict(ckpt["model_state"], strict=True)

    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        classnames=list(eval_dataset.classnames),
        device=device,
        split_name=args.split_name,
        out_dir=Path(args.out_dir),
        summary_only=bool(args.summary_only),
    )
    print(json.dumps(metrics, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    summary = aggregate_split_summaries(out_dir, model_name=str(args.model))
    out_path = out_dir / f"summary_{mode_name_for_model(args.model)}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": out_path.as_posix(), "num_splits": summary["num_splits"]}, indent=2), flush=True)


def list_models_cmd(args: argparse.Namespace) -> None:
    models = list_available_models()
    if args.json:
        print(json.dumps(models, indent=2), flush=True)
        return
    for model_name in models:
        print(model_name, flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "list_models":
        list_models_cmd(args)
    elif args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
