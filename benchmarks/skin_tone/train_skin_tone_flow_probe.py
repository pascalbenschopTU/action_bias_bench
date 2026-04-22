from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import REPO_ROOT


PRESETS: Dict[str, Dict[str, object]] = {
    "i3d_raft_flow_only": {
        "summary_name": "flow_i3d_raft_only",
        "model_name": "i3d",
        "default_pretrained_ckpt": "out/train_i3d_224_raft_of_only/checkpoints/checkpoint_epoch_014_loss2.6486.pt",
        "default_video_root_dir": "",
        "default_zstd_root_dir": "",
        "default_motion_data_source": "video",
        "finetune_configs": ["configs/benchmarks/skin_tone/finetune/common.toml"],
        "eval_configs": ["configs/benchmarks/skin_tone/eval/common.toml"],
    },
    "x3d_e2s_second_raft": {
        "summary_name": "flow_x3d_e2s_second_raft",
        "model_name": "x3d",
        "default_pretrained_ckpt": "out/train_x3d_e2s_raft_only_ce/checkpoints/checkpoint_epoch_014_loss3.6727.pt",
        "default_video_root_dir": "",
        "default_zstd_root_dir": "",
        "default_motion_data_source": "zstd",
        "finetune_configs": [
            "configs/benchmarks/skin_tone/finetune/common.toml",
            "configs/benchmarks/skin_tone/finetune/x3d_flow_only.toml",
        ],
        "eval_configs": [
            "configs/benchmarks/skin_tone/eval/common.toml",
            "configs/benchmarks/skin_tone/eval/x3d_flow_only.toml",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a pure optical-flow skin-tone probe using local motion checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    add_common_args(train)
    train.add_argument("--out_dir", type=str, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch_size", type=int, default=8)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--num_workers", type=int, default=8)
    train.add_argument("--device", type=str, default="cuda")

    ev = subparsers.add_parser("eval")
    add_common_args(ev)
    ev.add_argument("--ckpt", type=str, required=True)
    ev.add_argument("--out_dir", type=str, required=True)
    ev.add_argument("--split_name", type=str, default="eval")
    ev.add_argument("--batch_size", type=int, default=8)
    ev.add_argument("--num_workers", type=int, default=8)
    ev.add_argument("--device", type=str, default="cuda")
    ev.add_argument("--summary_only", action="store_true")

    ag = subparsers.add_parser("aggregate")
    ag.add_argument("--preset", type=str, default="i3d_raft_flow_only", choices=sorted(PRESETS))
    ag.add_argument("--out_dir", type=str, required=True)

    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", type=str, default="i3d_raft_flow_only", choices=sorted(PRESETS))
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--class_id_to_label_csv", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--motion_data_source", type=str, default="auto", choices=["auto", "video", "zstd"])
    parser.add_argument("--seed", type=int, default=0)


def resolve_preset(args: argparse.Namespace) -> Dict[str, object]:
    preset = dict(PRESETS[str(args.preset)])
    motion_data_source = str(args.motion_data_source)
    if motion_data_source == "auto":
        motion_data_source = str(preset["default_motion_data_source"])
    preset["motion_data_source"] = motion_data_source
    if args.root_dir:
        preset["root_dir"] = str(args.root_dir)
    else:
        key = "default_zstd_root_dir" if motion_data_source == "zstd" else "default_video_root_dir"
        preset["root_dir"] = str(preset[key])
    preset["pretrained_ckpt"] = str(args.pretrained_ckpt or preset["default_pretrained_ckpt"])
    return preset


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def add_configs(cmd: List[str], config_paths: List[str]) -> None:
    for config_path in config_paths:
        cmd.extend(["--config", str(config_path)])


def train(args: argparse.Namespace) -> None:
    preset = resolve_preset(args)
    cmd = [sys.executable, "finetune.py"]
    add_configs(cmd, list(preset["finetune_configs"]))
    cmd.extend(
        [
            "--root_dir",
            str(preset["root_dir"]),
            "--manifest",
            str(args.manifest),
            "--class_id_to_label_csv",
            str(args.class_id_to_label_csv),
            "--pretrained_ckpt",
            str(preset["pretrained_ckpt"]),
            "--out_dir",
            str(args.out_dir),
            "--seed",
            str(args.seed),
            "--val_subset_seed",
            str(args.seed),
            "--train_modality",
            "motion",
            "--val_modality",
            "motion",
            "--motion_data_source",
            str(preset["motion_data_source"]),
            "--active_branch",
            "second",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--num_workers",
            str(args.num_workers),
            "--device",
            str(args.device),
        ]
    )
    if str(preset["model_name"]) == "i3d":
        cmd.extend(["--model", "i3d"])
        if str(preset["motion_data_source"]) == "video":
            cmd.extend(["--flow_backend", "raft_large"])
    else:
        cmd.extend(["--model", "x3d"])
    run_cmd(cmd)


def evaluate(args: argparse.Namespace) -> None:
    preset = resolve_preset(args)
    cmd = [sys.executable, "eval.py"]
    add_configs(cmd, list(preset["eval_configs"]))
    cmd.extend(
        [
            "--root_dir",
            str(preset["root_dir"]),
            "--ckpt",
            str(args.ckpt),
            "--manifests",
            str(args.manifest),
            "--class_id_to_label_csv",
            str(args.class_id_to_label_csv),
            "--out_dir",
            str(args.out_dir),
            "--active_branch",
            "second",
            "--motion_data_source",
            str(preset["motion_data_source"]),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--device",
            str(args.device),
            "--no_clip",
        ]
    )
    if args.summary_only:
        cmd.append("--summary_only")
    if str(preset["model_name"]) == "i3d" and str(preset["motion_data_source"]) == "video":
        cmd.extend(["--flow_backend", "raft_large"])
    run_cmd(cmd)


def aggregate_split_summaries(out_dir: Path, summary_name: str, preset_name: str, model_name: str) -> Dict[str, object]:
    split_summaries: Dict[str, Dict[str, float]] = {}
    for split_dir in sorted(path for path in out_dir.iterdir() if path.is_dir() and path.name.startswith("eval_")):
        summary_path = split_dir / "summary_motion_only.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        split_metrics = payload.get("splits", {}).get(split_dir.name)
        if isinstance(split_metrics, dict):
            split_summaries[split_dir.name] = {str(k): float(v) for k, v in split_metrics.items()}

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

    return {
        "mode": "motion_only",
        "summary_name": summary_name,
        "preset": preset_name,
        "model_name": model_name,
        "num_splits": len(split_summaries),
        "splits": split_summaries,
        "aggregate": aggregate,
    }


def aggregate(args: argparse.Namespace) -> None:
    preset = dict(PRESETS[str(args.preset)])
    out_dir = Path(args.out_dir)
    summary = aggregate_split_summaries(
        out_dir=out_dir,
        summary_name=str(preset["summary_name"]),
        preset_name=str(args.preset),
        model_name=str(preset["model_name"]),
    )
    out_path = out_dir / f"summary_{str(preset['summary_name'])}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path.as_posix())


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
