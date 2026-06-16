from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIN_TONE_DIR = REPO_ROOT / "benchmarks" / "skin_tone"
for _path in (REPO_ROOT, SKIN_TONE_DIR):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import analyze_skin_tone_swap_influence as swap_influence
import build_skin_tone_shortcut_probe as manifest_builder
import compare_color_jitter_conditions as color_jitter
import summarize_skin_tone_robustness as robustness
import summarize_skin_tone_significance as significance
from aggregate_skin_tone_probe import load_rows, load_rows_with_report


EVAL_SPLITS = (
    "eval_matched_seen_ids",
    "eval_matched_unseen_ids",
    "eval_shifted_seen_ids",
    "eval_shifted_unseen_ids",
)


def _touch_dataset_video(dataset_root: Path, background: str, action: str, base_id: int, variant: str) -> None:
    video_dir = dataset_root / background / "__generated_synthetic_videos" / action
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / f"{action}_{base_id}_modified_{variant}.mp4").write_text("fixture", encoding="utf-8")


def _make_complete_skin_tone_dataset(
    dataset_root: Path,
    *,
    backgrounds: Iterable[str],
    actions: Iterable[str],
    base_ids: Iterable[int],
    variants: Iterable[str] = ("african", "indian", "white", "asian"),
) -> None:
    for background in backgrounds:
        for action in actions:
            for base_id in base_ids:
                for variant in variants:
                    _touch_dataset_video(dataset_root, background, action, int(base_id), variant)


def _write_summary(path: Path, *, mode: str, split: str, f1_macro: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "num_splits": 1,
        "splits": {split: {"f1_macro": float(f1_macro), "top1": float(f1_macro)}},
        "aggregate": {
            "f1_macro": {"mean": float(f1_macro), "std": 0.0},
            "top1": {"mean": float(f1_macro), "std": 0.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_aggregate_summary(run_dir: Path, *, model: str, values_by_split: dict[str, float]) -> None:
    mode = f"rgb_{model}_model"
    payload = {
        "mode": mode,
        "model_name": model,
        "num_splits": len(values_by_split),
        "splits": {
            split: {"f1_macro": float(value), "top1": float(value)}
            for split, value in values_by_split.items()
        },
        "aggregate": {
            "f1_macro": {
                "mean": sum(values_by_split.values()) / max(1, len(values_by_split)),
                "std": 0.0,
            }
        },
    }
    (run_dir / f"summary_{mode}.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_rgb_run(root: Path, *, model: str, pair_tag: str, seed: int, values_by_split: dict[str, float]) -> Path:
    run_dir = root / "rgb_torchvision" / model / pair_tag / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = f"rgb_{model}_model"
    for split, value in values_by_split.items():
        _write_summary(run_dir / split / f"summary_{mode}.json", mode=mode, split=split, f1_macro=value)
    _write_aggregate_summary(run_dir, model=model, values_by_split=values_by_split)
    _write_summary(
        run_dir / "eval_matched_seen_ids" / "summary_rgb_wrong_model.json",
        mode="rgb_wrong_model",
        split="eval_matched_seen_ids",
        f1_macro=0.01,
    )
    return run_dir


def _prediction_row(rel_path: str, *, variant: str, y_true: int, y_pred: int, correct: int) -> dict[str, object]:
    stem = Path(rel_path).stem
    base_id = int(stem.split("_")[1])
    return {
        "model": "r3d_18",
        "pair_tag": "jump_vs_wave",
        "seed": 0,
        "eval_split": "",
        "rel_path": rel_path,
        "background": "bg_a",
        "action": "jump",
        "base_id": base_id,
        "variant": variant,
        "tone_group": "dark" if variant in {"african", "indian"} else "light",
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": correct,
        "top1_prob": 0.7,
        "top2_prob": 0.3,
        "margin": 0.4,
        "entropy": 0.5,
        "true_class_prob": 0.7 if correct else 0.3,
        "luma_mean": 0.1,
        "luma_std": 0.2,
        "saturation_mean": 0.3,
        "hue_mean": 0.4,
        "contrast": 0.5,
        "r_mean": 0.6,
        "g_mean": 0.7,
        "b_mean": 0.8,
    }


def _write_predictions(path: Path, rows: list[dict[str, object]], *, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["eval_split"] = split
            writer.writerow(item)


class SkinTonePipelineTests(unittest.TestCase):
    def test_manifest_builder_balances_splits_and_honors_explicit_train_cap(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            dataset_root = tmp_path / "dataset"
            _make_complete_skin_tone_dataset(
                dataset_root,
                backgrounds=("bg_a", "bg_b"),
                actions=("jump", "wave"),
                base_ids=(0, 1, 2, 3),
            )

            old_manifests_root = manifest_builder.MANIFESTS_ROOT
            old_labels_root = manifest_builder.LABELS_ROOT
            old_argv = list(sys.argv)
            manifest_builder.MANIFESTS_ROOT = tmp_path / "generated" / "manifests"
            manifest_builder.LABELS_ROOT = tmp_path / "generated" / "labels"
            sys.argv = [
                "build_skin_tone_shortcut_probe.py",
                "--dataset_root",
                str(dataset_root),
                "--dark_action",
                "jump",
                "--light_action",
                "wave",
                "--pair_tag",
                "jump_vs_wave",
                "--backgrounds",
                "bg_a,bg_b",
                "--train_ids",
                "0,1,2",
                "--same_id_eval_ids",
                "0,1,2",
                "--disjoint_eval_ids",
                "3",
                "--train_max_samples_per_class",
                "2",
                "--eval_max_samples_per_class",
                "1",
            ]
            try:
                manifest_builder.main()
            finally:
                manifest_builder.MANIFESTS_ROOT = old_manifests_root
                manifest_builder.LABELS_ROOT = old_labels_root
                sys.argv = old_argv

            summary_path = (
                tmp_path
                / "generated"
                / "manifests"
                / "skin_tone_camera_far_binary"
                / "jump_vs_wave"
                / "summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            train = summary["manifests"]["train_in_domain"]
            self.assertEqual(train["num_entries"], 4)
            self.assertEqual(train["counts_by_label"], {"0": 2, "1": 2})
            self.assertEqual(train["max_samples_per_class"], 2)

            shifted = summary["manifests"]["eval_shifted_unseen_ids"]
            self.assertEqual(shifted["num_entries"], 2)
            self.assertEqual(shifted["dark_variants"], ["white", "asian"])
            self.assertEqual(shifted["light_variants"], ["african", "indian"])

    def test_result_loading_deduplicates_split_summaries_and_feeds_robustness_and_jitter_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            baseline = tmp_path / "cj0p0"
            jittered = tmp_path / "cj0p4"
            base_values = {
                "eval_matched_seen_ids": 0.90,
                "eval_shifted_seen_ids": 0.70,
                "eval_matched_unseen_ids": 0.80,
                "eval_shifted_unseen_ids": 0.50,
            }
            jitter_values = {
                "eval_matched_seen_ids": 0.88,
                "eval_shifted_seen_ids": 0.82,
                "eval_matched_unseen_ids": 0.78,
                "eval_shifted_unseen_ids": 0.70,
            }
            _make_rgb_run(baseline, model="r3d_18", pair_tag="jump_vs_wave", seed=0, values_by_split=base_values)
            _make_rgb_run(jittered, model="r3d_18", pair_tag="jump_vs_wave", seed=0, values_by_split=jitter_values)

            rows = load_rows(baseline)
            reported_rows, report = load_rows_with_report(baseline)
            self.assertEqual(len(rows), 4)
            self.assertEqual(reported_rows, rows)
            self.assertEqual(report.accepted_rows, 4)
            self.assertEqual(len(report.skipped_mode_mismatches), 1)
            self.assertEqual({row["eval_split"] for row in rows}, set(EVAL_SPLITS))
            self.assertEqual({row["mode"] for row in rows}, {"rgb_r3d_18_model"})

            per_seed = robustness.build_per_seed_rows(rows, "f1_macro")
            self.assertEqual(len(per_seed), 1)
            self.assertAlmostEqual(per_seed[0]["f1_macro_drop_training_videos"], 0.20)
            self.assertAlmostEqual(per_seed[0]["f1_macro_drop_testing_videos"], 0.30)

            modality_rows = robustness.summarize_modalities(per_seed, "f1_macro")
            self.assertEqual(modality_rows[0]["display_name"], "r3d_18")
            self.assertEqual(modality_rows[0]["num_units"], 1)

            specs = [("cj0p0", baseline, 0.0), ("cj0p4", jittered, 0.4)]
            condition_rows, _ = color_jitter.summarize_condition(
                label="cj0p0",
                root=baseline,
                condition_value=0.0,
                metric_name="f1_macro",
            )
            self.assertAlmostEqual(condition_rows[0]["f1_macro_drop_testing_videos_mean"], 0.30)

            robustness_rows = color_jitter.build_robustness_rows(specs, "f1_macro")
            drop_row = next(row for row in robustness_rows if row["metric"] == "drop_testing_videos")
            self.assertAlmostEqual(drop_row["reference_mean"], 0.30)
            self.assertAlmostEqual(drop_row["comparator_mean"], 0.08)
            self.assertAlmostEqual(drop_row["comparator_minus_reference_mean"], -0.22)

    def test_prediction_pairing_and_variant_significance_use_directional_skin_tone_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "run"
            matched_dir = root / "rgb_torchvision" / "r3d_18" / "jump_vs_wave" / "seed_0" / "eval_matched_unseen_ids"
            shifted_dir = root / "rgb_torchvision" / "r3d_18" / "jump_vs_wave" / "seed_0" / "eval_shifted_unseen_ids"

            matched_rows = [
                _prediction_row("bg_a/__generated_synthetic_videos/jump/jump_0_modified_african.mp4", variant="african", y_true=0, y_pred=0, correct=1),
                _prediction_row("bg_a/__generated_synthetic_videos/jump/jump_1_modified_indian.mp4", variant="indian", y_true=0, y_pred=1, correct=0),
                _prediction_row("bg_a/__generated_synthetic_videos/jump/jump_2_modified_african.mp4", variant="african", y_true=0, y_pred=0, correct=1),
            ]
            shifted_rows = [
                _prediction_row("bg_a/__generated_synthetic_videos/jump/jump_0_modified_white.mp4", variant="white", y_true=0, y_pred=1, correct=0),
                _prediction_row("bg_a/__generated_synthetic_videos/jump/jump_1_modified_asian.mp4", variant="asian", y_true=0, y_pred=0, correct=1),
            ]
            _write_predictions(matched_dir / "predictions_rgb_r3d_18.csv", matched_rows, split="eval_matched_unseen_ids")
            _write_predictions(shifted_dir / "predictions_rgb_r3d_18.csv", shifted_rows, split="eval_shifted_unseen_ids")

            prediction_rows, prediction_csvs = swap_influence.load_prediction_rows(root, models_filter=None)
            pair_rows, report = swap_influence.build_pair_rows(prediction_rows, ["unseen"])

            self.assertEqual(len(prediction_csvs), 2)
            self.assertEqual(len(pair_rows), 2)
            self.assertEqual(report["total_missing_counterpart"], 1)
            self.assertEqual({row["variant_matched"] for row in pair_rows}, {"african", "indian"})
            self.assertEqual({row["variant_shifted"] for row in pair_rows}, {"white", "asian"})

            african_pair = next(row for row in pair_rows if row["variant_matched"] == "african")
            self.assertEqual(african_pair["correctness_drop"], 1)
            self.assertEqual(african_pair["pred_flip"], 1)

            variant_rows = significance.compute_variant_significance_rows(pair_rows, "unseen")
            african_stats = next(row for row in variant_rows if row["variant_matched"] == "african")
            indian_stats = next(row for row in variant_rows if row["variant_matched"] == "indian")
            self.assertEqual(african_stats["n_matched_correct_shifted_wrong"], 1)
            self.assertAlmostEqual(african_stats["accuracy_drop"], 1.0)
            self.assertEqual(indian_stats["n_matched_wrong_shifted_correct"], 1)
            self.assertAlmostEqual(indian_stats["accuracy_drop"], -1.0)


if __name__ == "__main__":
    unittest.main()
