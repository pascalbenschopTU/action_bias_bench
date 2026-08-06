"""
Pooled directional McNemar re-analysis of the ZERO-SHOT synthetic-audit predictions,
replacing the per-pair Bonferroni divergence comparison in the paper's Supplementary A
with the same statistical machinery the main paper uses for the fine-tuning stress
test (Sec 4.1): pool clip-level correctness flips across all actions/instances, run an
exact McNemar test per skin-tone pair, and control FDR within each model with
Benjamini-Hochberg.

Input: Ana's zero-shot prediction CSVs, one per (camera, background, model):
    <root>/<camera>/<background>/<model>/video_results.csv
    columns: video,true_label,raw_prediction,mapped_prediction,result,top5
    video names: <action>_<id>_initial.mp4 / <action>_<id>_modified_<tone>.mp4

The 'initial' render's skin tone is inferred as the one modified_<tone> variant that
is absent for that (action, id) - verified: every (action, id) is missing exactly one
of the 7 tones, consistently across all 6 camera/background configs.

For a tone pair (A, B) and a set of paired clips (same action, id, camera,
background; only the skin texture differs):
    b = clips correct under A but incorrect under B
    c = clips incorrect under A but correct under B
Exact two-sided McNemar = binomial test of b successes out of b+c at p=0.5.
b > c means tone B is the harmed direction.

Two config modes are reported:
  - all:  pool every camera x background config (max data; near-chance configs
          contribute mostly non-discordant pairs and dilute nothing).
  - best: per (model, action), keep only the camera x background config with the
          highest accuracy for that action - matches the paper's zero-shot protocol
          of selecting a reliable viewpoint/background per action (Supp B.1).

Usage (run from the ActionBiasBench directory):
    python scripts/zero_shot_pooled_mcnemar.py
    python scripts/zero_shot_pooled_mcnemar.py --models mvit_base_16x4 slowfast_r50
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/Volumes/MoDDL/Pascal/Video_LLM_testing/datasets_AR/ana_bedlam_actions/top_20_kinetics_actions"
)
DEFAULT_OUT = REPO_ROOT / "out" / "zero_shot_pooled_mcnemar"

TONES = ["african", "asian", "hispanic", "indian", "middle_eastern", "south_east_asian", "white"]
CAMERAS = ["camera_far", "camera_near"]
BACKGROUNDS = ["autumn_hockey", "konzerthaus", "stadium_01"]
VIDEO_RE = re.compile(r"(.+)_(\d+)_(?:initial|modified_(\w+))\.mp4")


def load_predictions(root: Path, models: List[str]):
    """Returns correct[(model, camera, background, action, vid, tone)] = bool."""
    correct: Dict[Tuple[str, str, str, str, str, str], bool] = {}
    tones_seen: Dict[Tuple[str, str], set] = defaultdict(set)
    raw_rows: List[Tuple[str, str, str, str, str, Optional[str], bool]] = []

    for camera in CAMERAS:
        for background in BACKGROUNDS:
            for model in models:
                path = root / camera / background / model / "video_results.csv"
                if not path.is_file():
                    print(f"[WARN] missing {path}", file=sys.stderr)
                    continue
                for row in csv.DictReader(path.open()):
                    m = VIDEO_RE.match(row["video"])
                    if not m:
                        raise ValueError(f"unparseable video name: {row['video']}")
                    action, vid, tone = m.group(1), m.group(2), m.group(3)
                    is_correct = row["result"].strip().lower() == "correct"
                    raw_rows.append((model, camera, background, action, vid, tone, is_correct))
                    if tone is not None:
                        tones_seen[(action, vid)].add(tone)

    # Infer each (action, id)'s initial tone as the single absent modified tone.
    initial_tone: Dict[Tuple[str, str], str] = {}
    for key, seen in tones_seen.items():
        missing = set(TONES) - seen
        if len(missing) != 1:
            raise ValueError(f"cannot infer initial tone for {key}: missing={missing}")
        initial_tone[key] = next(iter(missing))

    for model, camera, background, action, vid, tone, is_correct in raw_rows:
        resolved = tone if tone is not None else initial_tone[(action, vid)]
        correct[(model, camera, background, action, vid, resolved)] = is_correct
    return correct, initial_tone


def best_config_per_action(correct, model: str) -> Dict[str, Tuple[str, str]]:
    """Per action, the (camera, background) with highest accuracy for this model."""
    acc: Dict[Tuple[str, str, str], List[int]] = defaultdict(lambda: [0, 0])
    for (m, camera, background, action, _vid, _tone), ok in correct.items():
        if m != model:
            continue
        entry = acc[(action, camera, background)]
        entry[0] += int(ok)
        entry[1] += 1
    best: Dict[str, Tuple[str, str]] = {}
    best_acc: Dict[str, float] = {}
    for (action, camera, background), (n_ok, n) in sorted(acc.items()):
        a = n_ok / n if n else 0.0
        if action not in best or a > best_acc[action]:
            best[action] = (camera, background)
            best_acc[action] = a
    return best


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial test of b out of b+c at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5**n for k in range(n + 1)]
    observed = pmf[b]
    return min(1.0, sum(p for p in pmf if p <= observed * (1 + 1e-9)))


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    running_min = 1.0
    for rank_from_end in range(m, 0, -1):
        i = order[rank_from_end - 1]
        running_min = min(running_min, p_values[i] * m / rank_from_end)
        q[i] = running_min
    return q


def pooled_pair_tests(correct, model: str, config_filter) -> List[Dict[str, object]]:
    by_clip: Dict[Tuple[str, str, str, str], Dict[str, bool]] = defaultdict(dict)
    for (m, camera, background, action, vid, tone), ok in correct.items():
        if m != model or not config_filter(action, camera, background):
            continue
        by_clip[(camera, background, action, vid)][tone] = ok

    rows: List[Dict[str, object]] = []
    for tone_a, tone_b in combinations(TONES, 2):
        b = c = both_correct = both_wrong = 0
        for tone_map in by_clip.values():
            if tone_a not in tone_map or tone_b not in tone_map:
                continue
            ok_a, ok_b = tone_map[tone_a], tone_map[tone_b]
            if ok_a and not ok_b:
                b += 1
            elif ok_b and not ok_a:
                c += 1
            elif ok_a:
                both_correct += 1
            else:
                both_wrong += 1
        rows.append(
            {
                "model": model,
                "tone_a": tone_a,
                "tone_b": tone_b,
                "n_pairs": b + c + both_correct + both_wrong,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "b_correct_a_wrong_b": b,
                "c_wrong_a_correct_b": c,
                "net_harm_toward": tone_b if b > c else (tone_a if c > b else ""),
                "p_mcnemar": mcnemar_exact_p(b, c),
            }
        )
    q_values = benjamini_hochberg([float(r["p_mcnemar"]) for r in rows])
    for row, q in zip(rows, q_values):
        row["q_bh"] = q
    return rows


def overall_accuracy(correct, model: str, config_filter) -> Tuple[float, int]:
    n_ok = n = 0
    for (m, camera, background, action, _vid, _tone), ok in correct.items():
        if m != model or not config_filter(action, camera, background):
            continue
        n_ok += int(ok)
        n += 1
    return (n_ok / n if n else float("nan")), n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mvit_base_16x4", "slowfast_r50", "slow_r50", "x3d_xs"],
        help="Models to analyze (paper retained mvit_base_16x4 and slowfast_r50; slow_r50/x3d_xs near-chance).",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    correct, _ = load_predictions(args.root, args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for model in args.models:
        best = best_config_per_action(correct, model)
        modes = {
            "all": lambda action, camera, background: True,
            "best": lambda action, camera, background: best.get(action) == (camera, background),
        }
        for mode, config_filter in modes.items():
            acc, n = overall_accuracy(correct, model, config_filter)
            print(f"\n=== {model} | configs={mode} | overall acc={acc:.3f} (n={n}) ===")
            rows = pooled_pair_tests(correct, model, config_filter)
            for row in rows:
                row["config_mode"] = mode
            all_rows.extend(rows)
            significant = [r for r in rows if float(r["q_bh"]) < args.alpha]
            header = f"{'pair':34s} {'b':>4s} {'c':>4s} {'p':>10s} {'q(BH)':>10s}  harmed"
            print(header)
            for row in sorted(rows, key=lambda r: float(r["q_bh"])):
                mark = "*" if float(row["q_bh"]) < args.alpha else " "
                print(
                    f"{row['tone_a']} vs {row['tone_b']:20s} {row['b_correct_a_wrong_b']:4d} "
                    f"{row['c_wrong_a_correct_b']:4d} {float(row['p_mcnemar']):10.4f} "
                    f"{float(row['q_bh']):10.4f} {mark} {row['net_harm_toward']}"
                )
            print(f"significant after BH (q<{args.alpha}): {len(significant)}/21")

    out_csv = args.out_dir / "pooled_mcnemar_by_tone_pair.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
