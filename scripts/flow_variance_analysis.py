import numpy as np
from pathlib import Path

FLOW_ROOT = Path("/Volumes/MoDDL/Pascal/motion_only_AR/datasets/skin_tone_actions/camera_far_flow_tvl1_npz")
BGs = ["autumn_hockey", "konzerthaus", "stadium_01"]
actions = ["cartwheel", "celebrate", "clap", "dribble", "fish", "golf", "lunge", "squat", "tie", "yawn"]

results = {}
for action in actions:
    mags, tstds = [], []
    for bg in BGs:
        adir = FLOW_ROOT / bg / "__generated_synthetic_videos" / action
        if not adir.exists():
            continue
        for npz_path in sorted(adir.glob(f"{action}_*.npz")):
            d = np.load(npz_path)
            flow = d["flow"].astype(np.float32) - 128.0
            mag = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5  # (T, H, W)
            per_frame = mag.mean(axis=(1, 2))  # (T,)
            mags.append(per_frame.mean())
            tstds.append(per_frame.std())
    mags = np.array(mags)
    tstds = np.array(tstds)
    results[action] = {
        "n": len(mags),
        "mean_mag": mags.mean(),
        "std_mag": mags.std(),    # intra-class variability of motion intensity
        "mean_tsd": tstds.mean(),
        "std_tsd": tstds.std(),
        "mags": mags,
    }
    print(f"  done {action} ({len(mags)} clips)", flush=True)

print()
print(f"{'Action':<12} {'n':>5}  {'mean_mag':>9}  {'std_mag':>9}  {'mean_tsd':>9}  {'std_tsd':>9}")
print("-" * 62)
for action in sorted(results, key=lambda a: -results[a]["std_mag"]):
    r = results[action]
    print(f"{action:<12} {r['n']:>5}  {r['mean_mag']:>9.4f}  {r['std_mag']:>9.4f}  {r['mean_tsd']:>9.4f}  {r['std_tsd']:>9.4f}")

# Per-action pair: compare variability
print()
print("=== Action pair comparison (std_mag = intra-class motion variability) ===")
pairs = [("lunge","cartwheel"),("squat","tie"),("clap","celebrate"),("dribble","golf"),("yawn","fish")]
for a, b in pairs:
    ra, rb = results[a], results[b]
    print(f"  {a} vs {b}: std_mag {ra['std_mag']:.4f} vs {rb['std_mag']:.4f}  |  mean_mag {ra['mean_mag']:.4f} vs {rb['mean_mag']:.4f}")
