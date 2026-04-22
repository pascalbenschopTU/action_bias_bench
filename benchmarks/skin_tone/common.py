from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / "finetune.py").exists() and (candidate / "benchmarks").exists():
            return candidate
    raise RuntimeError(f"Could not locate ActionBiasBench repo root from {current}")


REPO_ROOT = find_repo_root()
BENCHMARK_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = BENCHMARK_ROOT / "generated"
MANIFESTS_ROOT = GENERATED_ROOT / "manifests"
LABELS_ROOT = GENERATED_ROOT / "labels"
