from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKIN_TONE_DIR = REPO_ROOT / "benchmarks" / "skin_tone"

for path in (REPO_ROOT, SKIN_TONE_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
