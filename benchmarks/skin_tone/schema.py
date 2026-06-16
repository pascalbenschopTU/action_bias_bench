from __future__ import annotations

import hashlib
from typing import Iterable


DARK_VARIANT_ORDER = ("african", "indian")
LIGHT_VARIANT_ORDER = ("white", "asian")
DARK_VARIANTS = frozenset(DARK_VARIANT_ORDER)
LIGHT_VARIANTS = frozenset(LIGHT_VARIANT_ORDER)
VARIANT_ORDER = ("african", "asian", "indian", "white")
VARIANT_SWAP = {
    "african": "white",
    "white": "african",
    "indian": "asian",
    "asian": "indian",
}

MATCHED_SEEN_SPLIT = "eval_matched_seen_ids"
MATCHED_UNSEEN_SPLIT = "eval_matched_unseen_ids"
SHIFTED_SEEN_SPLIT = "eval_shifted_seen_ids"
SHIFTED_UNSEEN_SPLIT = "eval_shifted_unseen_ids"
EVAL_SPLITS = (
    MATCHED_UNSEEN_SPLIT,
    MATCHED_SEEN_SPLIT,
    SHIFTED_SEEN_SPLIT,
    SHIFTED_UNSEEN_SPLIT,
)
SPLIT_FAMILY_TO_SPLITS = {
    "seen": (MATCHED_SEEN_SPLIT, SHIFTED_SEEN_SPLIT),
    "unseen": (MATCHED_UNSEEN_SPLIT, SHIFTED_UNSEEN_SPLIT),
}

SWAP_ORDER = (
    ("african", "white"),
    ("indian", "asian"),
    ("asian", "indian"),
    ("white", "african"),
)
SWAP_LABELS = (
    "african\n→ white",
    "indian\n→ asian",
    "asian\n→ indian",
    "white\n→ african",
)

RGB_MODEL_COLORS = {
    "mc3_18": "#1f77b4",
    "mvit_v2_s": "#ff7f0e",
    "r2plus1d_18": "#2ca02c",
    "r3d_18": "#d62728",
    "s3d": "#9467bd",
    "swin3d_s": "#8c564b",
}


def tone_group_for_variant(variant: str) -> str:
    value = str(variant).lower()
    if value in DARK_VARIANTS:
        return "dark"
    if value in LIGHT_VARIANTS:
        return "light"
    if value == "initial":
        return "initial"
    return "unknown"


def stable_seed(parts: Iterable[object] | object, *, modulo: int = 2**32) -> int:
    if isinstance(parts, (str, bytes)):
        payload = parts
    else:
        try:
            payload = "\x1f".join(str(part) for part in parts)  # type: ignore[arg-type]
        except TypeError:
            payload = str(parts)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)
