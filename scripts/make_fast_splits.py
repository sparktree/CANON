"""Reduced training-split assembly for the local-CPU shrunk-plan run.

Reads the existing Phase 2.7 train.jsonl (which already concatenates BioRED
gold + BC5CDR gold + SNOMED synthetic + PubTator3 silver) and emits two
deterministic subsets:

    train_gold.jsonl  -- BioRED + BC5CDR only (no augmentation).
    train_fast.jsonl  -- gold + capped synthetic + capped silver.

Both files preserve the unified-format JSONL schema. dev/test splits are
unchanged because they are gold-only by Phase 2.7's verified-mapping filter.

CLI
---
    python scripts/make_fast_splits.py
    python scripts/make_fast_splits.py --silver 1500 --synthetic 3000

Outputs:
    outputs/phase2/splits/train_gold.jsonl
    outputs/phase2/splits/train_fast.jsonl
    outputs/phase2/splits/fast_split_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional

try:
    from config import REPO_ROOT, relative_to_repo
    from unified_format import Document, read_jsonl, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, relative_to_repo
    from unified_format import Document, read_jsonl, write_jsonl


SPLITS_DIR  = REPO_ROOT / "outputs" / "phase2" / "splits"
TRAIN_IN    = SPLITS_DIR / "train.jsonl"
GOLD_OUT    = SPLITS_DIR / "train_gold.jsonl"
FAST_OUT    = SPLITS_DIR / "train_fast.jsonl"
SUMMARY_OUT = SPLITS_DIR / "fast_split_summary.json"

GOLD_CORPORA       = ("BioRED", "BC5CDR")
SYNTHETIC_CORPUS   = "SNOMED_synthetic"
SILVER_CORPUS      = "PubTator3_silver"


def _stream_filter_caps(
    src: Path,
    gold_corpora: tuple[str, ...],
    silver_cap: Optional[int],
    synthetic_cap: Optional[int],
    seed: int,
) -> tuple[List[Document], Counter, Counter]:
    """Return the gold list and capped-augmentation list.

    The caps use reservoir sampling (one pass, deterministic seed) so the
    output is stable across re-runs without holding the full silver/synthetic
    corpora in memory.
    """
    rng = random.Random(seed)
    gold: List[Document] = []
    silver_reservoir: List[Document] = []
    synth_reservoir: List[Document] = []
    seen = Counter()
    seen_after_caps = Counter()

    silver_seen = 0
    synth_seen = 0

    for doc in read_jsonl(src):
        seen[doc.corpus] += 1
        if doc.corpus in gold_corpora:
            gold.append(doc)
            seen_after_caps[doc.corpus] += 1
        elif doc.corpus == SILVER_CORPUS:
            silver_seen += 1
            if silver_cap is None or silver_cap <= 0:
                continue
            if len(silver_reservoir) < silver_cap:
                silver_reservoir.append(doc)
            else:
                # Vitter's reservoir sampling: replace with prob k / n
                j = rng.randint(0, silver_seen - 1)
                if j < silver_cap:
                    silver_reservoir[j] = doc
        elif doc.corpus == SYNTHETIC_CORPUS:
            synth_seen += 1
            if synthetic_cap is None or synthetic_cap <= 0:
                continue
            if len(synth_reservoir) < synthetic_cap:
                synth_reservoir.append(doc)
            else:
                j = rng.randint(0, synth_seen - 1)
                if j < synthetic_cap:
                    synth_reservoir[j] = doc

    seen_after_caps[SILVER_CORPUS] = len(silver_reservoir)
    seen_after_caps[SYNTHETIC_CORPUS] = len(synth_reservoir)
    augmentation = silver_reservoir + synth_reservoir
    return gold, augmentation, seen, seen_after_caps  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--silver", type=int, default=1500,
                   help="Number of PubTator3 silver docs to keep in train_fast (default 1500).")
    p.add_argument("--synthetic", type=int, default=3000,
                   help="Number of SNOMED synthetic docs to keep in train_fast (default 3000).")
    p.add_argument("--seed", type=int, default=42,
                   help="Reservoir-sampling seed (default 42).")
    p.add_argument("--input", default=str(TRAIN_IN),
                   help="Source train.jsonl (default outputs/phase2/splits/train.jsonl).")
    p.add_argument("--out-gold", default=str(GOLD_OUT))
    p.add_argument("--out-fast", default=str(FAST_OUT))
    p.add_argument("--out-summary", default=str(SUMMARY_OUT))
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run Phase 2.7 (assemble_splits) first.")

    print(f"[fast-splits] streaming {relative_to_repo(src)} ...", flush=True)
    gold, aug, seen, seen_after = _stream_filter_caps(
        src,
        gold_corpora=GOLD_CORPORA,
        silver_cap=args.silver,
        synthetic_cap=args.synthetic,
        seed=args.seed,
    )

    n_gold = write_jsonl(iter(gold), Path(args.out_gold))

    # Mix the gold + augmentation deterministically so the train_fast file is
    # not corpus-segmented (gold-then-aug ordering can bias minibatch dynamics).
    rng = random.Random(args.seed)
    fast = list(gold) + list(aug)
    rng.shuffle(fast)
    n_fast = write_jsonl(iter(fast), Path(args.out_fast))

    summary = {
        "policy": {
            "gold_corpora": list(GOLD_CORPORA),
            "silver_cap": args.silver,
            "synthetic_cap": args.synthetic,
            "seed": args.seed,
        },
        "input": relative_to_repo(src),
        "input_seen": dict(seen),
        "kept_after_caps": dict(seen_after),
        "outputs": {
            "train_gold": relative_to_repo(Path(args.out_gold)),
            "train_fast": relative_to_repo(Path(args.out_fast)),
            "summary":    relative_to_repo(Path(args.out_summary)),
        },
        "documents_written": {
            "train_gold": n_gold,
            "train_fast": n_fast,
        },
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[fast-splits] gold      -> {n_gold:,} docs ({relative_to_repo(Path(args.out_gold))})")
    print(f"[fast-splits] fast      -> {n_fast:,} docs ({relative_to_repo(Path(args.out_fast))})")
    print(f"[fast-splits] summary   -> {relative_to_repo(Path(args.out_summary))}")
    print(f"[fast-splits] kept by corpus: {dict(seen_after)}")


if __name__ == "__main__":
    main()
