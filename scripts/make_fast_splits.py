"""Build small deterministic Phase 3 train files for constrained local runs.

The production train split is intentionally large because it includes gold,
synthetic, and silver documents. This helper keeps all gold documents and
adds bounded samples from augmentation sources so laptop runs can finish while
still exercising every task.

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config


GOLD_CORPORA = {"BioRED", "BC5CDR"}


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, docs: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n


def by_corpus(docs: Iterable[Dict]) -> Counter:
    return Counter(str(d.get("corpus") or "") for d in docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.PHASE2_SPLITS_DIR / "train.jsonl"))
    parser.add_argument("--output-dir", default=str(config.PHASE2_SPLITS_DIR))
    parser.add_argument("--silver", type=int, default=1500)
    parser.add_argument("--synthetic", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs = read_jsonl(Path(args.input))

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for doc in docs:
        grouped[str(doc.get("corpus") or "")].append(doc)

    gold = []
    for corpus in sorted(GOLD_CORPORA):
        gold.extend(grouped.get(corpus, []))

    silver = grouped.get("PubTator3_silver", [])[:]
    synthetic = grouped.get("SNOMED_synthetic", [])[:]
    rng.shuffle(silver)
    rng.shuffle(synthetic)

    fast = gold + silver[: max(args.silver, 0)] + synthetic[: max(args.synthetic, 0)]

    output_dir = Path(args.output_dir)
    gold_path = output_dir / "train_gold.jsonl"
    fast_path = output_dir / "train_fast.jsonl"
    summary_path = output_dir / "fast_split_summary.json"

    write_jsonl(gold_path, gold)
    write_jsonl(fast_path, fast)

    summary = {
        "source": str(Path(args.input)),
        "seed": args.seed,
        "requested": {"silver": args.silver, "synthetic": args.synthetic},
        "source_counts": dict(by_corpus(docs)),
        "gold_counts": dict(by_corpus(gold)),
        "fast_counts": dict(by_corpus(fast)),
        "outputs": {
            "gold": str(gold_path),
            "fast": str(fast_path),
            "summary": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
