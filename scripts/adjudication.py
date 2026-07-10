"""Create and resolve the dual-review Tier-1 adjudication artifact."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


LABELS = ("tier1-correct", "tier2-correct", "both-plausible", "neither", "insufficient-context")


def sample(predictions: Path, output: Path, size: int = 200, seed: int = 42) -> int:
    groups = defaultdict(list)
    with predictions.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            for pair in row.get("csp", {}).get("pairs", []):
                if pair.get("event") == "tier1-escalation":
                    groups[pair.get("relation", "unknown")].append({
                        "corpus": row.get("corpus"), "pmid": row.get("pmid"),
                        "pair": pair, "reviewer_1": None, "reviewer_2": None,
                        "consensus": None, "notes": "",
                    })
    rng = random.Random(seed)
    selected = []
    while len(selected) < size and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < size:
                idx = rng.randrange(len(groups[key]))
                selected.append(groups[key].pop(idx))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(selected, 1):
            row["adjudication_id"] = f"T1-{idx:04d}"
            row["allowed_labels"] = LABELS
            fh.write(json.dumps(row) + "\n")
    return len(selected)


def consensus(input_path: Path, output_path: Path) -> dict:
    counts = defaultdict(int)
    unresolved = 0
    rows = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            a, b = row.get("reviewer_1"), row.get("reviewer_2")
            if a and a == b:
                row["consensus"] = a
                counts[a] += 1
            else:
                unresolved += 1
            rows.append(row)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return {"records": len(rows), "resolved": len(rows) - unresolved,
            "unresolved": unresolved, "labels": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sample", "consensus"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    if args.mode == "sample":
        print({"sampled": sample(Path(args.input), Path(args.output), args.size)})
    else:
        print(consensus(Path(args.input), Path(args.output)))


if __name__ == "__main__":
    main()
