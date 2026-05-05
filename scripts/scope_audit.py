"""Phase 1.3 entity-scope audit.

Walks BioRED, BC5CDR, NCBI Disease, and NLM-Chem PubTator files and counts
mention frequencies per (corpus, entity_type). Each row is tagged with its
SNOMED-scope status from entity_scope.ENTITY_TYPES so you can verify the
in/out-of-scope split matches the Phase 1.3 scoping decision.

Outputs (under CANON/outputs/phase1/):
    entity_scope_audit.csv  - per (corpus, entity_type) mention counts +
                              unique-id counts + scope flag.
    entity_scope_summary.csv - per (corpus, scope) totals.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    from config import BIORED_FILES, CDR_FILES, DATA_ROOT, REPO_ROOT
    from utils import parse_pubtator
    import entity_scope
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import BIORED_FILES, CDR_FILES, DATA_ROOT, REPO_ROOT
    from utils import parse_pubtator
    import entity_scope


OUTPUT_DIR = REPO_ROOT / "outputs" / "phase1"


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def _corpus_files() -> Dict[str, list[Path]]:
    return {
        "BioRED": _existing(BIORED_FILES.values()),
        "BC5CDR": _existing(CDR_FILES.values()),
        "NCBI_Disease": _existing(
            [
                DATA_ROOT / "NCBI_Disease" / "NCBItrainset_corpus.txt",
                DATA_ROOT / "NCBI_Disease" / "NCBIdevelopset_corpus.txt",
                DATA_ROOT / "NCBI_Disease" / "NCBItestset_corpus.txt",
            ]
        ),
        "NLM-Chem": _existing(
            [
                DATA_ROOT / "NLM-Chem" / "BC7T2-NLMChem-corpus-train.PubTator",
                DATA_ROOT / "NLM-Chem" / "BC7T2-NLMChem-corpus-dev.PubTator",
                DATA_ROOT / "NLM-Chem" / "BC7T2-NLMChem-corpus-test.PubTator",
            ]
        ),
    }


def audit():
    counts: Dict[Tuple[str, str], int] = Counter()
    unique_ids: Dict[Tuple[str, str], set] = defaultdict(set)

    for corpus, files in _corpus_files().items():
        if not files:
            continue
        for path in files:
            for doc in parse_pubtator(path):
                for ent in doc["entities"]:
                    key = (corpus, ent["entity_type"])
                    counts[key] += 1
                    raw = (ent.get("identifier_raw") or "").strip()
                    if raw and raw not in {"-", "-1"}:
                        unique_ids[key].add(raw)

    rows = []
    for (corpus, entity_type), mentions in counts.items():
        spec = entity_scope.lookup(corpus, entity_type)
        rows.append(
            {
                "corpus": corpus,
                "entity_type": entity_type,
                "vocabulary": spec.vocabulary if spec else "UNKNOWN",
                "semantic_class": spec.semantic_class if spec else "unknown",
                "scope": (
                    "snomed_normalized" if (spec and spec.snomed_normalized)
                    else ("ner_only" if spec else "unregistered")
                ),
                "rationale": spec.rationale if spec else "Entity type not present in entity_scope registry",
                "mentions": mentions,
                "unique_ids": len(unique_ids[(corpus, entity_type)]),
            }
        )
    rows.sort(key=lambda r: (r["corpus"], -r["mentions"]))
    return rows


def summarize(rows):
    totals = defaultdict(lambda: {"mentions": 0, "unique_ids": 0})
    for r in rows:
        key = (r["corpus"], r["scope"])
        totals[key]["mentions"] += r["mentions"]
        totals[key]["unique_ids"] += r["unique_ids"]
    out = []
    for (corpus, scope), v in sorted(totals.items()):
        out.append(
            {
                "corpus": corpus,
                "scope": scope,
                "mentions": v["mentions"],
                "unique_ids": v["unique_ids"],
            }
        )
    return out


def write_csv(rows, path: Path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(verbose: bool = True):
    rows = audit()
    summary = summarize(rows)

    audit_path = OUTPUT_DIR / "entity_scope_audit.csv"
    summary_path = OUTPUT_DIR / "entity_scope_summary.csv"
    write_csv(
        rows,
        audit_path,
        ["corpus", "entity_type", "vocabulary", "semantic_class", "scope", "mentions", "unique_ids", "rationale"],
    )
    write_csv(summary, summary_path, ["corpus", "scope", "mentions", "unique_ids"])

    if verbose:
        print("[scope_audit] per-corpus scope summary:")
        for r in summary:
            print(f"    {r['corpus']:<14s} {r['scope']:<18s} mentions={r['mentions']:>7,d}  unique_ids={r['unique_ids']:>6,d}")
        unregistered = [r for r in rows if r["scope"] == "unregistered"]
        if unregistered:
            print(f"[scope_audit] WARNING: {len(unregistered)} unregistered (corpus, entity_type) pairs:")
            for r in unregistered:
                print(f"    {r['corpus']} / {r['entity_type']}  ({r['mentions']} mentions)")
        print(f"[scope_audit] wrote {audit_path}")
        print(f"[scope_audit] wrote {summary_path}")
    return {"audit": rows, "summary": summary}


if __name__ == "__main__":
    run()
