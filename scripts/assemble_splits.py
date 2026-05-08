"""Train/Dev/Test split assembly for CANON Phase 2.7.

Concatenates the per-corpus unified-format JSONL outputs from Phases 2.3, 2.5,
and 2.6 into three files:

    outputs/phase2/splits/train.jsonl
    outputs/phase2/splits/dev.jsonl
    outputs/phase2/splits/test.jsonl
    outputs/phase2/splits/split_summary.json

Composition rules (per the project plan, Phase 2.7):

    * train -- BioRED train + BC5CDR train + SNOMED synthetic + PubTator3 silver
    * dev   -- BioRED dev   + BC5CDR dev   (gold only)
    * test  -- BioRED test  + BC5CDR test  (gold only)

Augmentation data (synthetic, silver, NCBI/NLM-Chem if present) is train-only by
construction. dev/test additionally enforce a verified-mapping filter: any
document containing a SNOMED-mappable entity (semantic_class in {chemical,
disease}, non_snomed=False, real original_code) whose mapping is not active in
the loaded SNOMED release (mapped_snomed_id missing or snomed_active != True)
is dropped from dev/test. The filter never runs on train.

Streams source files via unified_format.read_jsonl/write_jsonl, so the largest
silver corpus (~94 MB) is never resident in memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

try:
    from config import REPO_ROOT, relative_to_repo
    from unified_format import Document, read_jsonl, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, relative_to_repo
    from unified_format import Document, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PHASE2_DIR          = REPO_ROOT / "outputs" / "phase2"
RELATION_MAPPED_DIR = PHASE2_DIR / "relation_mapped"
SYNTHETIC_TRAIN     = PHASE2_DIR / "synthetic" / "train.jsonl"
SILVER_TRAIN        = PHASE2_DIR / "silver" / "PubTator3" / "train.jsonl"

OUT_DIR             = PHASE2_DIR / "splits"
TRAIN_OUT           = OUT_DIR / "train.jsonl"
DEV_OUT             = OUT_DIR / "dev.jsonl"
TEST_OUT            = OUT_DIR / "test.jsonl"
SUMMARY_OUT         = OUT_DIR / "split_summary.json"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

GOLD_CORPORA           = ("BioRED", "BC5CDR")
SNOMED_MAPPABLE_CLASSES = frozenset({"chemical", "disease"})

# Codes that mean "annotator could not assign a concept" rather than a real
# ontology ID. BC5CDR uses "-1" for composite/unmappable mentions, BioRED uses
# "-", PubTator-style readers leave "" for missing fields. None of these
# represents a SNOMED-mappable target, so they bypass the dev/test
# verified-mapping check (no mapping was ever attempted).
_SENTINEL_CODES = frozenset({"", "-", "-1"})


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def _doc_dev_test_eligible(doc: Document) -> Tuple[bool, Optional[str]]:
    """Return (keep, drop_reason).

    A dev/test document is dropped if any chemical/disease entity carries a
    real MeSH code but its SNOMED mapping is not verified active. Gene /
    variant / species / cell-line entities are non_snomed=True and skipped --
    they are NER-only by Phase 1.3 design and do not need SNOMED verification.
    """
    for em in doc.entities:
        if em.non_snomed:
            continue
        if em.semantic_class not in SNOMED_MAPPABLE_CLASSES:
            continue
        code = em.original_code
        if not code or code in _SENTINEL_CODES:
            continue
        if em.mapped_snomed_id is None or em.snomed_active is not True:
            return False, f"unverified_entity:{em.semantic_class}:{code}"
    return True, None


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _new_audit() -> dict:
    return {
        "missing_sources": [],
        "per_corpus_seen": {},
        "per_corpus_kept": {},
        "dropped_dev_test": {},
        "drop_reason_counts": {},
    }


def _stream_split(
    sources: List[Path],
    split_name: str,
    filter_unverified: bool,
    audit: dict,
    verbose: bool,
) -> Iterator[Document]:
    for src in sources:
        if not src.exists():
            audit["missing_sources"].append(relative_to_repo(src))
            if verbose:
                print(f"[2.7]   {split_name}: missing source {relative_to_repo(src)}",
                      flush=True)
            continue
        if verbose:
            print(f"[2.7]   {split_name}: streaming {relative_to_repo(src)}", flush=True)
        for doc in read_jsonl(src):
            audit["per_corpus_seen"][doc.corpus] = (
                audit["per_corpus_seen"].get(doc.corpus, 0) + 1
            )
            if filter_unverified:
                if doc.corpus not in GOLD_CORPORA:
                    raise AssertionError(
                        f"non-gold corpus {doc.corpus!r} reached {split_name} "
                        f"input list -- augmentation must be train-only"
                    )
                ok, reason = _doc_dev_test_eligible(doc)
                if not ok:
                    audit["dropped_dev_test"][doc.corpus] = (
                        audit["dropped_dev_test"].get(doc.corpus, 0) + 1
                    )
                    rkey = reason or "unknown"
                    audit["drop_reason_counts"][rkey] = (
                        audit["drop_reason_counts"].get(rkey, 0) + 1
                    )
                    continue
            audit["per_corpus_kept"][doc.corpus] = (
                audit["per_corpus_kept"].get(doc.corpus, 0) + 1
            )
            yield doc


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def assemble(verbose: bool = True) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_sources = [
        RELATION_MAPPED_DIR / "BioRED" / "train.jsonl",
        RELATION_MAPPED_DIR / "BC5CDR" / "train.jsonl",
        SYNTHETIC_TRAIN,
        SILVER_TRAIN,
    ]
    dev_sources = [
        RELATION_MAPPED_DIR / "BioRED" / "dev.jsonl",
        RELATION_MAPPED_DIR / "BC5CDR" / "dev.jsonl",
    ]
    test_sources = [
        RELATION_MAPPED_DIR / "BioRED" / "test.jsonl",
        RELATION_MAPPED_DIR / "BC5CDR" / "test.jsonl",
    ]

    audit = _new_audit()

    if verbose:
        print(f"[2.7] writing {relative_to_repo(TRAIN_OUT)} ...", flush=True)
    n_train = write_jsonl(
        _stream_split(train_sources, "train", filter_unverified=False,
                      audit=audit, verbose=verbose),
        TRAIN_OUT,
    )

    if verbose:
        print(f"[2.7] writing {relative_to_repo(DEV_OUT)} ...", flush=True)
    n_dev = write_jsonl(
        _stream_split(dev_sources, "dev", filter_unverified=True,
                      audit=audit, verbose=verbose),
        DEV_OUT,
    )

    if verbose:
        print(f"[2.7] writing {relative_to_repo(TEST_OUT)} ...", flush=True)
    n_test = write_jsonl(
        _stream_split(test_sources, "test", filter_unverified=True,
                      audit=audit, verbose=verbose),
        TEST_OUT,
    )

    summary = {
        "policy": {
            "gold_corpora": list(GOLD_CORPORA),
            "augmentation_in_train_only": True,
            "dev_test_filter": "drop_doc_if_any_snomed_mappable_entity_unverified",
            "snomed_mappable_classes": sorted(SNOMED_MAPPABLE_CLASSES),
            "silver_source": relative_to_repo(SILVER_TRAIN),
        },
        "documents_written": {"train": n_train, "dev": n_dev, "test": n_test},
        "expected_gold_sizes": {
            "BioRED": [400, 100, 100],
            "BC5CDR": [500, 500, 500],
        },
        "audit": audit,
        "outputs": {
            "train": relative_to_repo(TRAIN_OUT),
            "dev":   relative_to_repo(DEV_OUT),
            "test":  relative_to_repo(TEST_OUT),
        },
    }

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"[2.7] train={n_train:,}  dev={n_dev:,}  test={n_test:,}", flush=True)
        for corpus, dropped in audit["dropped_dev_test"].items():
            print(f"[2.7]   dropped {dropped:,} {corpus} dev/test docs (unverified entity)",
                  flush=True)
        print(f"[2.7] summary -> {relative_to_repo(SUMMARY_OUT)}", flush=True)

    return summary


if __name__ == "__main__":
    assemble(verbose=True)
