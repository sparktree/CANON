"""Apply unified relation labels to mapped documents (CANON Phase 2.3).

Reads the Phase 2.2 output, looks each relation's
``(corpus, source_relation_type, subject_semantic_class, object_semantic_class)``
tuple up in ``relation_schema.RELATION_MAPPINGS``, and stamps:

    target_relation        -- argmax target label (canonical training label)
    tier                   -- 1 (SNOMED-native) or 2 (empirical)
    target_probability     -- probability assigned to that argmax candidate
    extra["target_candidates"]
                           -- full list of candidates: [{target_relation, tier,
                              probability}, ...] -- consumed by Phase 3 soft-
                              label cross-entropy when temperature > 0.

Both representations land in the same record so Phase 3 can choose hard
(argmax) or soft (full distribution) labels per training stage.

Default policy
--------------
A relation whose ``(corpus, source, subj_class, obj_class)`` tuple is not
present in ``RELATION_MAPPINGS`` is stamped with:

    target_relation     = "associated-with"
    tier                = 2
    target_probability  = 1.0

with ``extra["default_used"] = True`` so Phase 4.6 error analysis can attribute
losses to defaulted rows. Reasons this fires:
  * subject or object has ``semantic_class is None`` (e.g. unregistered
    entity_type — should be zero given Phase 2.1 verified the registry).
  * the pair is rare and was not enumerated in Phase 1.4
    (e.g. species-disease, cell_line-cell_line).

The total number of defaulted rows is reported in
``outputs/phase2/relation_mapping_summary.json``; if it grows beyond a
small fraction the registry should be extended.

Outputs (under outputs/phase2/relation_mapped/):
    <Corpus>/train.jsonl, dev.jsonl, test.jsonl  -- fully stamped documents
    relation_mapping_summary.json                -- per-corpus + per-relation counts
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

try:
    from config import REPO_ROOT
    import relation_schema
    from unified_format import Document, Relation, read_jsonl, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT
    import relation_schema
    from unified_format import Document, Relation, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIR    = REPO_ROOT / "outputs" / "phase2" / "mapped"
OUTPUT_DIR   = REPO_ROOT / "outputs" / "phase2" / "relation_mapped"
SUMMARY_JSON = REPO_ROOT / "outputs" / "phase2" / "relation_mapping_summary.json"

DEFAULT_TARGET = "associated-with"
DEFAULT_TIER = 2
DEFAULT_PROBABILITY = 1.0


# ---------------------------------------------------------------------------
# Lookup index over relation_schema (avoids O(n) scan per relation)
# ---------------------------------------------------------------------------

def _build_index() -> Dict[Tuple[str, str, str, str], List[relation_schema.RelationMapping]]:
    idx: Dict[Tuple[str, str, str, str], List[relation_schema.RelationMapping]] = {}
    for m in relation_schema.iter_rows():
        key = (m.source_corpus, m.source_relation_type,
               m.subject_semantic_class, m.object_semantic_class)
        idx.setdefault(key, []).append(m)
    return idx


_INDEX: Dict[Tuple[str, str, str, str], List[relation_schema.RelationMapping]] = _build_index()


# ---------------------------------------------------------------------------
# Per-relation stamp
# ---------------------------------------------------------------------------

def _candidate_dicts(mappings: List[relation_schema.RelationMapping]) -> List[dict]:
    return [
        {"target_relation": m.target_relation,
         "tier": m.tier,
         "probability": m.probability}
        for m in mappings
    ]


def stamp_relation(rel: Relation, doc: Document, defaulted_counter: Counter) -> Relation:
    n = len(doc.entities)
    if rel.subject_idx < 0 or rel.subject_idx >= n or rel.object_idx < 0 or rel.object_idx >= n:
        return rel  # malformed; corpus_convert already drops dangling rows, this is defensive

    subj = doc.entities[rel.subject_idx]
    obj = doc.entities[rel.object_idx]
    subj_class = subj.semantic_class or "unknown"
    obj_class = obj.semantic_class or "unknown"

    key = (doc.corpus, rel.source_relation_type, subj_class, obj_class)
    mappings = _INDEX.get(key, [])

    if not mappings:
        defaulted_counter[key] += 1
        return Relation(
            subject_idx=rel.subject_idx,
            object_idx=rel.object_idx,
            source_relation_type=rel.source_relation_type,
            target_relation=DEFAULT_TARGET,
            tier=DEFAULT_TIER,
            target_probability=DEFAULT_PROBABILITY,
            novelty=rel.novelty,
            extra={
                **rel.extra,
                "subject_class": subj_class,
                "object_class": obj_class,
                "default_used": True,
                "target_candidates": [{
                    "target_relation": DEFAULT_TARGET,
                    "tier": DEFAULT_TIER,
                    "probability": DEFAULT_PROBABILITY,
                }],
            },
        )

    best = max(mappings, key=lambda m: m.probability)
    return Relation(
        subject_idx=rel.subject_idx,
        object_idx=rel.object_idx,
        source_relation_type=rel.source_relation_type,
        target_relation=best.target_relation,
        tier=best.tier,
        target_probability=best.probability,
        novelty=rel.novelty,
        extra={
            **rel.extra,
            "subject_class": subj_class,
            "object_class": obj_class,
            "target_candidates": _candidate_dicts(mappings),
        },
    )


def stamp_document(doc: Document, defaulted_counter: Counter) -> Document:
    return Document(
        pmid=doc.pmid,
        corpus=doc.corpus,
        split=doc.split,
        title=doc.title,
        abstract=doc.abstract,
        text=doc.text,
        entities=doc.entities,
        relations=[stamp_relation(r, doc, defaulted_counter) for r in doc.relations],
        schema_version=doc.schema_version,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def apply_all(verbose: bool = True) -> dict:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"{INPUT_DIR} not found; run Phase 2.2 (concept_map.py) first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    defaulted: Counter = Counter()
    target_dist: Counter = Counter()
    tier_dist: Counter = Counter()
    multi_candidate_count = 0
    total_relations = 0

    summary: dict = {
        "policy": {
            "default_target": DEFAULT_TARGET,
            "default_tier": DEFAULT_TIER,
            "default_probability": DEFAULT_PROBABILITY,
        },
        "tier1_relations": sorted(relation_schema.TIER1_RELATIONS),
        "tier2_relations": sorted(relation_schema.TIER2_RELATIONS),
        "corpora": {},
    }

    for corpus_dir in sorted(INPUT_DIR.iterdir()):
        if not corpus_dir.is_dir():
            continue
        corpus = corpus_dir.name
        per_split: dict = {}

        for jsonl_path in sorted(corpus_dir.glob("*.jsonl")):
            split = jsonl_path.stem
            out_path = OUTPUT_DIR / corpus / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            n_docs = 0
            n_rel = 0
            n_tier1 = 0
            n_tier2 = 0
            n_default = 0
            n_multi = 0
            split_target: Counter = Counter()

            def _stamped_docs() -> Iterator[Document]:
                nonlocal n_docs
                for doc in read_jsonl(jsonl_path):
                    n_docs += 1
                    yield stamp_document(doc, defaulted)

            written = write_jsonl(_stamped_docs(), out_path)

            # Second pass for accounting; cheap.
            for doc in read_jsonl(out_path):
                for r in doc.relations:
                    n_rel += 1
                    if r.tier == 1:
                        n_tier1 += 1
                    elif r.tier == 2:
                        n_tier2 += 1
                    if r.extra.get("default_used"):
                        n_default += 1
                    if len(r.extra.get("target_candidates", [])) > 1:
                        n_multi += 1
                    if r.target_relation:
                        split_target[r.target_relation] += 1

            total_relations += n_rel
            multi_candidate_count += n_multi
            for k, v in split_target.items():
                target_dist[k] += v
            tier_dist[1] += n_tier1
            tier_dist[2] += n_tier2

            per_split[split] = {
                "documents": written,
                "relations": n_rel,
                "tier1": n_tier1,
                "tier2": n_tier2,
                "defaulted": n_default,
                "multi_candidate": n_multi,
                "target_distribution": dict(split_target.most_common()),
                "output": str(out_path),
            }
            if verbose:
                print(
                    f"[2.3] {corpus:<14s} {split:<5s} -> "
                    f"{written:>5,d} docs  {n_rel:>5,d} rels  "
                    f"tier1={n_tier1:,}  tier2={n_tier2:,}  "
                    f"multi={n_multi:,}  defaulted={n_default:,}"
                )

        summary["corpora"][corpus] = per_split

    summary["aggregate"] = {
        "total_relations": total_relations,
        "tier1": tier_dist[1],
        "tier2": tier_dist[2],
        "multi_candidate": multi_candidate_count,
        "target_distribution": dict(target_dist.most_common()),
    }
    summary["defaulted_pairs"] = sorted(
        [
            {
                "corpus": k[0],
                "source_relation": k[1],
                "subject_class": k[2],
                "object_class": k[3],
                "count": v,
            }
            for k, v in defaulted.items()
        ],
        key=lambda r: -r["count"],
    )

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[2.3] summary -> {SUMMARY_JSON}")
        if summary["defaulted_pairs"]:
            print("[2.3] defaulted (corpus, source, subj, obj) tuples:")
            for r in summary["defaulted_pairs"][:10]:
                print(
                    f"    {r['corpus']:<8s} {r['source_relation']:<22s} "
                    f"{r['subject_class']:<10s} -> {r['object_class']:<10s}  "
                    f"count={r['count']}"
                )
            extra = len(summary["defaulted_pairs"]) - 10
            if extra > 0:
                print(f"    ... and {extra} more (see summary JSON)")

    return summary


if __name__ == "__main__":
    apply_all(verbose=True)
