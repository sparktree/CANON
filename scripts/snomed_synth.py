"""Generate synthetic Tier-1 training triples from SNOMED stated relationships (CANON Phase 2.5).

Scans sct2_Relationship_Snapshot (inferred) for active triples with SNOMED
attributes causative-agent (246075003), due-to (42752001), and after (255234002).
Note: sct2_StatedRelationship_Snapshot in the US edition contains only retired
US-extension rows (all active=0); the inferred relationships file is used instead.
Each triple is scope-checked using precomputed descendant frozensets from Phase 1.6,
then emitted as a unified-format Document into outputs/phase2/synthetic/train.jsonl.

These documents are the only source of Tier-1 training gradient for the
relation-extraction head; all 66 gold corpus schema groups have Tier-2 argmax
labels (confirmed by Phase 1.4 audit).

Scope rules:
    causative-agent:  subject ∈ ClinicalFinding(404684003),
                      object  ∈ Substance(105590001) OR PharmaBioProduct(373873005)
    due-to / after:   subject ∈ ClinicalFinding(404684003),
                      object  ∈ ClinicalFinding OR Event(272379006) OR Procedure(71388002)

Raises RuntimeError if causative-agent count < 3,000 (plan hard requirement).

Outputs:
    outputs/phase2/synthetic/train.jsonl          -- unified-format Documents
    outputs/phase2/synthetic/generation_summary.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, List, Set, Tuple

try:
    from config import REPO_ROOT, SNOMED_FILES, relative_to_repo
    import mrcm
    import snomed_hierarchy
    from unified_format import Document, EntityMention, Relation, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, SNOMED_FILES, relative_to_repo
    import mrcm
    import snomed_hierarchy
    from unified_format import Document, EntityMention, Relation, write_jsonl


OUTPUT_DIR  = REPO_ROOT / "outputs" / "phase2" / "synthetic"
SYNTH_JSONL = OUTPUT_DIR / "train.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "generation_summary.json"

# SNOMED attribute concept IDs targeted for Tier-1 synthesis
_ATTR_CAUSATIVE_AGENT = "246075003"
_ATTR_DUE_TO          = "42752001"
_ATTR_AFTER           = "255234002"

TARGET_ATTR_IDS: Set[str] = {_ATTR_CAUSATIVE_AGENT, _ATTR_DUE_TO, _ATTR_AFTER}

_ATTR_TO_LABEL: Dict[str, str] = {
    _ATTR_CAUSATIVE_AGENT: "causative-agent",
    _ATTR_DUE_TO:          "due-to",
    _ATTR_AFTER:           "after",
}

# SNOMED scope anchor concept IDs (precomputed descendant sets keyed by these)
_CLINICAL_FINDING = "404684003"
_SUBSTANCE        = "105590001"
_PHARMA_PRODUCT   = "373873005"
_EVENT            = "272379006"
_PROCEDURE        = "71388002"

SYNTHETIC_CONFIDENCE = 0.65
MIN_CAUSATIVE_AGENT  = 3000

_FSN_SUFFIX = re.compile(r"\s*\([^)]+\)\s*$")


def _strip_fsn(term: str) -> str:
    return _FSN_SUFFIX.sub("", term).strip()


def _iter_stated_rels(path: Path) -> Iterator[Tuple[str, str, str]]:
    """Yield (sourceId, destinationId, typeId) for active rows with relevant typeId."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if row.get("active") != "1":
                continue
            type_id = row.get("typeId", "")
            if type_id not in TARGET_ATTR_IDS:
                continue
            yield row["sourceId"], row["destinationId"], type_id


def _check_scope(
    src: str,
    dst: str,
    type_id: str,
    ancestors: Dict[str, FrozenSet[str]],
) -> bool:
    cf_set  = ancestors.get(f"descendant:{_CLINICAL_FINDING}", frozenset())
    sub_set = ancestors.get(f"descendant:{_SUBSTANCE}",        frozenset())
    ph_set  = ancestors.get(f"descendant:{_PHARMA_PRODUCT}",   frozenset())
    ev_set  = ancestors.get(f"descendant:{_EVENT}",            frozenset())
    pr_set  = ancestors.get(f"descendant:{_PROCEDURE}",        frozenset())

    if src not in cf_set:
        return False
    if type_id == _ATTR_CAUSATIVE_AGENT:
        return dst in sub_set or dst in ph_set
    else:  # due-to or after
        return dst in cf_set or dst in ev_set or dst in pr_set


def _obj_semantic(
    dst: str,
    ancestors: Dict[str, FrozenSet[str]],
) -> Tuple[str, str]:
    """Return (entity_type, semantic_class) for the object concept."""
    sub_set = ancestors.get(f"descendant:{_SUBSTANCE}",      frozenset())
    ph_set  = ancestors.get(f"descendant:{_PHARMA_PRODUCT}", frozenset())
    if dst in sub_set or dst in ph_set:
        return "ChemicalEntity", "chemical"
    return "DiseaseOrPhenotypicFeature", "disease"


def _make_document(
    triple_idx: int,
    src_id: str,
    src_term: str,
    dst_id: str,
    dst_term: str,
    rel_label: str,
    snomed_type_id: str,
    obj_entity_type: str,
    obj_semantic_class: str,
) -> Document:
    title    = src_term
    abstract = f"{rel_label} {dst_term}"
    text     = f"{src_term} {rel_label} {dst_term}"

    subj_end = len(src_term)
    obj_start = subj_end + 1 + len(rel_label) + 1

    subj = EntityMention(
        id="T1",
        span_start=0,
        span_end=subj_end,
        surface_text=src_term,
        entity_type="DiseaseOrPhenotypicFeature",
        semantic_class="disease",
        original_code=src_id,
        mapped_snomed_id=src_id,
        mapping_confidence=SYNTHETIC_CONFIDENCE,
        snomed_active=True,
        non_snomed=False,
        extra={},
    )
    obj = EntityMention(
        id="T2",
        span_start=obj_start,
        span_end=len(text),
        surface_text=dst_term,
        entity_type=obj_entity_type,
        semantic_class=obj_semantic_class,
        original_code=dst_id,
        mapped_snomed_id=dst_id,
        mapping_confidence=SYNTHETIC_CONFIDENCE,
        snomed_active=True,
        non_snomed=False,
        extra={},
    )
    rel = Relation(
        subject_idx=0,
        object_idx=1,
        source_relation_type=snomed_type_id,
        target_relation=rel_label,
        tier=1,
        target_probability=1.0,
        novelty=None,
        extra={
            "target_candidates": [
                {"target_relation": rel_label, "tier": 1, "probability": 1.0}
            ],
            "snomed_attribute_id": snomed_type_id,
            "subject_class": "disease",
            "object_class": obj_semantic_class,
        },
    )
    return Document(
        pmid=f"SNOMED_SYNTH_{triple_idx:07d}",
        corpus="SNOMED_synthetic",
        split="train",
        title=title,
        abstract=abstract,
        text=text,
        entities=[subj, obj],
        relations=[rel],
    )


def generate_all(verbose: bool = True) -> dict:
    # Use inferred relationships (sct2_Relationship_Snapshot): the US edition
    # stated-relationship snapshot contains only retired US-extension rows (all
    # active=0), so the internationally-classified inferred file is the correct
    # source for causative-agent / due-to / after triples.
    rel_path = SNOMED_FILES["relationships"]
    if not rel_path.exists():
        raise FileNotFoundError(
            f"{rel_path} not found; SNOMED CT RF2 files are required."
        )

    if verbose:
        print("[2.5] loading SNOMED hierarchy (ancestors) ...", flush=True)
    _, ancestors = snomed_hierarchy.load_or_build(force=False, verbose=False)

    if verbose:
        print(f"[2.5] scanning {rel_path.name} ...", flush=True)

    triples_by_attr: Dict[str, List[Tuple[str, str]]] = {k: [] for k in TARGET_ATTR_IDS}
    unscoped_by_attr: Dict[str, int] = {k: 0 for k in TARGET_ATTR_IDS}
    all_concept_ids: Set[str] = set()

    for src, dst, type_id in _iter_stated_rels(rel_path):
        if _check_scope(src, dst, type_id, ancestors):
            triples_by_attr[type_id].append((src, dst))
            all_concept_ids.add(src)
            all_concept_ids.add(dst)
        else:
            unscoped_by_attr[type_id] += 1

    if verbose:
        for attr_id, label in _ATTR_TO_LABEL.items():
            scoped   = len(triples_by_attr[attr_id])
            unscoped = unscoped_by_attr[attr_id]
            print(f"[2.5]   {label:<20s}: {scoped:,} scoped  {unscoped:,} unscoped")

    causative_count = len(triples_by_attr[_ATTR_CAUSATIVE_AGENT])
    if causative_count < MIN_CAUSATIVE_AGENT:
        raise RuntimeError(
            f"causative-agent triple count {causative_count:,} < required minimum "
            f"{MIN_CAUSATIVE_AGENT:,}. Widen scope anchors or check stated relationships file."
        )

    if verbose:
        print(
            f"[2.5] loading descriptions for {len(all_concept_ids):,} concept IDs ...",
            flush=True,
        )
    raw_descs = mrcm.get_descriptions(all_concept_ids)
    descs: Dict[str, str] = {cid: _strip_fsn(t) for cid, t in raw_descs.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _iter_docs() -> Iterator[Document]:
        idx = 0
        for attr_id, label in _ATTR_TO_LABEL.items():
            for src, dst in triples_by_attr[attr_id]:
                obj_entity_type, obj_semantic_class = _obj_semantic(dst, ancestors)
                yield _make_document(
                    idx,
                    src, descs.get(src, src),
                    dst, descs.get(dst, dst),
                    label, attr_id,
                    obj_entity_type, obj_semantic_class,
                )
                idx += 1

    total_written = write_jsonl(_iter_docs(), SYNTH_JSONL)

    summary = {
        "total_documents": total_written,
        "confidence": SYNTHETIC_CONFIDENCE,
        "counts_by_attribute": {
            _ATTR_TO_LABEL[attr_id]: len(triples_by_attr[attr_id])
            for attr_id in TARGET_ATTR_IDS
        },
        "unscoped_by_attribute": {
            _ATTR_TO_LABEL[attr_id]: unscoped_by_attr[attr_id]
            for attr_id in TARGET_ATTR_IDS
        },
        "scope_anchors": {
            "clinical_finding": _CLINICAL_FINDING,
            "substance":        _SUBSTANCE,
            "pharma_product":   _PHARMA_PRODUCT,
            "event":            _EVENT,
            "procedure":        _PROCEDURE,
        },
        "outputs": {"train_jsonl": relative_to_repo(SYNTH_JSONL)},
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[2.5] {total_written:,} synthetic documents written -> {SYNTH_JSONL}")
        print(f"[2.5] summary -> {SUMMARY_JSON}")

    return summary


if __name__ == "__main__":
    generate_all(verbose=True)
