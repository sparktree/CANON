"""PubTator 3.0 silver data processing for CANON Phase 2.6.

Reads the three flat TSV annotation files from the PubTator3 download:
    disease2pubtator3.gz   -- disease entity annotations (pmid, type, MESH:id, mention)
    chemical2pubtator3.gz  -- chemical entity annotations (same schema)
    relation2pubtator3.gz  -- cross-entity relations     (pmid, rel_type, subj, obj)

Produces unified-format silver training documents for Chemical–Disease pairs only.
Each relation row becomes one Document. Text is synthetic:
    "{subj_mention} {rel_label} {obj_mention}"

Scale controls:
    max_docs      -- hard ceiling on output documents (default 500,000)
    max_pmids     -- limit PMIDs read from relation file to control memory (default 2,000,000)

Quality gates (silver documents tolerate missing SNOMED mappings):
    1. PMID must not appear in the gold corpus.
    2. Both subject and object must have a valid MeSH identifier (not "-").
    3. At least one entity must have a recognisable mention string.
    4. silver weight (SILVER_WEIGHT=0.4) is applied to any entity that does receive
       a SNOMED mapping from the existing 2,750-entry table.

PubTator3 relation types are mapped inline to canonical labels:
    treat / cotreat   → treats / co-treats
    cause             → causes
    everything else   → associated-with

Outputs:
    outputs/phase2/silver/train.jsonl
    outputs/phase2/silver/silver_summary.json
    outputs/phase2/silver/silver_checkpoint.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

try:
    import orjson as _orjson
    def _dumps(obj: dict) -> str:
        return _orjson.dumps(obj, option=_orjson.OPT_NON_STR_KEYS).decode()
    _loads = _orjson.loads
except ImportError:
    def _dumps(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False)
    _loads = json.loads

try:
    from config import REPO_ROOT, DATA_ROOT
    import concept_map
    from unified_format import Document, EntityMention, Relation, read_jsonl, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, DATA_ROOT
    import concept_map
    from unified_format import Document, EntityMention, Relation, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PT3_DIR         = DATA_ROOT / "PubTator3-2"
DISEASE_FILE    = PT3_DIR / "disease2pubtator3.gz"
CHEMICAL_FILE   = PT3_DIR / "chemical2pubtator3.gz"
RELATION_FILE   = PT3_DIR / "relation2pubtator3.gz"

OUTPUT_DIR      = REPO_ROOT / "outputs" / "phase2" / "silver"
SILVER_JSONL    = OUTPUT_DIR / "train.jsonl"
SUMMARY_JSON    = OUTPUT_DIR / "silver_summary.json"
CHECKPOINT_JSON = OUTPUT_DIR / "silver_checkpoint.json"
GOLD_MAPPED_DIR = REPO_ROOT / "outputs" / "phase2" / "relation_mapped"


# ---------------------------------------------------------------------------
# Scale / quality constants
# ---------------------------------------------------------------------------

SILVER_WEIGHT   = 0.4      # applied to mapping_confidence where SNOMED mapping exists
MIN_RELATIONS   = 1        # min relations per document after construction
DEFAULT_MAX_DOCS  = 500_000
DEFAULT_MAX_PMIDS = 2_000_000  # PMID cap to bound memory during relation loading


# ---------------------------------------------------------------------------
# PubTator3 entity types (Chemical/Disease are SNOMED-mappable)
# ---------------------------------------------------------------------------

_PT3_TYPE_MAP: Dict[str, Tuple[str, str, bool]] = {
    "Disease":  ("DiseaseOrPhenotypicFeature", "disease",  False),
    "Chemical": ("ChemicalEntity",             "chemical", False),
}

_MAPPABLE_TYPES = frozenset({"Disease", "Chemical"})


# ---------------------------------------------------------------------------
# PubTator3 relation type → canonical unified label
# ---------------------------------------------------------------------------

_PT3_REL_CANONICAL: Dict[str, str] = {
    "treat":              "treats",
    "cotreat":            "co-treats",
    "cause":              "causes",
    "associate":          "associated-with",
    "negative_correlate": "associated-with",
    "positive_correlate": "associated-with",
    "stimulate":          "associated-with",
    "inhibit":            "associated-with",
    "prevent":            "associated-with",
    "compare":            "associated-with",
    "interact":           "associated-with",
    "drug_interact":      "associated-with",
}

# Tier for each canonical label
_LABEL_TIER: Dict[str, int] = {
    "treats":         2,
    "co-treats":      2,
    "causes":         2,
    "associated-with": 2,
}


# ---------------------------------------------------------------------------
# Streaming TSV reader
# ---------------------------------------------------------------------------

def _open_gz(path: Path):
    return gzip.open(path, "rt", encoding="utf-8")


def _iter_tsv(path: Path) -> Iterator[List[str]]:
    """Yield tab-split rows, skipping malformed lines."""
    with _open_gz(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                yield parts


# ---------------------------------------------------------------------------
# Gold PMID loader
# ---------------------------------------------------------------------------

def _load_gold_pmids() -> FrozenSet[str]:
    pmids: Set[str] = set()
    if not GOLD_MAPPED_DIR.exists():
        return frozenset()
    for jsonl_path in GOLD_MAPPED_DIR.rglob("*.jsonl"):
        for doc in read_jsonl(jsonl_path):
            pmids.add(doc.pmid)
    return frozenset(pmids)


# ---------------------------------------------------------------------------
# Phase 1: stream relation file → pmid_rels dict
# ---------------------------------------------------------------------------

def _load_relations(
    gold_pmids: FrozenSet[str],
    max_pmids: int,
    verbose: bool,
) -> Dict[str, List[Tuple[str, str, str, str, str]]]:
    """Return {pmid: [(rel_type, subj_type, subj_mesh, obj_type, obj_mesh)]} for
    Chemical/Disease pairs only, capped at max_pmids unique PMIDs.
    """
    pmid_rels: Dict[str, List[Tuple[str, str, str, str, str]]] = {}
    n_rows = 0
    n_skipped_type = 0
    n_skipped_gold = 0

    with _open_gz(RELATION_FILE) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            pmid, rel_type, subj_raw, obj_raw = parts[0], parts[1], parts[2], parts[3]
            n_rows += 1

            # Parse "Type|MESH:DXXXXXX" fields
            subj_parts = subj_raw.split("|", 1)
            obj_parts  = obj_raw.split("|", 1)
            if len(subj_parts) < 2 or len(obj_parts) < 2:
                n_skipped_type += 1
                continue

            subj_type, subj_id_raw = subj_parts[0], subj_parts[1]
            obj_type,  obj_id_raw  = obj_parts[0],  obj_parts[1]

            # Keep only Chemical/Disease pairs
            if subj_type not in _MAPPABLE_TYPES or obj_type not in _MAPPABLE_TYPES:
                n_skipped_type += 1
                continue

            # Strip MESH: prefix; skip entities without MeSH identifiers
            subj_mesh = subj_id_raw[5:] if subj_id_raw.startswith("MESH:") else subj_id_raw
            obj_mesh  = obj_id_raw[5:]  if obj_id_raw.startswith("MESH:")  else obj_id_raw
            if subj_mesh == "-" or obj_mesh == "-" or not subj_mesh or not obj_mesh:
                n_skipped_type += 1
                continue

            if pmid in gold_pmids:
                n_skipped_gold += 1
                continue

            if pmid not in pmid_rels:
                if len(pmid_rels) >= max_pmids:
                    break  # PMID cap reached
                pmid_rels[pmid] = []

            pmid_rels[pmid].append((rel_type, subj_type, subj_mesh, obj_type, obj_mesh))

    if verbose:
        total_rels = sum(len(v) for v in pmid_rels.values())
        print(f"[2.6] relation file: {n_rows:,} rows scanned  "
              f"{total_rels:,} Chem/Dis rels  {len(pmid_rels):,} PMIDs  "
              f"(skip_type={n_skipped_type:,}  skip_gold={n_skipped_gold:,})")
    return pmid_rels


# ---------------------------------------------------------------------------
# Phase 2: stream entity file → mention lookup
# ---------------------------------------------------------------------------

def _load_mentions(
    path: Path,
    needed_pmids: Set[str],
    verbose: bool,
    label: str,
) -> Dict[Tuple[str, str], str]:
    """Return {(pmid, mesh_id): first_mention} for needed_pmids."""
    lookup: Dict[Tuple[str, str], str] = {}
    n_rows = 0
    with _open_gz(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            pmid = parts[0]
            if pmid not in needed_pmids:
                continue
            ident = parts[2]
            if ident == "-" or not ident.startswith("MESH:"):
                continue
            mesh_id = ident[5:]
            key = (pmid, mesh_id)
            if key not in lookup:
                raw_mentions = parts[3]
                first_mention = raw_mentions.split("|")[0].strip()
                if first_mention:
                    lookup[key] = first_mention
            n_rows += 1
    if verbose:
        print(f"[2.6] {label}: {len(lookup):,} (pmid, mesh_id) → mention entries loaded")
    return lookup


# ---------------------------------------------------------------------------
# Phase 3: build unified Documents
# ---------------------------------------------------------------------------

def _make_silver_doc(
    pmid: str,
    rel_type: str,
    subj_type: str,
    subj_mesh: str,
    subj_mention: str,
    obj_type: str,
    obj_mesh: str,
    obj_mention: str,
    mapping_table: dict,
) -> Optional[Document]:
    canonical_label = _PT3_REL_CANONICAL.get(rel_type, "associated-with")
    tier = _LABEL_TIER.get(canonical_label, 2)

    subj_entity_type, subj_sem_class, subj_non_snomed = _PT3_TYPE_MAP[subj_type]
    obj_entity_type,  obj_sem_class,  obj_non_snomed  = _PT3_TYPE_MAP[obj_type]

    text = f"{subj_mention} {canonical_label} {obj_mention}"
    subj_end = len(subj_mention)
    obj_start = subj_end + 1 + len(canonical_label) + 1

    def _stamp_entity(
        idx: int,
        mesh_id: str,
        span_start: int,
        span_end: int,
        mention: str,
        entity_type: str,
        sem_class: str,
        non_snomed: bool,
    ) -> EntityMention:
        entry = mapping_table.get(mesh_id)
        snomed_id  = entry.snomed_id  if entry else None
        confidence = round(entry.confidence * SILVER_WEIGHT, 6) if entry else None
        active     = entry.active     if entry else None
        return EntityMention(
            id=f"T{idx}",
            span_start=span_start,
            span_end=span_end,
            surface_text=mention,
            entity_type=entity_type,
            semantic_class=sem_class,
            original_code=mesh_id,
            mapped_snomed_id=snomed_id,
            mapping_confidence=confidence,
            snomed_active=active,
            non_snomed=non_snomed,
            extra={},
        )

    subj_em = _stamp_entity(
        1, subj_mesh, 0, subj_end,
        subj_mention, subj_entity_type, subj_sem_class, subj_non_snomed,
    )
    obj_em = _stamp_entity(
        2, obj_mesh, obj_start, len(text),
        obj_mention, obj_entity_type, obj_sem_class, obj_non_snomed,
    )

    rel = Relation(
        subject_idx=0,
        object_idx=1,
        source_relation_type=rel_type,
        target_relation=canonical_label,
        tier=tier,
        target_probability=1.0,
        novelty=None,
        extra={
            "target_candidates": [
                {"target_relation": canonical_label, "tier": tier, "probability": 1.0}
            ],
            "silver": True,
            "pt3_relation": rel_type,
            "subject_class": subj_sem_class,
            "object_class": obj_sem_class,
        },
    )

    return Document(
        pmid=pmid,
        corpus="PubTator_silver",
        split="train",
        title=subj_mention,
        abstract=f"{canonical_label} {obj_mention}",
        text=text,
        entities=[subj_em, obj_em],
        relations=[rel],
    )


def _iter_silver_docs(
    pmid_rels: Dict[str, List[Tuple[str, str, str, str, str]]],
    dis_mentions: Dict[Tuple[str, str], str],
    chem_mentions: Dict[Tuple[str, str], str],
    mapping_table: dict,
) -> Iterator[Document]:
    for pmid, rels in pmid_rels.items():
        for rel_type, subj_type, subj_mesh, obj_type, obj_mesh in rels:
            # Look up mention text from the correct entity file
            subj_lookup = dis_mentions if subj_type == "Disease" else chem_mentions
            obj_lookup  = dis_mentions if obj_type  == "Disease" else chem_mentions

            subj_mention = subj_lookup.get((pmid, subj_mesh), "")
            obj_mention  = obj_lookup.get((pmid, obj_mesh),   "")

            if not subj_mention or not obj_mention:
                continue  # skip if no mention text available

            doc = _make_silver_doc(
                pmid, rel_type,
                subj_type, subj_mesh, subj_mention,
                obj_type,  obj_mesh,  obj_mention,
                mapping_table,
            )
            if doc is not None:
                yield doc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if CHECKPOINT_JSON.exists():
        try:
            return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_checkpoint(data: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_all(
    max_docs: int = DEFAULT_MAX_DOCS,
    max_pmids: int = DEFAULT_MAX_PMIDS,
    reset_checkpoint: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the full silver pipeline. Returns a summary dict."""
    missing = [p for p in (DISEASE_FILE, CHEMICAL_FILE, RELATION_FILE) if not p.exists()]
    if missing:
        if verbose:
            print(f"[2.6] PubTator3 files not found: {[p.name for p in missing]}")
            print(f"[2.6] Expected in: {PT3_DIR}")
            print(f"[2.6] Writing empty output.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SILVER_JSONL.write_text("", encoding="utf-8")
        summary = {"total_documents": 0, "missing_files": [p.name for p in missing]}
        with SUMMARY_JSON.open("w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    checkpoint = {} if reset_checkpoint else _load_checkpoint()

    if verbose:
        print(f"[2.6] PubTator3 data dir: {PT3_DIR}")
        print(f"[2.6] max_docs={max_docs:,}  max_pmids={max_pmids:,}")

    # Load gold PMIDs
    if verbose:
        print("[2.6] loading gold PMID exclusion set ...", flush=True)
    gold_pmids = _load_gold_pmids()
    if verbose:
        print(f"[2.6] {len(gold_pmids):,} gold PMIDs will be excluded")

    # Phase 1: relations
    if verbose:
        print("[2.6] streaming relation file ...", flush=True)
    pmid_rels = _load_relations(gold_pmids, max_pmids, verbose)

    needed_pmids: Set[str] = set(pmid_rels.keys())

    # Phase 2: entity mentions
    if verbose:
        print(f"[2.6] loading disease mentions for {len(needed_pmids):,} PMIDs ...", flush=True)
    dis_mentions = _load_mentions(DISEASE_FILE, needed_pmids, verbose, "disease")

    if verbose:
        print(f"[2.6] loading chemical mentions for {len(needed_pmids):,} PMIDs ...", flush=True)
    chem_mentions = _load_mentions(CHEMICAL_FILE, needed_pmids, verbose, "chemical")

    # Phase 3: SNOMED mapping table
    if verbose:
        print("[2.6] loading SNOMED concept mapping table ...", flush=True)
    mapping_table = concept_map.load_verified_table()
    if verbose:
        print(f"[2.6] {len(mapping_table):,} MeSH→SNOMED entries available")

    # Phase 4: build + write documents
    if verbose:
        print(f"[2.6] building silver documents (cap={max_docs:,}) ...", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_candidate = 0
    total_skip_mention = 0

    def _capped_docs() -> Iterator[Document]:
        nonlocal total_written, total_candidate, total_skip_mention
        for pmid, rels in pmid_rels.items():
            if total_written >= max_docs:
                break
            for rel_type, subj_type, subj_mesh, obj_type, obj_mesh in rels:
                if total_written >= max_docs:
                    break
                total_candidate += 1
                subj_lookup = dis_mentions if subj_type == "Disease" else chem_mentions
                obj_lookup  = dis_mentions if obj_type  == "Disease" else chem_mentions
                subj_mention = subj_lookup.get((pmid, subj_mesh), "")
                obj_mention  = obj_lookup.get((pmid, obj_mesh),   "")
                if not subj_mention or not obj_mention:
                    total_skip_mention += 1
                    continue
                doc = _make_silver_doc(
                    pmid, rel_type,
                    subj_type, subj_mesh, subj_mention,
                    obj_type,  obj_mesh,  obj_mention,
                    mapping_table,
                )
                if doc is not None:
                    total_written += 1
                    yield doc

    write_jsonl(_capped_docs(), SILVER_JSONL)

    # Relation type breakdown
    rel_counts: Dict[str, int] = {}
    for rels in pmid_rels.values():
        for rel_type, *_ in rels:
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

    summary = {
        "total_documents":   total_written,
        "total_candidate_relations": total_candidate,
        "skipped_no_mention": total_skip_mention,
        "max_docs":          max_docs,
        "max_pmids":         max_pmids,
        "pmids_loaded":      len(pmid_rels),
        "silver_weight":     SILVER_WEIGHT,
        "snomed_table_size": len(mapping_table),
        "relation_type_counts": dict(sorted(rel_counts.items(), key=lambda x: -x[1])),
        "outputs": {"train_jsonl": str(SILVER_JSONL)},
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    checkpoint.update({"total_written": total_written, "complete": True})
    _save_checkpoint(checkpoint)

    if verbose:
        print(f"[2.6] {total_written:,} silver documents written -> {SILVER_JSONL}")
        print(f"[2.6] summary -> {SUMMARY_JSON}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PubTator 3.0 silver data (Phase 2.6).")
    parser.add_argument("--max-docs",  type=int, default=DEFAULT_MAX_DOCS,
                        help=f"Hard cap on output documents (default {DEFAULT_MAX_DOCS:,})")
    parser.add_argument("--max-pmids", type=int, default=DEFAULT_MAX_PMIDS,
                        help=f"PMID cap for memory control (default {DEFAULT_MAX_PMIDS:,})")
    parser.add_argument("--reset-checkpoint", action="store_true")
    args = parser.parse_args()
    process_all(
        max_docs=args.max_docs,
        max_pmids=args.max_pmids,
        reset_checkpoint=args.reset_checkpoint,
        verbose=True,
    )
