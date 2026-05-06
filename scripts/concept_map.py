"""Apply SNOMED concept mappings to unified annotation documents (CANON Phase 2.2).

Reads the Phase 1.7 verified mapping table (mesh_to_snomed_verified.csv) and
stamps every SNOMED-mappable entity mention in the Phase 2.1 unified JSONL
documents with:

    mapped_snomed_id    -- SNOMED CT concept ID (string)
    mapping_confidence  -- float 0-1 from Phase 1.2; downweighted if inactive
    snomed_active       -- bool; False means the target is a retired SNOMED concept

Policy for snomed_active=False entries
---------------------------------------
Policy applied: **keep-and-flag with downweighted confidence**

    mapping_confidence *= INACTIVE_CONFIDENCE_FACTOR  (default 0.5)
    mapped_snomed_id and snomed_active=False are preserved in the output.

Rationale:
  - Dropping inactive entries reduces dataset size and silently removes a
    known failure mode, making it impossible to measure its impact in Phase 4.6.
  - Redirecting via UMLS MRCUI history is out of scope for Phase 1.7; it would
    require parsing MRCUI and making policy calls for chain-redirects.
  - Keep-and-flag lets the concept normalization head learn lower confidence
    for these mentions (confidence-weighted loss), while Phase 4.6 error
    analysis can filter by snomed_active=False to attribute downstream failures
    directly to this policy choice.

Non-SNOMED entities (genes, variants, species, cell lines — non_snomed=True)
are passed through unchanged: original_code is preserved as-is and
mapped_snomed_id remains None. The NER head trains on all entities; the
CN head and CSP solver skip non_snomed entities.

Composite original codes (e.g. "D003922,D003925" for multi-ID BioRED entities)
are resolved by trying each component; the component with the highest confidence
(preferring active over inactive) wins.

Outputs (under outputs/phase2/mapped/):
    <Corpus>/train.jsonl, dev.jsonl, test.jsonl  -- stamped documents
    mapping_application_summary.json             -- per-corpus counts
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from config import REPO_ROOT
    from unified_format import Document, EntityMention, read_jsonl, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT
    from unified_format import Document, EntityMention, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INACTIVE_CONFIDENCE_FACTOR = 0.5  # multiplied into mapping_confidence for retired SCTIDs

VERIFIED_CSV  = REPO_ROOT / "outputs" / "phase1" / "mesh_to_snomed_verified.csv"
UNIFIED_DIR   = REPO_ROOT / "outputs" / "phase2" / "unified"
OUTPUT_DIR    = REPO_ROOT / "outputs" / "phase2" / "mapped"
SUMMARY_JSON  = REPO_ROOT / "outputs" / "phase2" / "mapping_application_summary.json"


# ---------------------------------------------------------------------------
# Load verified mapping table
# ---------------------------------------------------------------------------

class _MappingEntry:
    __slots__ = ("snomed_id", "confidence", "active")

    def __init__(self, snomed_id: str, confidence: float, active: bool) -> None:
        self.snomed_id  = snomed_id
        self.confidence = confidence
        self.active     = active


def load_verified_table(path: Path = VERIFIED_CSV) -> Dict[str, _MappingEntry]:
    """Return dict mapping mesh_id -> _MappingEntry."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run Phase 1.7 (mapping_verify.py) first."
        )
    table: Dict[str, _MappingEntry] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mesh_id    = (row.get("mesh_id") or "").strip()
            snomed_id  = (row.get("snomed_id") or "").strip()
            active_str = (row.get("snomed_active") or "").strip().lower()
            if not mesh_id or not snomed_id:
                continue
            try:
                confidence = float(row.get("confidence") or 0.0)
            except ValueError:
                confidence = 0.0
            table[mesh_id] = _MappingEntry(
                snomed_id  = snomed_id,
                confidence = confidence,
                active     = (active_str == "true"),
            )
    return table


# ---------------------------------------------------------------------------
# Per-entity mapping logic
# ---------------------------------------------------------------------------

def _best_entry_for_code(
    code: str,
    table: Dict[str, _MappingEntry],
) -> Optional[_MappingEntry]:
    """Resolve a (possibly composite) original_code to the best mapping entry.

    Composite codes like "D003922,D003925" are split on commas; each
    component is tried independently. Active entries beat inactive ones;
    ties are broken by highest confidence.
    """
    if not code or code == "-":
        return None
    components = [c.strip() for c in code.split(",") if c.strip()]
    candidates: List[_MappingEntry] = []
    for comp in components:
        entry = table.get(comp)
        if entry is not None:
            candidates.append(entry)
    if not candidates:
        return None
    # Sort: active first, then highest confidence.
    candidates.sort(key=lambda e: (not e.active, -e.confidence))
    return candidates[0]


def _apply_policy(entry: _MappingEntry) -> Tuple[str, float, bool]:
    """Return (snomed_id, confidence, snomed_active) after policy application."""
    confidence = entry.confidence
    if not entry.active:
        confidence = round(confidence * INACTIVE_CONFIDENCE_FACTOR, 6)
    return entry.snomed_id, confidence, entry.active


def stamp_entity(
    em: EntityMention,
    table: Dict[str, _MappingEntry],
) -> EntityMention:
    """Return a copy of `em` with SNOMED mapping fields populated."""
    if em.non_snomed:
        return em  # pass through unchanged

    code  = (em.original_code or "").strip()
    entry = _best_entry_for_code(code, table)
    if entry is None:
        return em  # unmapped; leave fields as None

    snomed_id, confidence, active = _apply_policy(entry)
    return EntityMention(
        id               = em.id,
        span_start       = em.span_start,
        span_end         = em.span_end,
        surface_text     = em.surface_text,
        entity_type      = em.entity_type,
        semantic_class   = em.semantic_class,
        original_code    = em.original_code,
        mapped_snomed_id = snomed_id,
        mapping_confidence = confidence,
        snomed_active    = active,
        non_snomed       = em.non_snomed,
        extra            = em.extra,
    )


def stamp_document(
    doc: Document,
    table: Dict[str, _MappingEntry],
) -> Document:
    stamped = [stamp_entity(em, table) for em in doc.entities]
    return Document(
        pmid           = doc.pmid,
        corpus         = doc.corpus,
        split          = doc.split,
        title          = doc.title,
        abstract       = doc.abstract,
        text           = doc.text,
        entities       = stamped,
        relations      = doc.relations,
        schema_version = doc.schema_version,
    )


# ---------------------------------------------------------------------------
# Corpus driver
# ---------------------------------------------------------------------------

def _count_outcomes(
    entities: List[EntityMention],
) -> Dict[str, int]:
    mapped_active   = 0
    mapped_inactive = 0
    unmapped_snomed = 0
    non_snomed      = 0
    for em in entities:
        if em.non_snomed:
            non_snomed += 1
        elif em.mapped_snomed_id is not None:
            if em.snomed_active:
                mapped_active += 1
            else:
                mapped_inactive += 1
        else:
            unmapped_snomed += 1
    return {
        "mapped_active":   mapped_active,
        "mapped_inactive": mapped_inactive,
        "unmapped_snomed": unmapped_snomed,
        "non_snomed":      non_snomed,
    }


def apply_all(verbose: bool = True) -> dict:
    """Apply SNOMED mappings to every corpus/split in the unified directory."""
    table = load_verified_table()
    if verbose:
        print(f"[2.2] loaded {len(table):,} verified mapping entries from Phase 1.7")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "policy": "keep-and-flag",
        "inactive_confidence_factor": INACTIVE_CONFIDENCE_FACTOR,
        "mapping_table_rows": len(table),
        "corpora": {},
    }

    for corpus_dir in sorted(UNIFIED_DIR.iterdir()):
        if not corpus_dir.is_dir():
            continue
        corpus = corpus_dir.name
        per_split: dict = {}

        for jsonl_path in sorted(corpus_dir.glob("*.jsonl")):
            split = jsonl_path.stem
            out_path = OUTPUT_DIR / corpus / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            totals: Dict[str, int] = {
                "mapped_active": 0, "mapped_inactive": 0,
                "unmapped_snomed": 0, "non_snomed": 0,
            }
            n_docs = 0

            def _stamped_docs() -> Iterator[Document]:
                nonlocal n_docs
                for doc in read_jsonl(jsonl_path):
                    n_docs += 1
                    yield stamp_document(doc, table)

            written = write_jsonl(_stamped_docs(), out_path)

            # Second pass for counts (small files; read again).
            for doc in read_jsonl(out_path):
                for k, v in _count_outcomes(doc.entities).items():
                    totals[k] += v

            total_ents = sum(totals.values())
            mapped     = totals["mapped_active"] + totals["mapped_inactive"]
            map_pct    = round(100.0 * mapped / total_ents, 1) if total_ents else 0.0

            per_split[split] = {
                "documents": written,
                "entities":  total_ents,
                **totals,
                "mapped_pct": map_pct,
                "output": str(out_path),
            }
            if verbose:
                print(
                    f"[2.2] {corpus:<14s} {split:<5s} -> "
                    f"{written:>5,d} docs  {total_ents:>7,d} ents  "
                    f"mapped={mapped:,} ({map_pct}%)  "
                    f"inactive={totals['mapped_inactive']:,}  "
                    f"unmapped={totals['unmapped_snomed']:,}  "
                    f"non_snomed={totals['non_snomed']:,}"
                )

        summary["corpora"][corpus] = per_split

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[2.2] summary -> {SUMMARY_JSON}")

    return summary


if __name__ == "__main__":
    apply_all(verbose=True)
