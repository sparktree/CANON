"""MeSH -> SNOMED CT concept mapping (CANON Phase 1.2).

Four-priority pipeline applied to every unique MeSH descriptor harvested from
BioRED, BC5CDR, NCBI Disease, and NLM-Chem:

    1. MRMAP curated mapping            (confidence 0.95)
    2. Shared CUI alignment             (confidence 0.90)
    3. MRREL traversal (SY/RQ/RB/RN)    (confidence 0.55-0.70)
    4. Semantic-type fallback           (confidence 0.40)

Selection rule within each tier: pick the SNOMED atom with the best TTY
(PT > FN > SY > others), tie-broken by shortest preferred-term length, then
lexicographically smallest concept ID. This gives the deterministic
"context-independent best match" the plan calls for.

Outputs (under CANON/outputs/phase1/):
    mesh_to_snomed.csv             - flat mapping table
    mesh_to_snomed_unmapped.csv    - MeSH IDs we could not map
    coverage_stats.csv             - per-corpus / per-entity-class coverage
    low_confidence_top100.csv      - top-100 high-frequency, low-confidence
                                     entries for manual SNOMED browser review
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from config import REPO_ROOT
    from mesh_harvest import aggregate, harvest_all
    import umls_query
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT
    from mesh_harvest import aggregate, harvest_all
    import umls_query


OUTPUT_DIR = REPO_ROOT / "outputs" / "phase1"

# ---------------------------------------------------------------------------
# Confidence schedule for each mapping method
# ---------------------------------------------------------------------------
CONF_MRMAP = 0.95
CONF_SHARED_CUI = 0.90
CONF_MRREL_SY = 0.70
CONF_MRREL_RQ = 0.68
CONF_MRREL_RB = 0.65
CONF_MRREL_RN = 0.55
CONF_STY_FALLBACK = 0.40
CONFIDENCE_THRESHOLD = 0.80  # Plan's high-confidence cutoff

_TTY_RANK = {"PT": 0, "FN": 1, "SY": 2, "PTGB": 3}


# Semantic type -> top-level SNOMED concept used for the last-ditch fallback.
# These are deliberately broad so that the CSP solver still has a coherent
# domain/range to enforce in Phase 3.5.
_STY_TO_SNOMED_ROOT: Dict[str, Tuple[str, str]] = {
    # Diseases / clinical findings
    "Disease or Syndrome": ("64572001", "Disease"),
    "Pathologic Function": ("64572001", "Disease"),
    "Mental or Behavioral Dysfunction": ("74732009", "Mental disorder"),
    "Neoplastic Process": ("108369006", "Neoplasm"),
    "Sign or Symptom": ("404684003", "Clinical finding"),
    "Finding": ("404684003", "Clinical finding"),
    "Anatomical Abnormality": ("49755003", "Morphologically abnormal structure"),
    "Congenital Abnormality": ("66091009", "Congenital disease"),
    "Acquired Abnormality": ("49755003", "Morphologically abnormal structure"),
    "Injury or Poisoning": ("417746004", "Traumatic injury"),
    "Cell or Molecular Dysfunction": ("404684003", "Clinical finding"),
    # Substances / chemicals
    "Pharmacologic Substance": ("373873005", "Pharmaceutical / biologic product"),
    "Antibiotic": ("373873005", "Pharmaceutical / biologic product"),
    "Clinical Drug": ("373873005", "Pharmaceutical / biologic product"),
    "Organic Chemical": ("105590001", "Substance"),
    "Inorganic Chemical": ("105590001", "Substance"),
    "Chemical": ("105590001", "Substance"),
    "Chemical Viewed Functionally": ("105590001", "Substance"),
    "Chemical Viewed Structurally": ("105590001", "Substance"),
    "Hormone": ("105590001", "Substance"),
    "Enzyme": ("105590001", "Substance"),
    "Nucleic Acid, Nucleoside, or Nucleotide": ("105590001", "Substance"),
    "Element, Ion, or Isotope": ("105590001", "Substance"),
    "Vitamin": ("105590001", "Substance"),
    "Steroid": ("105590001", "Substance"),
    "Lipid": ("105590001", "Substance"),
    "Carbohydrate": ("105590001", "Substance"),
    "Amino Acid, Peptide, or Protein": ("105590001", "Substance"),
    "Biomedical or Dental Material": ("105590001", "Substance"),
    "Indicator, Reagent, or Diagnostic Aid": ("105590001", "Substance"),
    "Hazardous or Poisonous Substance": ("105590001", "Substance"),
    "Immunologic Factor": ("105590001", "Substance"),
    "Receptor": ("105590001", "Substance"),
    "Neuroreactive Substance or Biogenic Amine": ("105590001", "Substance"),
    "Body Substance": ("105590001", "Substance"),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    snomed_id: str
    snomed_term: str
    tty: str
    confidence: float
    method: str
    source_cui: Optional[str] = None
    intermediate_cui: Optional[str] = None
    rel: Optional[str] = None

    def quality_key(self) -> tuple:
        """Lower is better."""
        return (
            -self.confidence,
            _TTY_RANK.get(self.tty, 99),
            len(self.snomed_term or ""),
            self.snomed_id,
        )


# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------
def _best_snomed_atom(atoms: Iterable[dict]) -> Optional[dict]:
    snomed = [a for a in atoms if a.get("sab") == "SNOMEDCT_US" and a.get("code")]
    if not snomed:
        return None
    snomed.sort(key=lambda a: (_TTY_RANK.get(a.get("tty", ""), 99), len(a.get("str") or ""), a["code"]))
    return snomed[0]


def _candidates_mrmap(mesh_id: str) -> List[Candidate]:
    out: List[Candidate] = []
    for entry in umls_query.get_curated_mapping(mesh_id):
        # MRMAP entries currently use mapsetsab as a proxy; only accept rows whose
        # to_code looks like a SNOMED concept id (numeric, length 6-18).
        to_code = (entry.get("to_code") or "").strip()
        if not to_code.isdigit() or not (6 <= len(to_code) <= 18):
            continue
        # Look up the preferred SNOMED term via the atoms index.
        cuis = umls_query.code_to_cuis.get(("SNOMEDCT_US", to_code), [])
        term, tty = to_code, ""
        for cui in cuis:
            atom = _best_snomed_atom(
                a for a in umls_query.cui_to_atoms.get(cui, [])
                if a.get("code") == to_code
            )
            if atom:
                term = atom.get("str") or term
                tty = atom.get("tty", "")
                break
        out.append(
            Candidate(
                snomed_id=to_code,
                snomed_term=term,
                tty=tty,
                confidence=CONF_MRMAP,
                method="mrmap_curated",
            )
        )
    return out


def _candidates_shared_cui(mesh_id: str) -> List[Candidate]:
    out: List[Candidate] = []
    for cui in umls_query.get_cuis_for_mesh(mesh_id):
        atom = _best_snomed_atom(umls_query.cui_to_atoms.get(cui, []))
        if atom:
            out.append(
                Candidate(
                    snomed_id=atom["code"],
                    snomed_term=atom.get("str") or atom["code"],
                    tty=atom.get("tty", ""),
                    confidence=CONF_SHARED_CUI,
                    method="shared_cui",
                    source_cui=cui,
                )
            )
    return out


_REL_CONFIDENCE = {
    "SY": (CONF_MRREL_SY, "mrrel_sy"),
    "RQ": (CONF_MRREL_RQ, "mrrel_rq"),
    "RB": (CONF_MRREL_RB, "mrrel_rb"),
    "RN": (CONF_MRREL_RN, "mrrel_rn"),
}


def _candidates_mrrel(mesh_id: str) -> List[Candidate]:
    out: List[Candidate] = []
    for cui in umls_query.get_cuis_for_mesh(mesh_id):
        for rel in umls_query.get_relations(cui):
            rel_type = rel.get("rel")
            if rel_type not in _REL_CONFIDENCE:
                continue
            other = rel.get("cui2")
            if not other:
                continue
            atom = _best_snomed_atom(umls_query.cui_to_atoms.get(other, []))
            if not atom:
                continue
            conf, method = _REL_CONFIDENCE[rel_type]
            out.append(
                Candidate(
                    snomed_id=atom["code"],
                    snomed_term=atom.get("str") or atom["code"],
                    tty=atom.get("tty", ""),
                    confidence=conf,
                    method=method,
                    source_cui=cui,
                    intermediate_cui=other,
                    rel=rel_type,
                )
            )
    return out


def _candidates_sty(mesh_id: str) -> List[Candidate]:
    out: List[Candidate] = []
    seen: set[str] = set()
    for cui in umls_query.get_cuis_for_mesh(mesh_id):
        for sty in umls_query.get_semantic_types(cui):
            root = _STY_TO_SNOMED_ROOT.get(sty)
            if not root or root[0] in seen:
                continue
            seen.add(root[0])
            out.append(
                Candidate(
                    snomed_id=root[0],
                    snomed_term=root[1],
                    tty="PT",
                    confidence=CONF_STY_FALLBACK,
                    method=f"sty_fallback:{sty}",
                    source_cui=cui,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Top-level mapper
# ---------------------------------------------------------------------------
def map_mesh(mesh_id: str) -> Optional[Candidate]:
    for tier in (
        _candidates_mrmap,
        _candidates_shared_cui,
        _candidates_mrrel,
        _candidates_sty,
    ):
        candidates = tier(mesh_id)
        if candidates:
            candidates.sort(key=Candidate.quality_key)
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_mapping_table(rows: List[dict], path: Path) -> None:
    fields = [
        "mesh_id",
        "snomed_id",
        "snomed_term",
        "confidence",
        "mapping_method",
        "frequency",
        "entity_classes",
        "corpora",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_unmapped_table(rows: List[dict], path: Path) -> None:
    fields = ["mesh_id", "frequency", "entity_classes", "corpora", "reason"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def compute_coverage(per_corpus, mapping_by_mesh):
    """{(corpus, entity_class): {total, mapped, high_conf, mean_confidence}}"""
    stats = defaultdict(lambda: {"total": 0, "mapped": 0, "high_conf": 0, "conf_sum": 0.0})
    for corpus, counts in per_corpus.items():
        for (entity_class, mesh_id), freq in counts.items():
            key = (corpus, entity_class)
            stats[key]["total"] += freq
            mapping = mapping_by_mesh.get(mesh_id)
            if mapping is not None:
                stats[key]["mapped"] += freq
                stats[key]["conf_sum"] += mapping.confidence * freq
                if mapping.confidence >= CONFIDENCE_THRESHOLD:
                    stats[key]["high_conf"] += freq
    rows = []
    for (corpus, entity_class), v in sorted(stats.items()):
        total = v["total"]
        rows.append(
            {
                "corpus": corpus,
                "entity_class": entity_class,
                "total_mentions": total,
                "mapped_mentions": v["mapped"],
                "high_confidence_mentions": v["high_conf"],
                "mapped_pct": f"{(v['mapped'] / total * 100) if total else 0:.1f}",
                "high_conf_pct": f"{(v['high_conf'] / total * 100) if total else 0:.1f}",
                "mean_confidence": f"{(v['conf_sum'] / v['mapped']) if v['mapped'] else 0:.3f}",
            }
        )
    return rows


def write_coverage(rows: List[dict], path: Path) -> None:
    fields = [
        "corpus",
        "entity_class",
        "total_mentions",
        "mapped_mentions",
        "high_confidence_mentions",
        "mapped_pct",
        "high_conf_pct",
        "mean_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def low_confidence_top(rows: List[dict], n: int = 100) -> List[dict]:
    """Top-n entries: high frequency × low confidence, suitable for SNOMED browser review."""
    candidates = [r for r in rows if float(r["confidence"]) < CONFIDENCE_THRESHOLD]
    candidates.sort(key=lambda r: (-int(r["frequency"]), float(r["confidence"])))
    return candidates[:n]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_mapping(verbose: bool = True):
    if verbose:
        print("[mesh_to_snomed] preloading UMLS dictionaries...", flush=True)
    umls_query.preload()

    if verbose:
        print("[mesh_to_snomed] harvesting MeSH IDs from corpora...", flush=True)
    per_corpus = harvest_all()
    agg = aggregate(per_corpus)
    if verbose:
        print(f"[mesh_to_snomed] {len(agg):,} unique MeSH descriptors to map")

    method_counter: Counter = Counter()
    mapping_rows: List[dict] = []
    unmapped_rows: List[dict] = []
    mapping_by_mesh: Dict[str, Candidate] = {}

    for mesh_id, meta in agg.items():
        cand = map_mesh(mesh_id)
        entity_classes = ",".join(sorted(meta["entity_classes"]))
        corpora = ",".join(sorted(meta["corpora"]))
        if cand is None:
            method_counter["unmapped"] += 1
            unmapped_rows.append(
                {
                    "mesh_id": mesh_id,
                    "frequency": meta["frequency"],
                    "entity_classes": entity_classes,
                    "corpora": corpora,
                    "reason": "no_cui" if not umls_query.get_cuis_for_mesh(mesh_id) else "no_snomed_path",
                }
            )
            continue
        method_counter[cand.method.split(":", 1)[0]] += 1
        mapping_by_mesh[mesh_id] = cand
        mapping_rows.append(
            {
                "mesh_id": mesh_id,
                "snomed_id": cand.snomed_id,
                "snomed_term": cand.snomed_term,
                "confidence": f"{cand.confidence:.2f}",
                "mapping_method": cand.method,
                "frequency": meta["frequency"],
                "entity_classes": entity_classes,
                "corpora": corpora,
            }
        )

    out_dir = _ensure_output_dir()
    mapping_path = out_dir / "mesh_to_snomed.csv"
    unmapped_path = out_dir / "mesh_to_snomed_unmapped.csv"
    coverage_path = out_dir / "coverage_stats.csv"
    review_path = out_dir / "low_confidence_top100.csv"

    mapping_rows.sort(key=lambda r: (-int(r["frequency"]), r["mesh_id"]))
    unmapped_rows.sort(key=lambda r: (-int(r["frequency"]), r["mesh_id"]))

    write_mapping_table(mapping_rows, mapping_path)
    write_unmapped_table(unmapped_rows, unmapped_path)
    write_coverage(compute_coverage(per_corpus, mapping_by_mesh), coverage_path)
    write_mapping_table(low_confidence_top(mapping_rows), review_path)

    if verbose:
        print("[mesh_to_snomed] method tally:")
        for method, count in method_counter.most_common():
            print(f"    {method:<22s} {count:>6d}")
        print(f"[mesh_to_snomed] wrote {mapping_path}")
        print(f"[mesh_to_snomed] wrote {unmapped_path}")
        print(f"[mesh_to_snomed] wrote {coverage_path}")
        print(f"[mesh_to_snomed] wrote {review_path}")

    return {
        "mapping_rows": mapping_rows,
        "unmapped_rows": unmapped_rows,
        "method_counts": dict(method_counter),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the MeSH -> SNOMED mapping table.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--probe", metavar="MESH_ID", help="Map a single MeSH ID and exit.")
    args = p.parse_args()

    if args.probe:
        umls_query.preload()
        cand = map_mesh(args.probe)
        if cand is None:
            print(f"{args.probe}: UNMAPPED")
        else:
            print(
                f"{args.probe} -> SNOMED {cand.snomed_id} ({cand.snomed_term}) "
                f"conf={cand.confidence:.2f} method={cand.method}"
            )
    else:
        build_mapping(verbose=not args.quiet)
