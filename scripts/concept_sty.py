"""Resolve CANON concept/source identifiers to UMLS Semantic Network types."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

try:
    import config
    import umls_query
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import umls_query


FALLBACK_STYS: Dict[str, List[str]] = {
    "chemical": ["Pharmacologic Substance"],
    "disease": ["Disease or Syndrome"],
    "gene": ["Gene or Genome"],
    "variant": ["Genetic Function"],
    "species": ["Organism"],
    "cell_line": ["Cell"],
    "clinical_finding": ["Disease or Syndrome"],
    "substance": ["Chemical"],
    "pharmaceutical_product": ["Pharmacologic Substance"],
}

SAB_BY_URI_MARKER = {
    "id.nlm.nih.gov/mesh/": "MSH",
    "snomed.info/id/": "SNOMEDCT_US",
    "ncbi.nlm.nih.gov/gene/": "NCBI",
}


def _notation(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1].replace("MESH:", "")


def resolve_stys(identifier: Optional[str], semantic_class: Optional[str] = None,
                 *, preload: bool = True) -> List[str]:
    """Resolve an ontology identifier through CUI/MRSTY, then audited fallback."""
    if preload:
        umls_query.preload(force=False)
    stys: Set[str] = set()
    if identifier:
        code = _notation(str(identifier))
        sabs: Iterable[str]
        marker_match = [sab for marker, sab in SAB_BY_URI_MARKER.items() if marker in str(identifier)]
        if marker_match:
            sabs = marker_match
        elif code.isdigit():
            sabs = ("SNOMEDCT_US", "NCBI")
        else:
            sabs = ("MSH", "SNOMEDCT_US", "NCBI")
        for sab in sabs:
            for cui in umls_query.code_to_cuis.get((sab, code), []):
                stys.update(umls_query.cui_to_stys.get(cui, []))
    if not stys and semantic_class:
        stys.update(FALLBACK_STYS.get(semantic_class, []))
    return sorted(stys)


def build_lookup(concept_ids: Iterable[str], output_path: Path) -> Dict[str, List[str]]:
    umls_query.preload(force=False)
    lookup = {str(cid): resolve_stys(str(cid), preload=False) for cid in concept_ids}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lookup, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lookup


def load_lookup(path: Path) -> Dict[str, List[str]]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["FALLBACK_STYS", "resolve_stys", "build_lookup", "load_lookup"]
