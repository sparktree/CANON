"""Build MeSH-keyed MED-RT therapeutic attestation from UMLS MRREL."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, FrozenSet, Set, Tuple

try:
    import umls_query
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import umls_query


THERAPEUTIC_RELAS = frozenset({
    "may_treat", "may_prevent", "may_diagnose", "mechanism_of_action",
    "physiologic_effect", "therapeutic_class",
})


def build_mesh_attestation() -> Dict[str, FrozenSet[Tuple[str, str]]]:
    umls_query.preload(force=False)
    pairs: Set[Tuple[str, str]] = set()
    mesh_by_cui = {
        cui: {a["code"] for a in atoms if a.get("sab") == "MSH" and a.get("code")}
        for cui, atoms in umls_query.cui_to_atoms.items()
    }
    for cui, rels in umls_query.cui_to_rels.items():
        left = mesh_by_cui.get(cui, set())
        if not left:
            continue
        for rel in rels:
            if rel.get("rela") not in THERAPEUTIC_RELAS:
                continue
            right = mesh_by_cui.get(rel.get("cui2"), set())
            pairs.update((a, b) for a in left for b in right)
    return {"treats": frozenset(pairs)}


__all__ = ["THERAPEUTIC_RELAS", "build_mesh_attestation"]
