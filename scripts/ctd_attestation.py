"""CTD direct-evidence lower-bound attestation for CANON Phase 4.6."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, FrozenSet, Tuple


def load_direct_evidence(path: Path, cache_path: Path | None = None) -> Dict[str, FrozenSet[Tuple[str, str]]]:
    if cache_path and cache_path.exists() and cache_path.stat().st_mtime >= Path(path).stat().st_mtime:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return {key: frozenset((a, b) for a, b in values) for key, values in data.items()}
    causes, treats = set(), set()
    with Path(path).open("r", encoding="utf-8") as fh:
        header = None
        for line in fh:
            if line.startswith("# ChemicalName"):
                header = line[2:].rstrip("\n").split("\t")
                break
            if not line.startswith("#"):
                header = line.rstrip("\n").split("\t")
                break
        if not header:
            return {"causes": frozenset(), "treats": frozenset()}
        reader = csv.DictReader(fh, fieldnames=header, delimiter="\t")
        for row in reader:
            evidence = (row.get("DirectEvidence") or "").strip()
            chemical = (row.get("ChemicalID") or "").replace("MESH:", "").strip()
            disease = (row.get("DiseaseID") or "").replace("MESH:", "").strip()
            if not evidence or not chemical or not disease:
                continue
            if evidence == "marker/mechanism":
                causes.add((chemical, disease))
            elif evidence == "therapeutic":
                treats.add((chemical, disease))
    result = {"causes": frozenset(causes), "treats": frozenset(treats)}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({k: sorted(v) for k, v in result.items()}) + "\n",
                              encoding="utf-8")
    return result


def attested(label: str, chemical_mesh: str, disease_mesh: str,
             lookup: Dict[str, FrozenSet[Tuple[str, str]]]) -> bool:
    normalized = "causes" if label == "causative-agent" else label
    return (chemical_mesh.replace("MESH:", ""), disease_mesh.replace("MESH:", "")) in lookup.get(normalized, frozenset())


__all__ = ["load_direct_evidence", "attested"]
