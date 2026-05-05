"""Harvest unique MeSH descriptor IDs from corpora used in CANON Phase 1.2.

Each corpus contributes (corpus_name, entity_class, mesh_id, frequency).
entity_class is normalized to one of: 'chemical', 'disease', 'other'.
Genes, variants, species, and cell lines are dropped per Phase 1.3 scoping.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

try:
    from config import BIORED_FILES, CDR_FILES, DATA_ROOT
    from utils import parse_pubtator
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import BIORED_FILES, CDR_FILES, DATA_ROOT
    from utils import parse_pubtator


_MESH_RE = re.compile(r"^(?:MESH:)?(D\d{6,7}|C\d{6,7})$")

_BIORED_TYPE_TO_CLASS = {
    "ChemicalEntity": "chemical",
    "DiseaseOrPhenotypicFeature": "disease",
}
_CDR_TYPE_TO_CLASS = {
    "Chemical": "chemical",
    "Disease": "disease",
}
# NCBI Disease has DiseaseClass / SpecificDisease / Modifier / CompositeMention.
_NCBI_DISEASE_TYPES = {"DiseaseClass", "SpecificDisease", "Modifier", "CompositeMention"}


def _normalize_mesh(raw: str) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned or cleaned in {"-", "-1"}:
        return None
    m = _MESH_RE.match(cleaned)
    return m.group(1) if m else None


def _split_composite(raw: str) -> Iterator[str]:
    for part in re.split(r"[,;|]", raw):
        part = part.strip()
        if part:
            yield part


def _iter_pubtator_mesh(path: Path, type_to_class: Dict[str, str]) -> Iterator[Tuple[str, str]]:
    if not path.exists():
        print(f"[mesh_harvest] missing corpus file: {path}", file=sys.stderr)
        return
    for doc in parse_pubtator(path):
        for ent in doc["entities"]:
            entity_class = type_to_class.get(ent["entity_type"])
            if entity_class is None:
                continue
            for token in _split_composite(ent.get("identifier_raw", "") or ""):
                mesh = _normalize_mesh(token)
                if mesh is not None:
                    yield entity_class, mesh


def harvest_corpus(name: str, files: Iterable[Path], type_to_class: Dict[str, str]):
    counts: Dict[Tuple[str, str], int] = Counter()
    for path in files:
        for entity_class, mesh in _iter_pubtator_mesh(path, type_to_class):
            counts[(entity_class, mesh)] += 1
    return counts


def _resolve_optional_pubtator(*candidates: Path) -> list[Path]:
    return [p for p in candidates if p.exists()]


def harvest_all() -> Dict[str, Dict[Tuple[str, str], int]]:
    """Return {corpus_name: {(entity_class, mesh_id): freq}}."""
    out: Dict[str, Dict[Tuple[str, str], int]] = {}

    out["BioRED"] = harvest_corpus("BioRED", BIORED_FILES.values(), _BIORED_TYPE_TO_CLASS)
    out["BC5CDR"] = harvest_corpus("BC5CDR", CDR_FILES.values(), _CDR_TYPE_TO_CLASS)

    ncbi_dir = DATA_ROOT / "NCBI_Disease"
    ncbi_files = _resolve_optional_pubtator(
        ncbi_dir / "NCBItrainset_corpus.txt",
        ncbi_dir / "NCBIdevelopset_corpus.txt",
        ncbi_dir / "NCBItestset_corpus.txt",
    )
    if ncbi_files:
        ncbi_map = {t: "disease" for t in _NCBI_DISEASE_TYPES}
        out["NCBI_Disease"] = harvest_corpus("NCBI_Disease", ncbi_files, ncbi_map)
    else:
        out["NCBI_Disease"] = {}

    nlm_dir = DATA_ROOT / "NLM-Chem"
    nlm_files = _resolve_optional_pubtator(
        nlm_dir / "BC7T2-NLMChem-corpus-train.PubTator",
        nlm_dir / "BC7T2-NLMChem-corpus-dev.PubTator",
        nlm_dir / "BC7T2-NLMChem-corpus-test.PubTator",
    )
    if nlm_files:
        nlm_map = {"Chemical": "chemical"}
        out["NLM-Chem"] = harvest_corpus("NLM-Chem", nlm_files, nlm_map)
    else:
        out["NLM-Chem"] = {}

    return out


def aggregate(per_corpus: Dict[str, Dict[Tuple[str, str], int]]):
    """Collapse to {mesh_id: {'frequency', 'entity_classes', 'corpora'}}."""
    agg: Dict[str, dict] = defaultdict(
        lambda: {"frequency": 0, "entity_classes": set(), "corpora": set()}
    )
    for corpus, counts in per_corpus.items():
        for (entity_class, mesh_id), freq in counts.items():
            row = agg[mesh_id]
            row["frequency"] += freq
            row["entity_classes"].add(entity_class)
            if freq > 0:
                row["corpora"].add(corpus)
    return agg


if __name__ == "__main__":
    per_corpus = harvest_all()
    for corpus, counts in per_corpus.items():
        print(f"{corpus}: {len(counts):,} (entity_class, mesh_id) keys, "
              f"total mentions={sum(counts.values()):,}")
    agg = aggregate(per_corpus)
    print(f"Aggregate unique MeSH IDs: {len(agg):,}")
