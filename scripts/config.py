"""Active-pipeline filesystem paths.

Only path-resolution lives here. Database / batch-size knobs needed by the
deprecated SQL staging path are in ``scripts/legacy_sql/config.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent


def relative_to_repo(path: Path) -> str:
    """Render *path* as a forward-slash string relative to REPO_ROOT or WORKSPACE_ROOT.

    Used by every producer that emits a path into a summary JSON so the
    generated artifacts are byte-identical across Windows / macOS / Linux
    after a fresh run.

    Anchor order:
      1. REPO_ROOT (CANON/)         -> e.g. "outputs/phase1/foo.csv"
      2. WORKSPACE_ROOT (CANON_root/) -> e.g. "Data/BioRED/Train.PubTator"
                                         (input data lives outside REPO_ROOT)
      3. absolute string fallback (graceful, never raises)
    """
    resolved = path.resolve()
    for anchor in (REPO_ROOT, WORKSPACE_ROOT):
        try:
            return str(resolved.relative_to(anchor)).replace("\\", "/")
        except ValueError:
            continue
    return str(resolved)


def _first_existing(candidates: list[Path]) -> Path:
    if not candidates:
        raise ValueError("Expected at least one candidate path.")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_data_root() -> Path:
    env_data_root = os.getenv("DATA_ROOT")
    if env_data_root:
        return Path(env_data_root).expanduser().resolve()
    return _first_existing(
        [
            (WORKSPACE_ROOT / "Data").resolve(),
            (REPO_ROOT / "Data").resolve(),
        ]
    )


DATA_ROOT = _resolve_data_root()


CDR_FILES: Dict[str, Path] = {
    "train": DATA_ROOT / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.PubTator.txt",
    "dev": DATA_ROOT / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_DevelopmentSet.PubTator.txt",
    "test": DATA_ROOT / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.PubTator.txt",
}

BIORED_FILES: Dict[str, Path] = {
    "train": DATA_ROOT / "BioRED" / "Train.PubTator",
    "dev": DATA_ROOT / "BioRED" / "Dev.PubTator",
    "test": DATA_ROOT / "BioRED" / "Test.PubTator",
}


def _first_glob_or_default(root: Path, pattern: str, default_name: str) -> Path:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else (root / default_name)


def _discover_snomed_snapshot(root: Path) -> Path | None:
    """Find the newest directory containing both Terminology/ and Refset/Metadata/ underneath a Snapshot/ folder."""
    candidates = []
    for snap in root.glob("**/Snapshot"):
        if (snap / "Terminology").is_dir() and (snap / "Refset" / "Metadata").is_dir():
            candidates.append(snap)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.as_posix(), reverse=True)[0]


_snomed_snapshot = _discover_snomed_snapshot(DATA_ROOT)
SNOMED_SNAPSHOT_ROOT = _snomed_snapshot if _snomed_snapshot is not None else (DATA_ROOT / "SnomedCT_SNAPSHOT_MISSING")

SNOMED_FILES: Dict[str, Path] = {
    "concepts": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Terminology",
        "sct2_Concept_Snapshot_*.txt",
        "sct2_Concept_Snapshot_MISSING.txt",
    ),
    "descriptions": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Terminology",
        "sct2_Description_Snapshot-*.txt",
        "sct2_Description_Snapshot_MISSING.txt",
    ),
    "text_definitions": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Terminology",
        "sct2_TextDefinition_Snapshot-*.txt",
        "sct2_TextDefinition_Snapshot_MISSING.txt",
    ),
    "relationships": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Terminology",
        "sct2_Relationship_Snapshot_*.txt",
        "sct2_Relationship_Snapshot_MISSING.txt",
    ),
    "stated_relationships": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Terminology",
        "sct2_StatedRelationship_Snapshot_*.txt",
        "sct2_StatedRelationship_Snapshot_MISSING.txt",
    ),
    "extended_map": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Refset" / "Map",
        "der2_*ExtendedMapSnapshot_*.txt",
        "der2_ExtendedMapSnapshot_MISSING.txt",
    ),
    "simple_map": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Refset" / "Map",
        "der2_*SimpleMapSnapshot_*.txt",
        "der2_SimpleMapSnapshot_MISSING.txt",
    ),
}

MRCM_FILES: Dict[str, Path] = {
    "domain": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Refset" / "Metadata",
        "der2_sssssssRefset_MRCMDomainSnapshot_*.txt",
        "der2_sssssssRefset_MRCMDomainSnapshot_MISSING.txt",
    ),
    "attribute_domain": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Refset" / "Metadata",
        "der2_cissccRefset_MRCMAttributeDomainSnapshot_*.txt",
        "der2_cissccRefset_MRCMAttributeDomainSnapshot_MISSING.txt",
    ),
    "attribute_range": _first_glob_or_default(
        SNOMED_SNAPSHOT_ROOT / "Refset" / "Metadata",
        "der2_ssccRefset_MRCMAttributeRangeSnapshot_*.txt",
        "der2_ssccRefset_MRCMAttributeRangeSnapshot_MISSING.txt",
    ),
}


BIOLINKBERT_DIR = Path(
    os.getenv(
        "CANON_BIOLINKBERT",
        str((WORKSPACE_ROOT / "BioLinkBERT").resolve()),
    )
)


UMLS_META_DIR = Path(
    os.getenv(
        "UMLS_META_DIR",
        str((DATA_ROOT / "UMLS_MeSH_and_SNOMED" / "2025AB" / "META").resolve()),
    )
)
UMLS_FILES: Dict[str, Path] = {
    "mrconso": UMLS_META_DIR / "MRCONSO.RRF",
    "mrrel": UMLS_META_DIR / "MRREL.RRF",
    "mrsty": UMLS_META_DIR / "MRSTY.RRF",
    "mrdef": UMLS_META_DIR / "MRDEF.RRF",
    "mrmap": UMLS_META_DIR / "MRMAP.RRF",
}
