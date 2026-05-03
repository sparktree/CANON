from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent


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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value}") from exc


BATCH_SIZE = _env_int("BATCH_SIZE", 5000)


@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = _env_int("PGPORT", 5432)
    dbname: str = os.getenv("PGDATABASE", "postgres")
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "")
    sslmode: str = os.getenv("PGSSLMODE", "prefer")


DB_CONFIG = PostgresConfig()


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


_snomed_snapshots = sorted(DATA_ROOT.glob("SnomedCT_*/Snapshot"), reverse=True)
SNOMED_SNAPSHOT_ROOT = _snomed_snapshots[0] if _snomed_snapshots else (DATA_ROOT / "SnomedCT_SNAPSHOT_MISSING")

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

UMLS_META_DIR = Path(
    os.getenv("UMLS_META_DIR", str((DATA_ROOT / "2025AB-full" / "META").resolve()))
)
UMLS_FILES: Dict[str, Path] = {
    "mrconso": UMLS_META_DIR / "MRCONSO.RRF",
    "mrrel": UMLS_META_DIR / "MRREL.RRF",
    "mrsty": UMLS_META_DIR / "MRSTY.RRF",
    "mrdef": UMLS_META_DIR / "MRDEF.RRF",
}
