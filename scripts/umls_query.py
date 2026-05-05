"""In-memory UMLS RRF query module (CANON Phase 1.1).

Parses the four RRF files (MRCONSO, MRREL, MRSTY, MRMAP) into in-memory
dictionaries on first use and exposes query helpers used by the rest of the
pipeline. A pickle cache is written next to the RRF files so subsequent runs
skip the 30-60s parse.

Public API:
    preload()                       -- force the parse/load cycle now
    get_cuis_for_mesh(mesh_id)      -- list[str]
    get_snomed_for_cui(cui)         -- list[dict]  (SNOMEDCT_US atoms)
    get_relations(cui)              -- list[dict]
    get_semantic_types(cui)         -- list[str]
    get_curated_mapping(mesh_id)    -- list[dict]  (MRMAP entries)

The underlying dictionaries (code_to_cuis, cui_to_atoms, cui_to_rels,
cui_to_stys, mrmap_entries) are also exposed as module attributes after load.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from config import UMLS_FILES, UMLS_META_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import UMLS_FILES, UMLS_META_DIR


CACHE_VERSION = 1
CACHE_PATH = Path(
    os.getenv("UMLS_CACHE_PATH", str(UMLS_META_DIR / ".canon_umls_cache.pkl"))
)

# RRF column counts (used to defensively handle the trailing pipe).
_MRCONSO_COLS = 18
_MRREL_COLS = 16
_MRSTY_COLS = 6
_MRMAP_COLS = 26

# Module-level dictionaries. Populated by preload(); empty until then.
code_to_cuis: Dict[Tuple[str, str], List[str]] = {}
cui_to_atoms: Dict[str, List[dict]] = {}
cui_to_rels: Dict[str, List[dict]] = {}
cui_to_stys: Dict[str, List[str]] = {}
mrmap_entries: Dict[Tuple[str, str], List[dict]] = {}

_loaded = False


def _iter_rrf(path: Path, expected_columns: int):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="|")
        for row in reader:
            if not row:
                continue
            if len(row) == expected_columns + 1 and row[-1] == "":
                row = row[:-1]
            elif len(row) > expected_columns:
                row = row[:expected_columns]
            elif len(row) < expected_columns:
                row = row + [""] * (expected_columns - len(row))
            yield row


def _signature(paths: List[Path]) -> List[Tuple[str, int, float]]:
    return [(p.name, p.stat().st_size, p.stat().st_mtime) for p in paths]


def _parse_mrconso(path: Path) -> None:
    for r in _iter_rrf(path, _MRCONSO_COLS):
        cui = r[0]
        sab = r[11]
        code = r[13]
        if not cui or not sab or not code or code == "NOCODE":
            continue
        atom = {"sab": sab, "code": code, "str": r[14], "tty": r[12]}
        cui_to_atoms.setdefault(cui, []).append(atom)
        bucket = code_to_cuis.setdefault((sab, code), [])
        if cui not in bucket:
            bucket.append(cui)


def _parse_mrrel(path: Path) -> None:
    for r in _iter_rrf(path, _MRREL_COLS):
        cui1 = r[0]
        if not cui1:
            continue
        cui_to_rels.setdefault(cui1, []).append(
            {"rel": r[3], "rela": r[7], "cui2": r[4], "sab": r[10]}
        )


def _parse_mrsty(path: Path) -> None:
    for r in _iter_rrf(path, _MRSTY_COLS):
        cui = r[0]
        sty = r[3]
        if not cui or not sty:
            continue
        bucket = cui_to_stys.setdefault(cui, [])
        if sty not in bucket:
            bucket.append(sty)


def _parse_mrmap(path: Path) -> None:
    for r in _iter_rrf(path, _MRMAP_COLS):
        from_code = r[8] or r[6]  # FROMEXPR else FROMID
        to_code = r[16] or r[14]  # TOEXPR else TOID
        mapsetsab = r[1]
        if not from_code:
            continue
        entry = {
            "to_sab": mapsetsab,  # the mapset SAB stands in for target vocab
            "to_code": to_code,
            "maprule": r[20],
            "rel": r[12],
            "rela": r[13],
            "maptype": r[22],
        }
        mrmap_entries.setdefault((mapsetsab, from_code), []).append(entry)


def _parse_all() -> None:
    code_to_cuis.clear()
    cui_to_atoms.clear()
    cui_to_rels.clear()
    cui_to_stys.clear()
    mrmap_entries.clear()

    print("[umls_query] parsing MRCONSO...", flush=True)
    _parse_mrconso(UMLS_FILES["mrconso"])
    print(f"[umls_query]   {len(cui_to_atoms):,} CUIs, {len(code_to_cuis):,} (sab,code) keys")

    print("[umls_query] parsing MRREL...", flush=True)
    _parse_mrrel(UMLS_FILES["mrrel"])
    print(f"[umls_query]   {sum(len(v) for v in cui_to_rels.values()):,} relations")

    print("[umls_query] parsing MRSTY...", flush=True)
    _parse_mrsty(UMLS_FILES["mrsty"])
    print(f"[umls_query]   {len(cui_to_stys):,} CUIs with semantic types")

    if UMLS_FILES["mrmap"].exists():
        print("[umls_query] parsing MRMAP...", flush=True)
        _parse_mrmap(UMLS_FILES["mrmap"])
        print(f"[umls_query]   {len(mrmap_entries):,} MRMAP from-keys")
    else:
        print(f"[umls_query] MRMAP not found at {UMLS_FILES['mrmap']} -- skipping")


def _save_cache(signature: List[Tuple[str, int, float]]) -> None:
    payload = {
        "version": CACHE_VERSION,
        "signature": signature,
        "code_to_cuis": code_to_cuis,
        "cui_to_atoms": cui_to_atoms,
        "cui_to_rels": cui_to_rels,
        "cui_to_stys": cui_to_stys,
        "mrmap_entries": mrmap_entries,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_PATH)
    print(f"[umls_query] wrote cache -> {CACHE_PATH}")


def _load_cache(signature: List[Tuple[str, int, float]]) -> bool:
    if not CACHE_PATH.exists():
        return False
    try:
        with CACHE_PATH.open("rb") as fh:
            payload = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        print(f"[umls_query] cache unreadable ({exc}); re-parsing")
        return False
    if payload.get("version") != CACHE_VERSION:
        return False
    if payload.get("signature") != signature:
        return False
    code_to_cuis.update(payload["code_to_cuis"])
    cui_to_atoms.update(payload["cui_to_atoms"])
    cui_to_rels.update(payload["cui_to_rels"])
    cui_to_stys.update(payload["cui_to_stys"])
    mrmap_entries.update(payload["mrmap_entries"])
    return True


def preload(force: bool = False) -> None:
    """Populate in-memory dictionaries from cache if fresh, else parse RRFs."""
    global _loaded
    if _loaded and not force:
        return

    required = [UMLS_FILES["mrconso"], UMLS_FILES["mrrel"], UMLS_FILES["mrsty"]]
    if UMLS_FILES["mrmap"].exists():
        required.append(UMLS_FILES["mrmap"])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "UMLS RRF files missing:\n" + "\n".join(f"- {m}" for m in missing)
        )

    signature = _signature(required)
    t0 = time.time()
    if not force and _load_cache(signature):
        print(f"[umls_query] loaded cache in {time.time() - t0:.1f}s")
    else:
        _parse_all()
        _save_cache(signature)
        print(f"[umls_query] parsed RRFs in {time.time() - t0:.1f}s")
    _loaded = True


def _ensure_loaded() -> None:
    if not _loaded:
        preload()


def get_cuis_for_mesh(mesh_id: str) -> List[str]:
    _ensure_loaded()
    return list(code_to_cuis.get(("MSH", mesh_id), []))


def get_snomed_for_cui(cui: str) -> List[dict]:
    _ensure_loaded()
    return [a for a in cui_to_atoms.get(cui, []) if a["sab"] == "SNOMEDCT_US"]


def get_relations(cui: str) -> List[dict]:
    _ensure_loaded()
    return list(cui_to_rels.get(cui, []))


def get_semantic_types(cui: str) -> List[str]:
    _ensure_loaded()
    return list(cui_to_stys.get(cui, []))


def get_curated_mapping(mesh_id: str) -> List[dict]:
    """Return MRMAP entries whose FROM code is mesh_id (typically MSH→SNOMED)."""
    _ensure_loaded()
    out: List[dict] = []
    for (mapset_sab, from_code), entries in mrmap_entries.items():
        if from_code == mesh_id:
            out.extend(entries)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build/refresh the UMLS in-memory cache.")
    p.add_argument("--force", action="store_true", help="Re-parse even if cache is valid.")
    p.add_argument("--probe", metavar="MESH_ID", help="Probe a MeSH id after loading.")
    args = p.parse_args()

    preload(force=args.force)
    if args.probe:
        cuis = get_cuis_for_mesh(args.probe)
        print(f"CUIs for MeSH {args.probe}: {cuis}")
        for c in cuis:
            print(f"  {c} stys={get_semantic_types(c)}")
            for atom in get_snomed_for_cui(c):
                print(f"    SNOMED {atom['code']} ({atom['tty']}): {atom['str']}")
