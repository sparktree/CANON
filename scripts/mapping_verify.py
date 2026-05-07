"""Verify Phase 1.2 mappings against the active SNOMED CT release (CANON Phase 1.7).

Cross-checks every SNOMED concept ID emitted by the four-priority mapper
against the set of active concepts in the loaded RF2 release. SNOMED
inactivates a small percentage of concepts each release; UMLS atoms can
preserve those legacy IDs even though SNOMED itself has retired them.
Training a CN head on a retired concept is silently broken: the CSP solver
cannot verify it (not in Phase 1.6's hierarchy graph), the bi-encoder
cannot embed it (no description in the active set), and the ancestor-match
metric returns the empty set.

This module produces:

    outputs/phase1/mesh_to_snomed_verified.csv  -- the original Phase 1.2
        table with one extra column ``snomed_active`` (true/false). Phase 2.2
        consumes this in place of the raw mapping.
    outputs/phase1/mesh_to_snomed_inactive.csv  -- inactive subset only,
        ordered by frequency, for the manual top-100 review queue.
    outputs/phase1/mapping_verification_summary.json -- counts + active
        concept set size + total mention impact, for the Phase 1 audit log.

The active concept set is read directly from
``sct2_Concept_Snapshot_*.txt`` (active=1 rows) so this step does NOT
depend on Phase 1.6's pickled graph and can run before or after it. The
distinction between active and inactive is what we expose; the *policy*
for inactive entries (drop / redirect via MRCUI / keep as-is) is a
Phase 2.2 decision and is intentionally left to that step.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterator, Set

try:
    from config import REPO_ROOT, SNOMED_FILES, relative_to_repo
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, SNOMED_FILES, relative_to_repo


OUTPUT_DIR = REPO_ROOT / "outputs" / "phase1"
INPUT_CSV = OUTPUT_DIR / "mesh_to_snomed.csv"
VERIFIED_CSV = OUTPUT_DIR / "mesh_to_snomed_verified.csv"
INACTIVE_CSV = OUTPUT_DIR / "mesh_to_snomed_inactive.csv"
SUMMARY_JSON = OUTPUT_DIR / "mapping_verification_summary.json"


def _iter_rf2(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(f"RF2 file missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            yield row


def load_active_concepts() -> Set[str]:
    """Return the set of SCTIDs marked active=1 in sct2_Concept_Snapshot."""
    out: Set[str] = set()
    for row in _iter_rf2(SNOMED_FILES["concepts"]):
        if row.get("active") == "1":
            out.add(row["id"])
    return out


def verify(verbose: bool = True) -> dict:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found; run Phase 1.2 (mesh_to_snomed.py) first."
        )

    if verbose:
        print("[1.7] loading active SNOMED concept set ...", flush=True)
    active = load_active_concepts()
    if verbose:
        print(f"[1.7] {len(active):,} active concepts in current SNOMED release")

    rows_verified: list[dict] = []
    inactive_rows: list[dict] = []
    inactive_mentions = 0
    total_mentions = 0

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            sid = (row.get("snomed_id") or "").strip()
            is_active = sid in active
            row_out = dict(row)
            row_out["snomed_active"] = "true" if is_active else "false"
            rows_verified.append(row_out)
            try:
                freq = int(row.get("frequency", "0") or 0)
            except ValueError:
                freq = 0
            total_mentions += freq
            if not is_active:
                inactive_mentions += freq
                inactive_rows.append(row_out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fields = fieldnames + ["snomed_active"]

    with VERIFIED_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        for r in rows_verified:
            w.writerow(r)

    inactive_rows.sort(key=lambda r: -int(r.get("frequency", "0") or 0))
    with INACTIVE_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        for r in inactive_rows:
            w.writerow(r)

    summary = {
        "active_concepts_in_release": len(active),
        "mapped_rows_total": len(rows_verified),
        "mapped_rows_active": len(rows_verified) - len(inactive_rows),
        "mapped_rows_inactive": len(inactive_rows),
        "inactive_pct_of_rows": (
            round(100.0 * len(inactive_rows) / len(rows_verified), 2)
            if rows_verified else 0.0
        ),
        "total_mentions": total_mentions,
        "inactive_mentions": inactive_mentions,
        "inactive_pct_of_mentions": (
            round(100.0 * inactive_mentions / total_mentions, 2)
            if total_mentions else 0.0
        ),
        "release_concepts_file": SNOMED_FILES["concepts"].name,
        "outputs": {
            "verified_csv": relative_to_repo(VERIFIED_CSV),
            "inactive_csv": relative_to_repo(INACTIVE_CSV),
        },
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(
            f"[1.7] mapped rows: {summary['mapped_rows_total']:,} total, "
            f"{summary['mapped_rows_active']:,} active "
            f"({100 - summary['inactive_pct_of_rows']:.1f}%); "
            f"{summary['mapped_rows_inactive']:,} inactive "
            f"({summary['inactive_pct_of_rows']:.1f}%)"
        )
        print(
            f"[1.7] mention impact: {inactive_mentions:,} of {total_mentions:,} "
            f"mentions ({summary['inactive_pct_of_mentions']:.2f}%) point at retired SCTIDs"
        )
        print(f"[1.7] verified -> {VERIFIED_CSV}")
        print(f"[1.7] inactive -> {INACTIVE_CSV}")
        print(f"[1.7] summary  -> {SUMMARY_JSON}")

    return summary


if __name__ == "__main__":
    verify(verbose=True)
