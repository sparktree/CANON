"""CANON top-level entry point.

Runs every implemented phase step end-to-end. Update this file as new steps
land so a single command exercises the whole pipeline.

Currently implemented:
    Phase 1.1 -- UMLS RRF parser + pickle cache         (umls_query.py)
    Phase 1.2 -- MeSH -> SNOMED mapping pipeline        (mesh_to_snomed.py)
    Phase 1.3 -- Non-MeSH vocabulary scoping audit      (entity_scope.py + scope_audit.py)
    Phase 1.4 -- Relation schema alignment table        (relation_schema.py)
    Phase 1.5 -- MRCM constraint dictionary             (mrcm.py)
    Phase 1.6 -- SNOMED hierarchy graph                 (snomed_hierarchy.py)
    Phase 1.7 -- Active-release mapping verification    (mapping_verify.py)
    Phase 2.1 -- Unified annotation format + converters (unified_format.py + corpus_convert.py)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import corpus_convert  # noqa: E402
import entity_scope  # noqa: E402
import mapping_verify  # noqa: E402
import mesh_to_snomed  # noqa: E402
import mrcm  # noqa: E402
import relation_schema  # noqa: E402
import scope_audit  # noqa: E402
import snomed_hierarchy  # noqa: E402
import umls_query  # noqa: E402


def _banner(text: str) -> None:
    bar = "=" * len(text)
    print(f"\n{bar}\n{text}\n{bar}", flush=True)


def step_1_1(force_reparse: bool) -> None:
    _banner("Phase 1.1 -- UMLS RRF parser + pickle cache")
    t0 = time.time()
    umls_query.preload(force=force_reparse)
    print(
        f"[1.1] CUIs={len(umls_query.cui_to_atoms):,}  "
        f"(sab,code) keys={len(umls_query.code_to_cuis):,}  "
        f"CUIs-with-rels={len(umls_query.cui_to_rels):,}  "
        f"CUIs-with-stys={len(umls_query.cui_to_stys):,}  "
        f"MRMAP from-keys={len(umls_query.mrmap_entries):,}"
    )
    print(f"[1.1] elapsed {time.time() - t0:.1f}s")


def step_1_2() -> None:
    _banner("Phase 1.2 -- MeSH -> SNOMED concept mapping")
    t0 = time.time()
    result = mesh_to_snomed.build_mapping(verbose=True)
    mapped = len(result["mapping_rows"])
    unmapped = len(result["unmapped_rows"])
    total = mapped + unmapped
    pct = (mapped / total * 100) if total else 0.0
    print(f"[1.2] mapped {mapped:,} / {total:,} ({pct:.1f}%); unmapped {unmapped:,}")
    print(f"[1.2] outputs in {mesh_to_snomed.OUTPUT_DIR}")
    print(f"[1.2] elapsed {time.time() - t0:.1f}s")


def step_1_3() -> None:
    _banner("Phase 1.3 -- Non-MeSH vocabulary scoping audit")
    t0 = time.time()
    in_scope = sum(1 for s in entity_scope.iter_specs() if s.snomed_normalized)
    out_scope = sum(1 for s in entity_scope.iter_specs() if not s.snomed_normalized)
    print(f"[1.3] registry: {in_scope} SNOMED-normalized types, {out_scope} NER-only types")
    scope_audit.run(verbose=True)
    print(f"[1.3] elapsed {time.time() - t0:.1f}s")


def step_1_4() -> None:
    _banner("Phase 1.4 -- Relation schema alignment")
    t0 = time.time()
    rows = list(relation_schema.iter_rows())
    tier1 = sum(1 for r in rows if r.tier == 1)
    tier2 = sum(1 for r in rows if r.tier == 2)
    print(f"[1.4] {len(rows)} mapping rows  ({tier1} Tier-1, {tier2} Tier-2)")
    out = relation_schema.dump_csv()
    print(f"[1.4] CSV written to {out}")
    print(f"[1.4] elapsed {time.time() - t0:.1f}s")


def step_1_5() -> None:
    _banner("Phase 1.5 -- MRCM constraint dictionary")
    t0 = time.time()
    out = mrcm.main(verbose=True)
    print(f"[1.5] JSON written to {out}")
    print(f"[1.5] elapsed {time.time() - t0:.1f}s")


def step_1_6(force_reparse: bool = False) -> None:
    _banner("Phase 1.6 -- SNOMED Hierarchy Graph")
    t0 = time.time()
    out = snomed_hierarchy.main(force=force_reparse, verbose=True)
    print(f"[1.6] stats written to {out}")
    print(f"[1.6] elapsed {time.time() - t0:.1f}s")


def step_1_7() -> None:
    _banner("Phase 1.7 -- Active-release mapping verification")
    t0 = time.time()
    summary = mapping_verify.verify(verbose=True)
    if summary["mapped_rows_inactive"]:
        print(
            f"[1.7] {summary['mapped_rows_inactive']} mapped concepts are retired in the loaded "
            f"SNOMED release; Phase 2.2 must apply a policy (drop / redirect / keep)."
        )
    print(f"[1.7] elapsed {time.time() - t0:.1f}s")


def step_2_1() -> None:
    _banner("Phase 2.1 -- Unified annotation format + converters")
    t0 = time.time()
    summary = corpus_convert.convert_all(verbose=True)
    converted = sum(
        1 for v in summary["corpora"].values() if v.get("status") == "converted"
    )
    print(f"[2.1] {converted} corpus paths converted to unified JSONL")
    print(f"[2.1] elapsed {time.time() - t0:.1f}s")


STEPS = {
    "1.1": step_1_1,
    "1.2": step_1_2,
    "1.3": step_1_3,
    "1.4": step_1_4,
    "1.5": step_1_5,
    "1.6": step_1_6,
    "1.7": step_1_7,
    "2.1": step_2_1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all implemented CANON steps.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(STEPS),
        help="Run only the listed step IDs (default: all).",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Ignore the pickled UMLS cache and re-parse the RRFs.",
    )
    args = parser.parse_args()

    selected = args.only or sorted(STEPS)
    overall = time.time()
    for step_id in selected:
        if step_id in ("1.1", "1.6"):
            STEPS[step_id](force_reparse=args.force_reparse)  # type: ignore[call-arg]
        else:
            STEPS[step_id]()  # type: ignore[operator]
    print(f"\n[main] all selected steps done in {time.time() - overall:.1f}s")


if __name__ == "__main__":
    main()
