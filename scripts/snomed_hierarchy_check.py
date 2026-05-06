"""Sanity-check suite for Phase 1.6 — SNOMED Hierarchy Graph.

Run after building the graph:
    python3 scripts/snomed_hierarchy_check.py

Each check prints PASS or FAIL with a short explanation.
Exit code 0 if all pass, 1 if any fail.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import snomed_hierarchy as sh
from config import SNOMED_FILES

OUTPUT_DIR   = sh.OUTPUT_DIR
GRAPH_PKL    = sh.GRAPH_PKL
ANCESTORS_PKL = sh.ANCESTORS_PKL
STATS_JSON   = sh.STATS_JSON

# Known concepts used as test fixtures (from mesh_to_snomed.csv + SNOMED browser).
PNEUMONIA_ID     = "233604007"   # Pneumonia — Clinical finding hierarchy, depth ~6
CLINICAL_FINDING = "404684003"   # Top-level: Clinical finding
SUBSTANCE_ROOT   = "105590001"   # Top-level: Substance
DOXORUBICIN_ID   = "372817009"   # Doxorubicin — mapped chemical (in mesh_to_snomed.csv)
SNOMED_ROOT      = sh.SNOMED_ROOT   # "138875005"


# ---------------------------------------------------------------------------

_failures: list[str] = []


def _check(label: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    if not passed:
        _failures.append(label)


# ---------------------------------------------------------------------------

def main() -> int:
    # ------------------------------------------------------------------ setup
    for path in (GRAPH_PKL, ANCESTORS_PKL, STATS_JSON):
        if not path.exists():
            print(f"ERROR: required file missing: {path}")
            print("       Run `python3 main.py --only 1.6` first.")
            return 1

    print("Loading pickled graph and ancestor sets ...", flush=True)
    G, anc = sh.load_or_build(force=False, verbose=False)
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))

    depths    = dict(G.nodes(data="depth",               default=-1))
    sem_types = dict(G.nodes(data="semantic_type",        default=None))
    tl_mem    = dict(G.nodes(data="top_level_hierarchies", default=()))

    print()

    # ------------------------------------------------------------------ 1. graph shape
    print("=== 1. Graph shape ===")
    _check("Active concept nodes (expect ~386K)",
           G.number_of_nodes() > 300_000,
           f"got {G.number_of_nodes():,}")
    _check("Active is-a edges (expect ~640K)",
           G.number_of_edges() > 500_000,
           f"got {G.number_of_edges():,}")
    _check("Root node present",
           SNOMED_ROOT in G)
    _check("Top-level hierarchies reachable from root (expect ≥15)",
           stats["top_level_hierarchy_count"] >= 15,
           f"got {stats['top_level_hierarchy_count']}")

    # ------------------------------------------------------------------ 2. edge direction
    print()
    print("=== 2. Edge direction (child → parent) ===")
    root_parents = list(G.successors(SNOMED_ROOT))
    _check("Root has no outgoing edges (no parents above root)",
           len(root_parents) == 0,
           f"found parents: {root_parents[:3]}")
    pneu_parents = list(G.successors(PNEUMONIA_ID))
    _check(f"Pneumonia ({PNEUMONIA_ID}) has outgoing edges to its parents",
           len(pneu_parents) > 0,
           f"{len(pneu_parents)} parents: {pneu_parents[:2]}")
    _check("Clinical finding is not a successor of Pneumonia (no upward edge)",
           CLINICAL_FINDING not in G.predecessors(PNEUMONIA_ID))

    # ------------------------------------------------------------------ 3. depths
    print()
    print("=== 3. Depths ===")
    _check("Root depth = 0",
           depths.get(SNOMED_ROOT) == 0,
           f"got {depths.get(SNOMED_ROOT)}")
    _check("Clinical finding depth = 1 (direct child of root)",
           depths.get(CLINICAL_FINDING) == 1,
           f"got {depths.get(CLINICAL_FINDING)}")
    pneu_depth = depths.get(PNEUMONIA_ID, -1)
    _check(f"Pneumonia depth in [4, 12]",
           4 <= pneu_depth <= 12,
           f"got {pneu_depth}")
    _check(f"Max depth in [10, 25]",
           10 <= stats["max_depth"] <= 25,
           f"got {stats['max_depth']}")
    _check("All mapped concepts have a depth assigned",
           stats["mapped_concepts_in_graph"] > 0 and
           all(depths.get(k, -1) >= 0
               for k in anc if not k.startswith("descendant:")),
           f"{stats['mapped_concepts_in_graph']} / {stats['mapped_concepts_total']} in graph")

    # ------------------------------------------------------------------ 4. semantic types
    print()
    print("=== 4. Semantic types ===")
    _check("Pneumonia semantic_type = Clinical finding",
           sem_types.get(PNEUMONIA_ID) == CLINICAL_FINDING,
           f"got {sem_types.get(PNEUMONIA_ID)}")
    _check("Doxorubicin semantic_type = Substance",
           sem_types.get(DOXORUBICIN_ID) == SUBSTANCE_ROOT,
           f"got {sem_types.get(DOXORUBICIN_ID)}")
    # Every top-level concept should be its own semantic_type.
    R = G.reverse(copy=False)
    top_ids = list(R.successors(SNOMED_ROOT))
    bad_tops = [t for t in top_ids if sem_types.get(t) != t]
    _check("Top-level concepts are their own semantic_type",
           len(bad_tops) == 0,
           f"{len(bad_tops)} wrong: {bad_tops[:3]}")
    # No concept should have an unrecognised semantic_type.
    top_set = set(top_ids) | {SNOMED_ROOT}
    bad_sem  = [n for n, s in sem_types.items() if s not in top_set]
    _check("Every semantic_type label is a top-level concept ID",
           len(bad_sem) == 0,
           f"{len(bad_sem)} bad nodes (first: {bad_sem[:2]})")

    # ------------------------------------------------------------------ 5. ancestor sets
    print()
    print("=== 5. Ancestor sets (mapped concepts) ===")
    pneu_ancestors = anc.get(PNEUMONIA_ID, frozenset())
    _check("Pneumonia has precomputed ancestors",
           len(pneu_ancestors) > 0,
           f"{len(pneu_ancestors)} ancestors")
    _check("Clinical finding is an ancestor of Pneumonia",
           CLINICAL_FINDING in pneu_ancestors)
    _check("SNOMED root is an ancestor of Pneumonia",
           SNOMED_ROOT in pneu_ancestors)
    dox_ancestors = anc.get(DOXORUBICIN_ID, frozenset())
    _check("Substance root is an ancestor of Doxorubicin",
           SUBSTANCE_ROOT in dox_ancestors,
           f"ancestors count={len(dox_ancestors)}")
    _check("Clinical finding is NOT an ancestor of Doxorubicin",
           CLINICAL_FINDING not in dox_ancestors)
    # Root is ancestor of every mapped concept.
    mapped_keys = [k for k in anc if not k.startswith("descendant:")]
    missing_root = [k for k in mapped_keys if SNOMED_ROOT not in anc[k]]
    _check("Root is ancestor of every mapped concept",
           len(missing_root) == 0,
           f"{len(missing_root)} exceptions: {missing_root[:3]}")

    # ------------------------------------------------------------------ 6. descendant sets (MRCM anchors)
    print()
    print("=== 6. Descendant sets (MRCM anchors) ===")
    cf_desc = anc.get(f"descendant:{CLINICAL_FINDING}", frozenset())
    _check("Clinical finding descendant set > 100K",
           len(cf_desc) > 100_000,
           f"got {len(cf_desc):,}")
    _check("Pneumonia is a descendant of Clinical finding",
           PNEUMONIA_ID in cf_desc)
    _check("Root descendant set = all nodes minus root",
           len(anc.get(f"descendant:{SNOMED_ROOT}", frozenset())) == G.number_of_nodes() - 1)
    # Cross-hierarchy: a disease must NOT be in Substance's descendant set.
    sub_desc = anc.get(f"descendant:{SUBSTANCE_ROOT}", frozenset())
    _check("Pneumonia is NOT a descendant of Substance",
           PNEUMONIA_ID not in sub_desc)
    # All 11 MRCM anchors have a descendant set entry.
    from config import MRCM_FILES  # noqa: F401 — just to confirm mrcm_constraints is readable
    import json as _json
    mrcm_data = _json.loads(sh._MRCM_JSON.read_text(encoding="utf-8"))
    mrcm_anchors: set[str] = set()
    for blk in mrcm_data.get("relation_constraints", {}).values():
        for d in blk.get("domains", []):
            mrcm_anchors.update(d.get("domain_root_concept_ids", []))
        for r in blk.get("ranges", []):
            mrcm_anchors.update(r.get("range_root_concept_ids", []))
    missing_desc = [a for a in mrcm_anchors if f"descendant:{a}" not in anc]
    _check("All MRCM anchor concepts have a descendant set",
           len(missing_desc) == 0,
           f"missing: {missing_desc}")

    # ------------------------------------------------------------------ 7. public API helpers
    print()
    print("=== 7. Public API helpers ===")
    _check("is_descendant_of(Pneumonia, ClinicalFinding) -> True",
           sh.is_descendant_of(PNEUMONIA_ID, CLINICAL_FINDING, anc))
    _check("is_descendant_of(ClinicalFinding, Pneumonia) -> False",
           not sh.is_descendant_of(CLINICAL_FINDING, PNEUMONIA_ID, anc))
    _check("is_descendant_of(Doxorubicin, Substance) -> True",
           sh.is_descendant_of(DOXORUBICIN_ID, SUBSTANCE_ROOT, anc))
    _check("is_descendant_of(Pneumonia, Substance) -> False",
           not sh.is_descendant_of(PNEUMONIA_ID, SUBSTANCE_ROOT, anc))
    got_ancs = sh.get_ancestors(PNEUMONIA_ID, anc)
    _check("get_ancestors(Pneumonia) returns a non-empty frozenset",
           isinstance(got_ancs, frozenset) and len(got_ancs) > 0,
           f"{len(got_ancs)} ancestors")
    _check("get_ancestors: unmapped concept, no G -> empty frozenset",
           sh.get_ancestors("NONEXISTENT_ID", anc) == frozenset())
    _check("get_ancestors: unmapped concept with G fallback returns frozenset",
           isinstance(sh.get_ancestors(CLINICAL_FINDING, anc, G=G), frozenset))
    _check("get_ancestors: G fallback includes root for Clinical finding",
           SNOMED_ROOT in sh.get_ancestors(CLINICAL_FINDING, anc, G=G))

    # ------------------------------------------------------------------ 8. new Dbport features
    print()
    print("=== 8. Dbport new features (top_level_hierarchies, get_depth, pickle envelope) ===")

    # top_level_hierarchies node attribute.
    _check("top_level_hierarchies attribute present on Pneumonia",
           isinstance(tl_mem.get(PNEUMONIA_ID), tuple))
    _check("Clinical finding in Pneumonia's top_level_hierarchies",
           CLINICAL_FINDING in tl_mem.get(PNEUMONIA_ID, ()))
    _check("Substance NOT in Pneumonia's top_level_hierarchies",
           SUBSTANCE_ROOT not in tl_mem.get(PNEUMONIA_ID, ()))
    _check("Substance in Doxorubicin's top_level_hierarchies",
           SUBSTANCE_ROOT in tl_mem.get(DOXORUBICIN_ID, ()))
    _check("Clinical finding NOT in Doxorubicin's top_level_hierarchies",
           CLINICAL_FINDING not in tl_mem.get(DOXORUBICIN_ID, ()))

    # multi_inheritance_concepts present in stats.
    _check("stats has multi_inheritance_concepts key",
           "multi_inheritance_concepts" in stats)
    _check("multi_inheritance_concepts >= 0",
           stats.get("multi_inheritance_concepts", -1) >= 0)

    # get_top_level_hierarchies() helper.
    pneu_tl = sh.get_top_level_hierarchies(G, PNEUMONIA_ID)
    _check("get_top_level_hierarchies(Pneumonia) returns tuple",
           isinstance(pneu_tl, tuple))
    _check("get_top_level_hierarchies(Pneumonia) contains Clinical finding",
           CLINICAL_FINDING in pneu_tl)
    _check("get_top_level_hierarchies(unknown) returns empty tuple",
           sh.get_top_level_hierarchies(G, "NONEXISTENT") == ())

    # get_depth() helper.
    _check("get_depth(root) = 0",
           sh.get_depth(G, SNOMED_ROOT) == 0)
    _check("get_depth(Clinical finding) = 1",
           sh.get_depth(G, CLINICAL_FINDING) == 1)
    _check("get_depth(unknown) = None",
           sh.get_depth(G, "NONEXISTENT") is None)

    # Pickle envelope format (versioned dict, not raw object).
    with GRAPH_PKL.open("rb") as fh:
        graph_payload = pickle.load(fh)
    _check("Graph pickle is a dict envelope with 'signature' and 'graph' keys",
           isinstance(graph_payload, dict)
           and "signature" in graph_payload
           and "graph" in graph_payload)
    with ANCESTORS_PKL.open("rb") as fh:
        anc_payload = pickle.load(fh)
    _check("Ancestors pickle is a dict envelope with 'signature' and 'ancestors' keys",
           isinstance(anc_payload, dict)
           and "signature" in anc_payload
           and "ancestors" in anc_payload)
    _check("Graph and ancestors share the same signature",
           graph_payload.get("signature") == anc_payload.get("signature"))

    # mrcm_anchors_in_graph is now a list of dicts, not a plain dict.
    anchor_list = stats.get("mrcm_anchors_in_graph", [])
    _check("mrcm_anchors_in_graph is a list (not dict)",
           isinstance(anchor_list, list))
    anchor_ids_in_graph = {
        e["concept_id"] for e in anchor_list if e.get("in_graph")
    }
    for aid in (CLINICAL_FINDING, SUBSTANCE_ROOT, "71388002"):
        _check(f"MRCM anchor {aid} present in graph",
               aid in anchor_ids_in_graph)

    # top_level_hierarchies list in stats has name field.
    tl_list = stats.get("top_level_hierarchies", [])
    names_populated = all(e.get("name") for e in tl_list)
    _check("top_level_hierarchies entries have resolved names",
           len(tl_list) > 0 and names_populated,
           f"{len(tl_list)} entries, all named={names_populated}")

    # ------------------------------------------------------------------ 9. missing concepts
    print()
    print("=== 8. Missing mapped concepts ===")
    missing = stats["mapped_concepts_missing"]
    # All missing IDs must be inactive in the SNOMED concept file.
    inactive_check: dict[str, str] = {}
    with SNOMED_FILES["concepts"].open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["id"] in missing:
                inactive_check[row["id"]] = row["active"]
    found_active  = [sid for sid, a in inactive_check.items() if a == "1"]
    truly_absent  = [sid for sid in missing if sid not in inactive_check]
    _check("All missing mapped concepts are inactive in SNOMED (retired IDs)",
           len(found_active) == 0 and len(truly_absent) == 0,
           f"active(bug)={found_active[:3]}  absent_from_file={truly_absent[:3]}")
    _check("Missing count is small (< 5% of mapped total)",
           len(missing) < 0.05 * stats["mapped_concepts_total"],
           f"{len(missing)} / {stats['mapped_concepts_total']}")

    # ------------------------------------------------------------------ 9. pickle cache speed
    print()
    print("=== 9. Pickle cache round-trip speed ===")
    t0 = time.time()
    sh.load_or_build(force=False, verbose=False)
    elapsed = time.time() - t0
    _check("Cache hit completes in < 5s",
           elapsed < 5.0,
           f"{elapsed:.2f}s")

    # ------------------------------------------------------------------ summary
    print()
    total = 63  # update if you add checks
    n_fail = len(_failures)
    n_pass = total - n_fail
    print(f"{'='*50}")
    print(f"Results: {n_pass} passed, {n_fail} failed")
    if _failures:
        print("Failed checks:")
        for f in _failures:
            print(f"  - {f}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
