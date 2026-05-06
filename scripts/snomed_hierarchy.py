"""SNOMED CT hierarchy graph — CANON Phase 1.6.

Loads the inferred SNOMED CT relationship file, keeps only active is-a edges,
and builds a NetworkX DiGraph where every edge points child -> parent (natural
SNOMED direction).  Three structures are precomputed and persisted to disk:

    snomed_hierarchy.pkl  — DiGraph with `depth` and `semantic_type` node attrs
    snomed_ancestors.pkl  — dict keyed two ways:
        "<snomed_id>"           -> frozenset of ancestor concept IDs
                                  (for every concept in mesh_to_snomed.csv)
        "descendant:<anchor_id>" -> frozenset of descendant concept IDs
                                  (for every MRCM anchor concept from
                                   mrcm_constraints.json)
    snomed_hierarchy_stats.json — human-readable summary for sanity checks

Downstream consumers:
    Phase 2.4  — ancestor similarity for hierarchical soft mapping
    Phase 3.5  — O(1) CSP domain/range membership via descendant sets
    Phase 4.2  — ancestor-match accuracy metric

Usage:
    python3 scripts/snomed_hierarchy.py          # build + save
    python3 scripts/snomed_hierarchy.py --force  # force rebuild (ignore cache)
    from snomed_hierarchy import load_or_build, get_ancestors, is_descendant_of
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Dict, FrozenSet, Iterator, Optional, Set

try:
    import networkx as nx
    from tqdm import tqdm
    from config import MRCM_FILES, REPO_ROOT, SNOMED_FILES
    import mrcm as _mrcm_mod
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import networkx as nx
    from tqdm import tqdm
    from config import MRCM_FILES, REPO_ROOT, SNOMED_FILES
    import mrcm as _mrcm_mod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IS_A_TYPE_ID = "116680003"   # SNOMED "Is a (attribute)" concept
SNOMED_ROOT  = "138875005"   # "SNOMED CT Concept (SNOMED RT+CTV3)"

CACHE_VERSION = 2  # bump whenever the pickled structure shape changes

OUTPUT_DIR   = REPO_ROOT / "outputs" / "phase1"
GRAPH_PKL    = OUTPUT_DIR / "snomed_hierarchy.pkl"
ANCESTORS_PKL = OUTPUT_DIR / "snomed_ancestors.pkl"
STATS_JSON   = OUTPUT_DIR / "snomed_hierarchy_stats.json"

# Inputs produced by earlier phases (consumed at build time, not imported).
_MESH_TO_SNOMED_CSV = OUTPUT_DIR / "mesh_to_snomed.csv"
_MRCM_JSON          = OUTPUT_DIR / "mrcm_constraints.json"


def _input_signature() -> tuple:
    """Tuple identifying the inputs the cache was built from. Cache invalidates if it changes."""
    parts: list = [CACHE_VERSION]
    for p in (
        SNOMED_FILES["concepts"],
        SNOMED_FILES["relationships"],
        _MESH_TO_SNOMED_CSV,
        _MRCM_JSON,
    ):
        if p.exists():
            stat = p.stat()
            parts.append((p.name, stat.st_size, int(stat.st_mtime)))
        else:
            parts.append((p.name, 0, 0))
    return tuple(parts)


# ---------------------------------------------------------------------------
# RF2 helpers (same pattern as mrcm.py)
# ---------------------------------------------------------------------------

def _read_rf2(path: Path) -> Iterator[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"RF2 file missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            yield row


def _active(row: Dict[str, str]) -> bool:
    return row.get("active") == "1"


# ---------------------------------------------------------------------------
# Phase-1 output readers
# ---------------------------------------------------------------------------

def _load_mapped_snomed_ids() -> Set[str]:
    """Return unique SNOMED concept IDs from mesh_to_snomed.csv."""
    if not _MESH_TO_SNOMED_CSV.exists():
        return set()
    ids: Set[str] = set()
    with _MESH_TO_SNOMED_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("snomed_id") or "").strip()
            if sid:
                ids.add(sid)
    return ids


def _load_mrcm_anchor_ids() -> Set[str]:
    """Return all domain/range root concept IDs referenced in mrcm_constraints.json."""
    if not _MRCM_JSON.exists():
        return set()
    data = json.loads(_MRCM_JSON.read_text(encoding="utf-8"))
    anchors: Set[str] = set()
    for blk in data.get("relation_constraints", {}).values():
        for d in blk.get("domains", []):
            anchors.update(d.get("domain_root_concept_ids", []))
        for r in blk.get("ranges", []):
            anchors.update(r.get("range_root_concept_ids", []))
    return anchors


# ---------------------------------------------------------------------------
# Step A: build graph
# ---------------------------------------------------------------------------

def build_graph(verbose: bool = True) -> nx.DiGraph:
    """Build DiGraph from active SNOMED is-a relationships (child -> parent edges)."""
    G: nx.DiGraph = nx.DiGraph()

    # Add all active concept nodes first so isolated concepts are present.
    if verbose:
        print("[1.6] loading active concepts ...", flush=True)
    concept_count = 0
    for row in _read_rf2(SNOMED_FILES["concepts"]):
        if _active(row):
            G.add_node(row["id"])
            concept_count += 1
    if verbose:
        print(f"[1.6] {concept_count:,} active concept nodes added")

    # Add is-a edges.
    if verbose:
        print("[1.6] loading is-a relationships ...", flush=True)
    edge_count = 0
    rel_path = SNOMED_FILES["relationships"]

    with tqdm(desc="  relationships", unit=" rows",
              disable=not verbose, mininterval=2.0) as bar:
        for row in _read_rf2(rel_path):
            bar.update(1)
            if not _active(row):
                continue
            if row.get("typeId") != IS_A_TYPE_ID:
                continue
            src = row["sourceId"]       # child
            dst = row["destinationId"]  # parent
            G.add_edge(src, dst)
            edge_count += 1

    if verbose:
        print(f"[1.6] {edge_count:,} active is-a edges added")
    return G


# ---------------------------------------------------------------------------
# Step B: compute depth + semantic type
# ---------------------------------------------------------------------------

def compute_metadata(G: nx.DiGraph, verbose: bool = True) -> None:
    """Attach `depth`, `semantic_type`, and `top_level_hierarchies` as node attributes.

    SNOMED is a DAG (every concept can have multiple is-a parents), so a
    concept may transitively inherit from more than one top-level hierarchy.
    We record both:

        depth                  -- shortest path length from SNOMED root.
        semantic_type          -- one top-level hierarchy SCTID (the
                                  shortest-path ancestor; deterministic via
                                  lexicographic tiebreak on equal depths).
        top_level_hierarchies  -- tuple of *all* top-level SCTIDs the concept
                                  ultimately inherits from. Multi-inheritance
                                  matters for the CSP solver: a concept that
                                  is both a clinical finding and a body
                                  structure satisfies attribute domain checks
                                  for either hierarchy.
    """
    R = G.reverse(copy=False)

    if verbose:
        print("[1.6] checking acyclicity ...", flush=True)
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))[:3]
        raise RuntimeError(
            f"SNOMED is-a graph is not a DAG; cycles detected (first 3): {cycles}"
        )

    if verbose:
        print("[1.6] computing depths via BFS from root ...", flush=True)
    depths: Dict[str, int] = {SNOMED_ROOT: 0}
    queue: deque = deque([SNOMED_ROOT])
    while queue:
        node = queue.popleft()
        for child in R.successors(node):
            if child in depths:
                continue
            depths[child] = depths[node] + 1
            queue.append(child)

    # Determine the set of top-level hierarchies (direct children of SNOMED root).
    top_level_ids: Set[str] = set(R.successors(SNOMED_ROOT))

    # Per-node: collect *all* top-level ancestors via downward BFS from each top-level.
    if verbose:
        print(f"[1.6] tagging top-level hierarchy membership ({len(top_level_ids)} hierarchies) ...", flush=True)
    membership: Dict[str, Set[str]] = {}
    for tl in top_level_ids:
        if tl not in G:
            continue
        for desc in _bfs_descendants(G, tl):
            membership.setdefault(desc, set()).add(tl)
        membership.setdefault(tl, set()).add(tl)
    membership_finalized: Dict[str, tuple] = {
        sid: tuple(sorted(tls)) for sid, tls in membership.items()
    }
    membership_finalized[SNOMED_ROOT] = ()

    # semantic_type = the shortest-path top-level ancestor; lex tiebreak on equal depth.
    sem_types: Dict[str, str] = {SNOMED_ROOT: SNOMED_ROOT}
    for sid, tls in membership_finalized.items():
        if not tls:
            continue
        sem_types[sid] = min(tls, key=lambda t: (depths.get(t, 1 << 30), t))

    nx.set_node_attributes(G, depths,                "depth")
    nx.set_node_attributes(G, sem_types,             "semantic_type")
    nx.set_node_attributes(G, membership_finalized,  "top_level_hierarchies")

    if verbose:
        reachable = len(depths)
        unreachable = G.number_of_nodes() - reachable
        multi = sum(1 for tls in membership_finalized.values() if len(tls) > 1)
        print(f"[1.6] depth assigned to {reachable:,} nodes "
              f"({unreachable:,} not reachable from root); "
              f"{multi:,} concepts under multiple top-level hierarchies")


# ---------------------------------------------------------------------------
# Step C: ancestor / descendant sets
# ---------------------------------------------------------------------------

def _bfs_ancestors(G: nx.DiGraph, source: str) -> FrozenSet[str]:
    """All nodes reachable from `source` following child->parent edges (= ancestors)."""
    visited: Set[str] = set()
    queue: deque = deque([source])
    while queue:
        node = queue.popleft()
        for parent in G.successors(node):
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)
    return frozenset(visited)


def _bfs_descendants(G: nx.DiGraph, source: str) -> FrozenSet[str]:
    """All nodes reachable from `source` in the *reversed* graph (= descendants)."""
    R = G.reverse(copy=False)
    visited: Set[str] = set()
    queue: deque = deque([source])
    while queue:
        node = queue.popleft()
        for child in R.successors(node):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return frozenset(visited)


def compute_ancestor_sets(
    G: nx.DiGraph,
    mapped_ids: Set[str],
    mrcm_ids: Set[str],
    verbose: bool = True,
) -> Dict[str, FrozenSet[str]]:
    """Return a unified lookup dict with two key namespaces:

        "<snomed_id>"            -> frozenset of ancestor concept IDs
        "descendant:<anchor_id>" -> frozenset of descendant concept IDs
    """
    result: Dict[str, FrozenSet[str]] = {}

    # Ancestor sets for mapped concepts (used in Phase 2 soft mapping + Phase 4).
    in_graph = [sid for sid in mapped_ids if sid in G]
    if verbose:
        print(f"[1.6] computing ancestor sets for {len(in_graph):,} mapped concepts ...",
              flush=True)
    for sid in tqdm(in_graph, desc="  ancestors", disable=not verbose, mininterval=2.0):
        result[sid] = _bfs_ancestors(G, sid)

    # Descendant sets for MRCM anchors (O(1) CSP membership checks in Phase 3.5).
    if verbose:
        print(f"[1.6] computing descendant sets for {len(mrcm_ids)} MRCM anchors ...",
              flush=True)
    for anchor in sorted(mrcm_ids):
        if anchor not in G:
            if verbose:
                print(f"[1.6]   WARNING: MRCM anchor {anchor} not in graph — skipping")
            continue
        result[f"descendant:{anchor}"] = _bfs_descendants(G, anchor)
        if verbose:
            print(f"[1.6]   descendant:{anchor} -> {len(result[f'descendant:{anchor}']):,} concepts")

    return result


# ---------------------------------------------------------------------------
# Step D: pickle cache
# ---------------------------------------------------------------------------

def _try_load_cache(verbose: bool) -> Optional[tuple[nx.DiGraph, Dict[str, FrozenSet[str]]]]:
    if not (GRAPH_PKL.exists() and ANCESTORS_PKL.exists()):
        return None
    sig = _input_signature()
    try:
        with GRAPH_PKL.open("rb") as fh:
            graph_payload = pickle.load(fh)
        with ANCESTORS_PKL.open("rb") as fh:
            anc_payload = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        if verbose:
            print(f"[1.6] cache unreadable ({exc}); rebuilding")
        return None
    if (
        not isinstance(graph_payload, dict)
        or not isinstance(anc_payload, dict)
        or graph_payload.get("signature") != sig
        or anc_payload.get("signature") != sig
    ):
        if verbose:
            print("[1.6] cache stale (input files changed); rebuilding")
        return None
    if verbose:
        print("[1.6] loading from cache ...", flush=True)
    return graph_payload["graph"], anc_payload["ancestors"]


def load_or_build(
    force: bool = False,
    verbose: bool = True,
) -> tuple[nx.DiGraph, Dict[str, FrozenSet[str]]]:
    """Return (G, ancestors).  Builds from scratch if cache is absent, stale, or force=True."""
    if not force:
        cached = _try_load_cache(verbose=verbose)
        if cached is not None:
            return cached

    mapped_ids = _load_mapped_snomed_ids()
    mrcm_ids   = _load_mrcm_anchor_ids()

    G = build_graph(verbose=verbose)
    compute_metadata(G, verbose=verbose)

    ancestors = compute_ancestor_sets(G, mapped_ids, mrcm_ids, verbose=verbose)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = _input_signature()
    with GRAPH_PKL.open("wb") as fh:
        pickle.dump({"signature": sig, "graph": G}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with ANCESTORS_PKL.open("wb") as fh:
        pickle.dump({"signature": sig, "ancestors": ancestors}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"[1.6] graph     -> {GRAPH_PKL}")
        print(f"[1.6] ancestors -> {ANCESTORS_PKL}")

    return G, ancestors


# ---------------------------------------------------------------------------
# Step E: stats JSON
# ---------------------------------------------------------------------------

def dump_stats(
    G: nx.DiGraph,
    ancestors: Dict[str, FrozenSet[str]],
    mapped_ids: Set[str],
    mrcm_ids: Set[str],
    verbose: bool = True,
) -> Path:
    depths    = nx.get_node_attributes(G, "depth")
    sem_types = nx.get_node_attributes(G, "semantic_type")
    tl_membership = nx.get_node_attributes(G, "top_level_hierarchies")

    # Top-level hierarchies = direct children of root in the reversed graph.
    R = G.reverse(copy=False)
    top_level_ids = sorted(R.successors(SNOMED_ROOT))

    # Resolve human-readable FSNs for top-level hierarchies + MRCM anchors.
    needed_names = set(top_level_ids) | set(mrcm_ids)
    name_index = _mrcm_mod._load_descriptions(needed_names) if needed_names else {}

    # Semantic-type distribution (primary).
    sem_dist: Dict[str, int] = {}
    for sem in sem_types.values():
        sem_dist[sem] = sem_dist.get(sem, 0) + 1

    multi_inherit_count = sum(1 for tls in tl_membership.values() if len(tls) > 1)

    # Ancestor set size stats (for mapped concepts only).
    anc_sizes = [len(v) for k, v in ancestors.items() if not k.startswith("descendant:")]

    # Per-anchor descendant counts.
    desc_sizes = {
        k[len("descendant:"):]: len(v)
        for k, v in ancestors.items()
        if k.startswith("descendant:")
    }

    missing_mapped = sorted(sid for sid in mapped_ids if sid not in G)
    anchor_present = {aid: (aid in G) for aid in sorted(mrcm_ids)}

    stats = {
        "snomed_root":              SNOMED_ROOT,
        "active_concepts":          G.number_of_nodes(),
        "is_a_edges":               G.number_of_edges(),
        "max_depth":                max(depths.values(), default=0),
        "top_level_hierarchy_ids":  top_level_ids,
        "top_level_hierarchies": [
            {"concept_id": tid, "name": name_index.get(tid, ""),
             "primary_count": sem_dist.get(tid, 0)}
            for tid in top_level_ids
        ],
        "top_level_hierarchy_count": len(top_level_ids),
        "multi_inheritance_concepts": multi_inherit_count,
        "semantic_type_distribution": {k: v for k, v in
                                        sorted(sem_dist.items(), key=lambda x: -x[1])},
        "mrcm_anchors_in_graph": [
            {"concept_id": aid, "name": name_index.get(aid, ""), "in_graph": present}
            for aid, present in sorted(anchor_present.items())
        ],
        "mapped_concepts_total":    len(mapped_ids),
        "mapped_concepts_in_graph": len(mapped_ids) - len(missing_mapped),
        "mapped_concepts_missing":  missing_mapped,
        "ancestor_set_sizes": {
            "count": len(anc_sizes),
            "min":   min(anc_sizes, default=0),
            "max":   max(anc_sizes, default=0),
            "mean":  round(mean(anc_sizes), 1) if anc_sizes else 0,
        },
        "descendant_set_sizes": desc_sizes,
        "outputs": {
            "graph_pkl":    str(GRAPH_PKL),
            "ancestors_pkl": str(ANCESTORS_PKL),
            "stats_json":   str(STATS_JSON),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with STATS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[1.6] active_concepts={stats['active_concepts']:,}  "
              f"is_a_edges={stats['is_a_edges']:,}  "
              f"max_depth={stats['max_depth']}  "
              f"top_level={stats['top_level_hierarchy_count']}  "
              f"multi_inherit={multi_inherit_count:,}")
        if missing_mapped:
            print(f"[1.6] WARNING: {len(missing_mapped)} mapped concepts not in graph")
        missing_anchors = [k for k, v in anchor_present.items() if not v]
        if missing_anchors:
            print(f"[1.6] WARNING: MRCM anchors missing from graph: {missing_anchors}")

    return STATS_JSON


# ---------------------------------------------------------------------------
# Public query helpers (for Phase 3.5 and Phase 4)
# ---------------------------------------------------------------------------

def get_ancestors(
    snomed_id: str,
    ancestors: Dict[str, FrozenSet[str]],
    G: Optional[nx.DiGraph] = None,
) -> FrozenSet[str]:
    """Return ancestor set for a SNOMED concept.

    Precomputed for concepts in the Phase 1.2 mapping table. For arbitrary
    concepts (Phase 4 evaluation, CSP candidates not in the mapping table)
    fall back to runtime BFS on G when provided. Returns empty frozenset if
    the concept is unknown to both.
    """
    cached = ancestors.get(snomed_id)
    if cached is not None:
        return cached
    if G is not None and snomed_id in G:
        return _bfs_ancestors(G, snomed_id)
    return frozenset()


def is_descendant_of(
    concept_id: str,
    anchor_id: str,
    ancestors: Dict[str, FrozenSet[str]],
) -> bool:
    """True iff concept_id is in the descendant closure of anchor_id.

    Uses the precomputed `descendant:<anchor_id>` set -- O(1) lookup.
    """
    desc_key = f"descendant:{anchor_id}"
    desc_set = ancestors.get(desc_key)
    if desc_set is None:
        return False
    return concept_id in desc_set


def get_top_level_hierarchies(G: nx.DiGraph, snomed_id: str) -> tuple:
    """Return the tuple of top-level SNOMED hierarchies that subsume `snomed_id`.

    A concept under multiple is-a paths may legitimately belong to several
    hierarchies (e.g. a finding that is also a body structure). The CSP
    solver should treat the concept as satisfying any of them.
    """
    return G.nodes.get(snomed_id, {}).get("top_level_hierarchies", ())


def get_depth(G: nx.DiGraph, snomed_id: str) -> Optional[int]:
    return G.nodes.get(snomed_id, {}).get("depth")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def main(force: bool = False, verbose: bool = True) -> Path:
    mapped_ids = _load_mapped_snomed_ids()
    mrcm_ids   = _load_mrcm_anchor_ids()
    if verbose:
        print(f"[1.6] mapped SNOMED IDs={len(mapped_ids):,}  "
              f"MRCM anchors={len(mrcm_ids)}")

    G, ancestors = load_or_build(force=force, verbose=verbose)
    return dump_stats(G, ancestors, mapped_ids, mrcm_ids, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SNOMED hierarchy graph (Phase 1.6).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore pickled cache and rebuild from RF2 files.")
    args = parser.parse_args()
    main(force=args.force, verbose=True)
