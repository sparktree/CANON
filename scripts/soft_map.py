"""Soft mapping preprocessing for CANON Phase 2.4.

For every MeSH->SNOMED mapping in mesh_to_snomed_verified.csv, produces a
probability distribution over nearby SNOMED candidates using the three
similarity signals the plan specifies:

    sim_string       -- SapBERT cosine similarity (cambridgeltl/SapBERT-from-
                        PubMedBERT-fulltext) between the MeSH preferred term
                        and the candidate SNOMED preferred term. Captures
                        surface-form / lexical similarity in a learned space.
    sim_ontological  -- 1 / (1 + hop_dist) where hop_dist is the is-a graph
                        distance (0 for primary, 1 for parent/child, 2 for
                        grandparent/grandchild/sibling). Captures structural
                        proximity in the SNOMED hierarchy.
    sim_ic           -- Lin similarity using Resnik information content over
                        the SNOMED is-a graph: 2 * IC(LCA) / (IC(c1) + IC(c2))
                        with IC(c) = -log2(|descendants(c)+1| / |G|). Captures
                        taxonomic similarity weighted by concept specificity:
                        an LCA deep in the hierarchy (high IC) means the two
                        concepts share a specific common reading; an LCA near
                        the root (low IC) means they share only a generic
                        ancestor.

The three signals are summed with equal weights (W_STRING=W_ONTO=W_IC=1/3),
then softmax(/, TEMPERATURE) produces the soft distribution. Each signal is
also persisted alongside the candidate so downstream code (Phase 3 CN head,
Phase 4.6 attribution) can interrogate which signal contributed.

Neighbourhood: for each primary SNOMED concept, collects parents, grandparents,
children, grandchildren, and siblings up to NEIGHBOURHOOD_CAP candidates.

Outputs:
    outputs/phase2/soft_mapping_lookup.json   -- {mesh_id: [{snomed_id, term,
                                                  prob, hop_dist, sim_string,
                                                  sim_ontological, sim_ic}, ...]}
    outputs/phase2/soft_mapping_summary.json  -- aggregate statistics + weights
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "soft_map.py requires sentence-transformers. "
        "Install with: pip install sentence-transformers"
    ) from exc

try:
    from config import REPO_ROOT, relative_to_repo
    import mrcm
    import snomed_hierarchy
    import umls_query
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, relative_to_repo
    import mrcm
    import snomed_hierarchy
    import umls_query


VERIFIED_CSV = REPO_ROOT / "outputs" / "phase1" / "mesh_to_snomed_verified.csv"
OUTPUT_DIR   = REPO_ROOT / "outputs" / "phase2"
LOOKUP_JSON  = OUTPUT_DIR / "soft_mapping_lookup.json"
SUMMARY_JSON = OUTPUT_DIR / "soft_mapping_summary.json"

SAPBERT_MODEL      = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
NEIGHBOURHOOD_CAP  = 30
TEMPERATURE        = 0.05

# Three-signal weights. Equal default; tunable.
W_STRING           = 1.0 / 3
W_ONTOLOGICAL      = 1.0 / 3
W_IC               = 1.0 / 3

_FSN_SUFFIX = re.compile(r"\s*\([^)]+\)\s*$")


def _strip_fsn(term: str) -> str:
    return _FSN_SUFFIX.sub("", term).strip()


def _load_verified_table() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with VERIFIED_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mesh_id   = (row.get("mesh_id")    or "").strip()
            snomed_id = (row.get("snomed_id")   or "").strip()
            snomed_term = (row.get("snomed_term") or "").strip()
            if mesh_id and snomed_id:
                rows.append({"mesh_id": mesh_id, "snomed_id": snomed_id, "snomed_term": snomed_term})
    return rows


def _get_mesh_term(mesh_id: str, fallback: str) -> str:
    """Look up the MeSH preferred name from the UMLS in-memory cache."""
    for cui in umls_query.code_to_cuis.get(("MSH", mesh_id), []):
        for atom in umls_query.cui_to_atoms.get(cui, []):
            if atom.get("sab") == "MSH" and atom.get("tty") in {"MH", "PEP"}:
                name = atom.get("str", "")
                if name:
                    return name
    return fallback


def _build_neighbourhood(G, snomed_id: str) -> List[Tuple[str, int]]:
    """Return (concept_id, hop_dist) pairs, primary concept first, capped at NEIGHBOURHOOD_CAP.

    In G edges are child -> parent, so G.successors(x) = parents of x
    and G.predecessors(x) = children of x.
    """
    if snomed_id not in G:
        return [(snomed_id, 0)]

    result: List[Tuple[str, int]] = [(snomed_id, 0)]
    seen: Set[str] = {snomed_id}
    cap = NEIGHBOURHOOD_CAP

    parents = list(G.successors(snomed_id))

    for p in parents:
        if p not in seen:
            result.append((p, 1))
            seen.add(p)
        if len(result) >= cap:
            return result

    for p in parents:
        for gp in G.successors(p):
            if gp not in seen:
                result.append((gp, 2))
                seen.add(gp)
            if len(result) >= cap:
                return result

    for c in G.predecessors(snomed_id):
        if c not in seen:
            result.append((c, 1))
            seen.add(c)
        if len(result) >= cap:
            return result

    for c in list(G.predecessors(snomed_id)):
        for gc in G.predecessors(c):
            if gc not in seen:
                result.append((gc, 2))
                seen.add(gc)
            if len(result) >= cap:
                return result

    for p in parents:
        for sib in G.predecessors(p):
            if sib not in seen:
                result.append((sib, 2))
                seen.add(sib)
            if len(result) >= cap:
                return result

    return result


def _softmax(sims: np.ndarray, temperature: float) -> np.ndarray:
    logits = sims / temperature
    logits = logits - logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def _compute_ic_and_ancestors(
    G,
    concepts: Set[str],
    verbose: bool = False,
) -> Tuple[Dict[str, FrozenSet[str]], Dict[str, float]]:
    """Compute ancestor-with-self closures and Resnik IC for each concept.

    IC(c) = -log2((|descendants(c)| + 1) / |G_active|), so leaves have the
    maximum IC and the root has the minimum (0). Both BFS calls reuse the
    Phase 1.6 graph helpers and run in the order of seconds for ~60K concepts.
    """
    from snomed_hierarchy import _bfs_ancestors, _bfs_descendants

    total_active = G.number_of_nodes()
    if verbose:
        print(f"[2.4]   computing ancestors + IC for {len(concepts):,} concepts ...", flush=True)

    ancestors: Dict[str, FrozenSet[str]] = {}
    ic: Dict[str, float] = {}
    for c in concepts:
        if c not in G:
            ancestors[c] = frozenset({c})
            ic[c] = 0.0
            continue
        ancestors[c] = _bfs_ancestors(G, c) | {c}
        desc_count = len(_bfs_descendants(G, c)) + 1  # include self
        ic[c] = -math.log2(desc_count / total_active) if desc_count > 0 else 0.0
    return ancestors, ic


def _lin_similarity(
    c1: str,
    c2: str,
    ancestors: Dict[str, FrozenSet[str]],
    ic: Dict[str, float],
) -> float:
    """Lin similarity = 2 * IC(LCA(c1,c2)) / (IC(c1) + IC(c2)).

    LCA in a DAG is the common ancestor with the highest IC (most specific
    shared ancestor). Returns 1.0 when c1 == c2, 0.0 when no common ancestor
    exists or both concepts are root-level (zero IC).
    """
    if c1 == c2:
        return 1.0
    a1 = ancestors.get(c1, frozenset())
    a2 = ancestors.get(c2, frozenset())
    common = a1 & a2
    if not common:
        return 0.0
    lca_ic = max((ic.get(a, 0.0) for a in common), default=0.0)
    denom = ic.get(c1, 0.0) + ic.get(c2, 0.0)
    if denom <= 0.0:
        return 0.0
    return 2.0 * lca_ic / denom


def apply_all(verbose: bool = True) -> Path:
    if not VERIFIED_CSV.exists():
        raise FileNotFoundError(f"{VERIFIED_CSV} not found; run Phase 1.7 first.")

    if verbose:
        print("[2.4] loading UMLS (uses cached pickle) ...", flush=True)
    umls_query.preload(force=False)

    if verbose:
        print("[2.4] loading SNOMED hierarchy ...", flush=True)
    G, _ = snomed_hierarchy.load_or_build(force=False, verbose=False)

    if verbose:
        print(f"[2.4] loading SapBERT model: {SAPBERT_MODEL} ...", flush=True)
    model = SentenceTransformer(SAPBERT_MODEL)

    rows = _load_verified_table()
    if verbose:
        print(f"[2.4] {len(rows):,} verified mappings to process")

    # Build neighbourhoods and collect all unique SNOMED concept IDs
    mesh_data: List[Dict] = []
    all_snomed_ids: Set[str] = set()

    for row in rows:
        neighbourhood = _build_neighbourhood(G, row["snomed_id"])
        all_snomed_ids.update(cid for cid, _ in neighbourhood)
        mesh_data.append({
            "mesh_id":          row["mesh_id"],
            "mesh_term":        _get_mesh_term(row["mesh_id"], fallback=row["snomed_term"]),
            "primary_snomed_id": row["snomed_id"],
            "neighbourhood":    neighbourhood,
        })

    # Load SNOMED descriptions for all neighbourhood concepts in one pass
    if verbose:
        print(f"[2.4] loading descriptions for {len(all_snomed_ids):,} unique SNOMED concepts ...", flush=True)
    raw_descs = mrcm.get_descriptions(all_snomed_ids)
    descs: Dict[str, str] = {cid: _strip_fsn(t) for cid, t in raw_descs.items()}

    # Encode all SNOMED terms + all MeSH terms in a single batched model.encode pass
    unique_snomed_ids  = sorted(all_snomed_ids)
    snomed_terms_list  = [descs.get(cid, cid) for cid in unique_snomed_ids]
    snomed_id_to_idx   = {cid: i for i, cid in enumerate(unique_snomed_ids)}
    mesh_terms_list    = [d["mesh_term"] for d in mesh_data]

    if verbose:
        print(
            f"[2.4] encoding {len(snomed_terms_list):,} SNOMED + "
            f"{len(mesh_terms_list):,} MeSH terms with SapBERT ...",
            flush=True,
        )

    all_embeddings = model.encode(
        snomed_terms_list + mesh_terms_list,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=verbose,
    )
    snomed_embs = all_embeddings[: len(snomed_terms_list)]  # (N_snomed, 768)
    mesh_embs   = all_embeddings[len(snomed_terms_list):]   # (N_mesh,   768)

    # Precompute ancestor closures + Resnik IC for the union of every primary
    # SNOMED ID and every neighbourhood candidate. Used by the Lin-similarity
    # signal in the loop below.
    concepts_for_ic: Set[str] = set(all_snomed_ids)
    concepts_for_ic.update(d["primary_snomed_id"] for d in mesh_data)
    ancestors_cache, ic_cache = _compute_ic_and_ancestors(
        G, concepts_for_ic, verbose=verbose
    )

    # Compute soft distributions over the three signals.
    lookup: Dict[str, List[Dict]] = {}
    sim_string_means: List[float] = []
    sim_onto_means: List[float] = []
    sim_ic_means: List[float] = []
    for i, d in enumerate(mesh_data):
        neighbourhood = d["neighbourhood"]
        primary_id    = d["primary_snomed_id"]
        mesh_emb      = mesh_embs[i]

        cand_ids  = [cid  for cid, _   in neighbourhood]
        cand_hops = [hop  for _,   hop in neighbourhood]
        cand_embs = snomed_embs[[snomed_id_to_idx[cid] for cid in cand_ids]]

        # Signal 1: SapBERT cosine in [-1, 1] -- string/lexical similarity.
        sim_string = cand_embs @ mesh_emb

        # Signal 2: ontological proximity 1/(1+hop) in (0, 1].
        sim_onto = np.array([1.0 / (1.0 + h) for h in cand_hops], dtype=np.float32)

        # Signal 3: Lin similarity (taxonomic via IC(LCA)) in [0, 1].
        sim_ic = np.array(
            [_lin_similarity(primary_id, cid, ancestors_cache, ic_cache)
             for cid in cand_ids],
            dtype=np.float32,
        )

        # Combined logit (equal weights by default).
        combined = (
            W_STRING       * sim_string +
            W_ONTOLOGICAL  * sim_onto +
            W_IC           * sim_ic
        )
        probs = _softmax(combined, TEMPERATURE)

        sim_string_means.append(float(sim_string.mean()))
        sim_onto_means.append(float(sim_onto.mean()))
        sim_ic_means.append(float(sim_ic.mean()))

        candidates = [
            {"snomed_id": cid,
             "term": descs.get(cid, cid),
             "prob": round(float(p), 6),
             "hop_dist": hop,
             "sim_string":      round(float(sim_string[k]), 4),
             "sim_ontological": round(float(sim_onto[k]),   4),
             "sim_ic":          round(float(sim_ic[k]),     4)}
            for k, (cid, hop, p) in enumerate(zip(cand_ids, cand_hops, probs.tolist()))
        ]
        candidates.sort(key=lambda x: -x["prob"])
        lookup[d["mesh_id"]] = candidates

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOOKUP_JSON.open("w", encoding="utf-8") as fh:
        json.dump(lookup, fh, indent=2, ensure_ascii=False)

    # Summary
    cand_counts = [len(v) for v in lookup.values()]
    primary_probs: List[float] = []
    for d, cands in zip(mesh_data, lookup.values()):
        for c in cands:
            if c["snomed_id"] == d["primary_snomed_id"]:
                primary_probs.append(c["prob"])
                break

    summary = {
        "model": SAPBERT_MODEL,
        "temperature": TEMPERATURE,
        "neighbourhood_cap": NEIGHBOURHOOD_CAP,
        "weights": {
            "string":       W_STRING,
            "ontological":  W_ONTOLOGICAL,
            "ic":           W_IC,
        },
        "total_mesh_ids": len(lookup),
        "candidates_per_mesh": {
            "min":  min(cand_counts),
            "max":  max(cand_counts),
            "mean": round(sum(cand_counts) / len(cand_counts), 2),
        },
        "primary_concept_probability": {
            "min":  round(min(primary_probs),  4) if primary_probs else 0.0,
            "max":  round(max(primary_probs),  4) if primary_probs else 0.0,
            "mean": round(sum(primary_probs) / len(primary_probs), 4) if primary_probs else 0.0,
        },
        "signal_means_over_candidates": {
            "sim_string":      round(sum(sim_string_means) / len(sim_string_means), 4) if sim_string_means else 0.0,
            "sim_ontological": round(sum(sim_onto_means)   / len(sim_onto_means),   4) if sim_onto_means   else 0.0,
            "sim_ic":          round(sum(sim_ic_means)     / len(sim_ic_means),     4) if sim_ic_means     else 0.0,
        },
        "outputs": {"lookup": relative_to_repo(LOOKUP_JSON)},
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[2.4] lookup  -> {LOOKUP_JSON}")
        print(f"[2.4] summary -> {SUMMARY_JSON}")
        c = summary["candidates_per_mesh"]
        p = summary["primary_concept_probability"]
        print(f"[2.4] candidates per mesh_id: min={c['min']} max={c['max']} mean={c['mean']}")
        print(f"[2.4] primary-concept prob:   min={p['min']} max={p['max']} mean={p['mean']}")

    return LOOKUP_JSON


if __name__ == "__main__":
    apply_all(verbose=True)
