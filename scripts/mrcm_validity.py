"""Shared MRCM + Semantic Network validity logic for the CSP solver and evaluator.

Single source of truth for the three constraint surfaces that the Phase 3.5 CSP
solver ENFORCES and the Phase 4.3 coherence evaluator MEASURES:

    entity-level         : concept_under_type -- the normalized concept lies
                           under the NER type's SNOMED anchor(s).
    relation-level Tier-1 : relation_pair_valid -- a Tier-1 relation's subject/
                           object semantic types satisfy the SNOMED MRCM domain/
                           range. Orientation-robust: SNOMED attributes are
                           directed (causative-agent reads finding -> substance)
                           but the corpora annotate the pair in a non-canonical
                           order, so a pair is valid if EITHER argument order
                           satisfies domain/range; relation_pair_orientation
                           reports which order was valid so a canonical triple
                           can be emitted / evaluated.
    relation-level Tier-2 : relation_pair_valid_sn -- a Tier-2 relation's
                           subject/object UMLS semantic types (STYs, from MRSTY)
                           form an allowed edge in the UMLS Semantic Network
                           (Phase 1.5). Same orientation-robust treatment via
                           relation_pair_orientation_sn. This is the surface that
                           covers the majority Tier-2 relation population; MRCM
                           alone leaves it near-vacuous on BioRED + BC5CDR.

Both MRCM and the SN give NECESSARY type conditions, not sufficiency -- they
reject type-impossible outputs, they do not confirm a specific asserted relation.

Both the solver and the evaluator import THIS module, so the guarantee the solver
provides and the coherence the evaluator reports cannot drift apart. The module
is intentionally free of torch / dataset / umls_query imports -- it is cheap to
import from either side; the relation-tier sets, semantic-class list, and (for
Tier-2) each concept's STYs are passed in by the caller rather than imported here.
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple


# Type-to-anchor mapping (per plan: both anchors valid for disease + chemical).
TYPE_ANCHORS: Dict[str, List[str]] = {
    "disease":  ["404684003", "64572001"],
    "chemical": ["105590001", "373873005"],
}

# Tier-1 SNOMED attribute SCTIDs (match mrcm_constraints.json keys).
TIER1_ATTRIBUTE_IDS: Dict[str, str] = {
    "causative-agent":       "246075003",
    "finding-site":          "363698007",
    "associated-morphology": "116676008",
    "due-to":                "42752001",
    "after":                 "255234002",
}

# CANON Tier-2 relation -> UMLS Semantic Network predicate name(s). The SN
# supplies the Tier-2 type-level domain/range constraint (Phase 1.5): a Tier-2
# relation is coherent iff (subject semantic type, sn_predicate, object semantic
# type) is an allowed edge in the SN. Relations mapped to an empty list have no
# SN predicate and are left unconstrained by construction (documented in the
# plan): converts-to and compared-with.
TIER2_TO_SN_PREDICATES: Dict[str, List[str]] = {
    "treats":          ["treats"],
    "causes":          ["causes"],
    "associated-with": ["associated_with"],
    "interacts-with":  ["interacts_with"],
    "co-treats":       ["treats"],
    "converts-to":     [],
    "compared-with":   [],
}

# Orientation tags returned by relation_pair_orientation.
ORIENT_FORWARD = "forward"    # subject satisfies domain, object satisfies range
ORIENT_REVERSED = "reversed"  # object satisfies domain, subject satisfies range
ORIENT_NONE = None            # no argument order satisfies domain/range


@dataclass
class MRCMTables:
    """Precomputed MRCM + Semantic Network tables shared by solver and evaluator."""
    type_to_anchors: Dict[str, List[str]] = field(default_factory=dict)
    descendants: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    # rel -> (domain_root_ids, range_root_ids)
    domain_range: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = field(default_factory=dict)
    # (relation_label, type_a, type_b) -> bool. Tier-2 always True.
    valid_pair_for_relation: Dict[Tuple[str, str, str], bool] = field(default_factory=dict)
    # UMLS Semantic Network allowed edges: {(subject_STY, sn_predicate, object_STY)}.
    sn_edges: FrozenSet[Tuple[str, str, str]] = field(default_factory=frozenset)
    # CANON Tier-2 relation -> SN predicate names (copy of TIER2_TO_SN_PREDICATES).
    tier2_to_sn: Dict[str, List[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_descendants(ancestors_path: Path) -> Dict[str, FrozenSet[str]]:
    with Path(ancestors_path).open("rb") as fh:
        blob = pickle.load(fh)
    anc_dict = blob.get("ancestors", {}) if isinstance(blob, dict) else blob
    out: Dict[str, FrozenSet[str]] = {}
    for k, v in anc_dict.items():
        key = str(k)
        if key.startswith("descendant:"):
            out[key.split(":", 1)[1]] = frozenset(str(x) for x in v)
    return out


def _parse_domain_range(
    relation_constraints: Dict[str, dict],
    tier1_relations: Iterable[str],
) -> Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]:
    """rel -> (domain_root_ids, range_root_ids) for Tier-1 attributes.

    mrcm_constraints.json stores each attribute as a dict with "domains" and
    "ranges" lists; each entry carries *_root_concept_ids.
    """
    tier1 = set(tier1_relations)
    out: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {}
    for rel, entry in relation_constraints.items():
        if rel not in tier1:
            continue
        domain_set = frozenset(
            str(cid)
            for dm in entry.get("domains", [])
            for cid in dm.get("domain_root_concept_ids", [])
        )
        range_set = frozenset(
            str(cid)
            for rg in entry.get("ranges", [])
            for cid in rg.get("range_root_concept_ids", [])
        )
        out[rel] = (domain_set, range_set)
    return out


def _anchor_in(anchors: Iterable[str], root_set: FrozenSet[str]) -> bool:
    return any(a in root_set for a in anchors)


def _load_semantic_network(srstre2_path: Path) -> FrozenSet[Tuple[str, str, str]]:
    """Parse SRSTRE2 (name-keyed, fully-inherited SN edges) into an edge set.

    Each line is ``subject_STY|relation|object_STY|``. Only edges whose relation
    is a predicate CANON Tier-2 relations map to are retained, so the table stays
    small and focused on the relations we can constrain.
    """
    wanted = {p for preds in TIER2_TO_SN_PREDICATES.values() for p in preds}
    edges: set = set()
    with Path(srstre2_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 3:
                continue
            subj, rel, obj = parts[0], parts[1], parts[2]
            if rel in wanted:
                edges.add((subj, rel, obj))
    return frozenset(edges)


def load_tables(
    mrcm_path: Path,
    ancestors_path: Path,
    *,
    tier1_relations: Iterable[str],
    tier2_relations: Iterable[str],
    semantic_classes: Iterable[str],
    semantic_network_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> MRCMTables:
    """Build the shared MRCM + Semantic Network tables once per session.

    *semantic_network_path* points at SRSTRE2 (UMLS NET/). When given, the Tier-2
    SN edge set is loaded and relation_pair_valid_sn becomes a real constraint;
    when omitted, Tier-2 relations remain unconstrained (sn_edges empty).
    """
    logger = logger or logging.getLogger("mrcm_validity")
    tables = MRCMTables(type_to_anchors=copy.deepcopy(TYPE_ANCHORS))
    tables.descendants = _load_descendants(ancestors_path)
    tables.tier2_to_sn = {k: list(v) for k, v in TIER2_TO_SN_PREDICATES.items()}

    with Path(mrcm_path).open("r", encoding="utf-8") as fh:
        mrcm = json.load(fh)
    relation_constraints = mrcm.get("relation_constraints", {})
    tables.domain_range = _parse_domain_range(relation_constraints, tier1_relations)

    # Tier-1 pair compatibility, orientation-robust (see module docstring).
    for rel, (domain_set, range_set) in tables.domain_range.items():
        for type_a, anchors_a in TYPE_ANCHORS.items():
            for type_b, anchors_b in TYPE_ANCHORS.items():
                fwd = _anchor_in(anchors_a, domain_set) and _anchor_in(anchors_b, range_set)
                rev = _anchor_in(anchors_b, domain_set) and _anchor_in(anchors_a, range_set)
                tables.valid_pair_for_relation[(rel, type_a, type_b)] = bool(fwd or rev)

    # Tier-2 relations are unconstrained at the coarse (semantic-class) level;
    # their real constraint is the Semantic Network check at UMLS-STY granularity
    # (relation_pair_valid_sn), which the solver/evaluator apply per predicted
    # relation using the concepts' MRSTY semantic types.
    for rel in tier2_relations:
        for ta in list(semantic_classes) + ["none"]:
            for tb in list(semantic_classes) + ["none"]:
                tables.valid_pair_for_relation[(rel, ta, tb)] = True

    if semantic_network_path is not None:
        tables.sn_edges = _load_semantic_network(semantic_network_path)

    logger.info(
        f"loaded MRCM+SN tables: {len(tables.descendants)} descendant sets; "
        f"{len(tables.domain_range)} Tier-1 domain/range; "
        f"{len(tables.valid_pair_for_relation)} (relation, ta, tb) pairs; "
        f"{len(tables.sn_edges)} SN Tier-2 edges"
    )
    return tables


# ---------------------------------------------------------------------------
# Predicates (imported by BOTH solver and evaluator)
# ---------------------------------------------------------------------------
def concept_under_type(concept_id: str, sem_class: str, tables: MRCMTables) -> bool:
    """Entity-level surface: concept lies under the NER type's SNOMED anchor(s)."""
    anchors = tables.type_to_anchors.get(sem_class)
    if not anchors:
        # NER-only types (gene, variant, species, cell_line) have no constraint.
        return True
    for anc in anchors:
        ds = tables.descendants.get(anc)
        if ds is not None and (concept_id == anc or concept_id in ds):
            return True
    return False


def relation_pair_valid(rel: str, type_a: str, type_b: str, tables: MRCMTables) -> bool:
    """Relation-level surface: the Tier-1 pair satisfies MRCM in some orientation.

    Tier-2 relations are always valid. Unknown (rel, ta, tb) defaults invalid,
    matching the solver's behaviour of forcing no-relation for such pairs.
    """
    return tables.valid_pair_for_relation.get((rel, type_a, type_b), False)


def relation_pair_orientation(
    rel: str, type_a: str, type_b: str, tables: MRCMTables
) -> Optional[str]:
    """Which argument order satisfies the directed MRCM domain/range.

    Returns ORIENT_FORWARD if the annotated (subject=a, object=b) order already
    matches domain/range, ORIENT_REVERSED if only the swapped order does, or
    ORIENT_NONE if neither. For Tier-2 (unconstrained) or symmetric-type Tier-1
    pairs the forward order is reported so the caller keeps annotation order.
    Lets the solver/evaluator emit a canonical (finding, attribute, substance)
    triple rather than the corpus's non-canonical order.
    """
    dr = tables.domain_range.get(rel)
    if dr is None:
        # Not a Tier-1 domain/range-constrained relation: keep annotation order.
        return ORIENT_FORWARD
    domain_set, range_set = dr
    anchors_a = TYPE_ANCHORS.get(type_a, [])
    anchors_b = TYPE_ANCHORS.get(type_b, [])
    fwd = _anchor_in(anchors_a, domain_set) and _anchor_in(anchors_b, range_set)
    if fwd:
        return ORIENT_FORWARD
    rev = _anchor_in(anchors_b, domain_set) and _anchor_in(anchors_a, range_set)
    if rev:
        return ORIENT_REVERSED
    return ORIENT_NONE


# ---------------------------------------------------------------------------
# Tier-2 Semantic Network predicates (relation-level surface iii)
# ---------------------------------------------------------------------------
# These operate on UMLS semantic-type NAMES (STYs), resolved from each mention's
# normalized concept via MRSTY by the caller (solver / evaluator) -- e.g. via
# umls_query.cui_to_stys. Kept concept-agnostic here so the module stays pure.

def _sn_edge_exists(
    sn_predicates: Iterable[str],
    subj_stys: Iterable[str],
    obj_stys: Iterable[str],
    tables: MRCMTables,
) -> bool:
    subj = list(subj_stys)
    obj = list(obj_stys)
    for pred in sn_predicates:
        for s in subj:
            for o in obj:
                if (s, pred, o) in tables.sn_edges:
                    return True
    return False


def relation_pair_valid_sn(
    rel: str,
    subj_stys: Iterable[str],
    obj_stys: Iterable[str],
    tables: MRCMTables,
) -> bool:
    """Tier-2 relation-level surface: the pair satisfies the UMLS Semantic Network.

    *subj_stys* / *obj_stys* are the subject's and object's UMLS semantic-type
    names (a concept may carry several). Returns True iff some (subject STY, SN
    predicate, object STY) is an allowed SN edge, in EITHER argument order (SN
    predicates are directed, but the corpora annotate pairs in a non-canonical
    order, mirroring the Tier-1 MRCM treatment). A relation with no SN predicate
    (converts-to, compared-with), an unknown relation, or an unloaded SN table is
    treated as unconstrained (True) -- absence of an edge set is not a violation.
    """
    preds = tables.tier2_to_sn.get(rel)
    if not preds or not tables.sn_edges:
        return True
    fwd = _sn_edge_exists(preds, subj_stys, obj_stys, tables)
    if fwd:
        return True
    return _sn_edge_exists(preds, obj_stys, subj_stys, tables)


def relation_pair_orientation_sn(
    rel: str,
    subj_stys: Iterable[str],
    obj_stys: Iterable[str],
    tables: MRCMTables,
) -> Optional[str]:
    """Which argument order satisfies the directed SN predicate (see relation_pair_orientation)."""
    preds = tables.tier2_to_sn.get(rel)
    if not preds or not tables.sn_edges:
        return ORIENT_FORWARD
    subj = list(subj_stys)
    obj = list(obj_stys)
    if _sn_edge_exists(preds, subj, obj, tables):
        return ORIENT_FORWARD
    if _sn_edge_exists(preds, obj, subj, tables):
        return ORIENT_REVERSED
    return ORIENT_NONE


# ---------------------------------------------------------------------------
# Tier-2 SN constraint table (Phase 1.5 deliverable, parallel to MRCM JSON)
# ---------------------------------------------------------------------------
def build_sn_constraint_table(tables: MRCMTables) -> dict:
    """Return the SN Tier-2 constraint table keyed by canon relation.

    Shape parallels the MRCM dictionary: each constrained Tier-2 relation lists
    its SN predicate(s) and the allowed (subject STY, object STY) type pairs.
    """
    by_pred: Dict[str, list] = {}
    for subj, pred, obj in tables.sn_edges:
        by_pred.setdefault(pred, []).append([subj, obj])
    out: Dict[str, dict] = {}
    for rel, preds in tables.tier2_to_sn.items():
        allowed: list = []
        for p in preds:
            allowed.extend(by_pred.get(p, []))
        out[rel] = {
            "sn_predicates": list(preds),
            "constrained": bool(preds),
            "allowed_type_pairs": sorted(allowed),
            "allowed_pair_count": len(allowed),
        }
    return {
        "source": "UMLS Semantic Network (SRSTRE2, fully-inherited)",
        "granularity": "UMLS semantic type (STY) names via MRSTY",
        "relations": out,
    }


def dump_sn_constraints(tables: MRCMTables, path: Path) -> Path:
    """Write the SN Tier-2 constraint table to *path* as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(build_sn_constraint_table(tables), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def build_and_dump_sn_constraints(srstre2_path: Path, out_path: Path) -> Path:
    """Standalone Phase 1.5 builder: load SRSTRE2 and write the SN Tier-2 table.

    Independent of MRCM / ancestors, so it can run at Phase 1.5 before the Phase
    1.6 hierarchy pickle exists.
    """
    tables = MRCMTables(
        sn_edges=_load_semantic_network(srstre2_path),
        tier2_to_sn={k: list(v) for k, v in TIER2_TO_SN_PREDICATES.items()},
    )
    return dump_sn_constraints(tables, out_path)


# ---------------------------------------------------------------------------
# Self-check / CLI
# ---------------------------------------------------------------------------
# Pins the intended MRCM truth table against the real constraint file so any
# future change to the parsing or the orientation logic fails loudly here. The
# CSP solver and the Phase 4.3 coherence evaluator both build their tables via
# load_tables, so this check guards the guarantee for both.

def _self_check() -> None:
    import sys

    try:
        import config
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import config
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS

    semantic_classes = ["chemical", "disease", "gene", "variant", "species", "cell_line"]
    tables = load_tables(
        config.MRCM_CONSTRAINTS_JSON,
        config.SNOMED_ANCESTORS_PKL,
        tier1_relations=TIER1_RELATIONS,
        tier2_relations=TIER2_RELATIONS,
        semantic_classes=semantic_classes,
        semantic_network_path=config.UMLS_SEMANTIC_NETWORK_FILES["srstre2"],
    )

    # causative-agent (finding -> substance): disease-chemical valid in BOTH
    # argument orders; same-type pairs invalid.
    assert relation_pair_valid("causative-agent", "disease", "chemical", tables)
    assert relation_pair_valid("causative-agent", "chemical", "disease", tables)
    assert not relation_pair_valid("causative-agent", "chemical", "chemical", tables)
    assert not relation_pair_valid("causative-agent", "disease", "disease", tables)

    # Orientation: annotation order (chemical, disease) must be flagged reversed
    # so a canonical (disease, causative-agent, chemical) triple can be emitted;
    # the SNOMED-canonical order is forward.
    assert relation_pair_orientation("causative-agent", "chemical", "disease", tables) == ORIENT_REVERSED
    assert relation_pair_orientation("causative-agent", "disease", "chemical", tables) == ORIENT_FORWARD

    # Tier-2 relations are unconstrained at the coarse level and keep annotation order.
    assert relation_pair_valid("causes", "chemical", "disease", tables)
    assert relation_pair_orientation("causes", "chemical", "disease", tables) == ORIENT_FORWARD

    # Tier-2 Semantic Network surface (STY-level). 'Pharmacologic Substance treats
    # Disease or Syndrome' is an allowed SN edge; the reverse is not.
    assert tables.sn_edges, "SN edge set failed to load"
    assert relation_pair_valid_sn("treats", ["Pharmacologic Substance"], ["Disease or Syndrome"], tables)
    assert not relation_pair_valid_sn("treats", ["Disease or Syndrome"], ["Pharmacologic Substance"], tables) \
        or relation_pair_orientation_sn("treats", ["Disease or Syndrome"], ["Pharmacologic Substance"], tables) == ORIENT_REVERSED
    # 'causes' with a plausible causal type pair is allowed; a type-impossible
    # pair (disease treats-> substance was reverse; here object of wrong kind).
    assert relation_pair_valid_sn("causes", ["Pharmacologic Substance"], ["Disease or Syndrome"], tables)
    # converts-to has no SN predicate -> unconstrained.
    assert relation_pair_valid_sn("converts-to", ["Organic Chemical"], ["Disease or Syndrome"], tables)

    # Entity-level surface: a real chemical concept lies under 'chemical'; a
    # non-SNOMED type is unconstrained.
    assert concept_under_type("387207008", "chemical", tables)      # Ibuprofen (substance)
    assert concept_under_type("anything", "gene", tables)

    print(f"mrcm_validity self-check OK: {len(tables.valid_pair_for_relation)} pairs, "
          f"{len(tables.domain_range)} Tier-1 domain/range, "
          f"{len(tables.descendants)} descendant sets, "
          f"{len(tables.sn_edges)} SN Tier-2 edges")


if __name__ == "__main__":
    _self_check()
