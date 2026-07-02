"""Shared MRCM validity logic for the CSP solver and the coherence evaluator.

Single source of truth for the two constraint surfaces that the Phase 3.5 CSP
solver ENFORCES and the Phase 4.3 coherence evaluator MEASURES:

    entity-level   : concept_under_type -- the normalized concept lies under the
                     NER type's SNOMED anchor(s).
    relation-level : relation_pair_valid -- a Tier-1 relation's subject/object
                     semantic types satisfy the MRCM domain/range. The check is
                     orientation-robust: SNOMED attributes are directed
                     (causative-agent reads finding -> substance) but the corpora
                     annotate the pair in a non-canonical order, so a pair is
                     valid if EITHER argument order satisfies domain/range.
                     relation_pair_orientation reports which order was the valid
                     one, so a canonical (finding, attribute, substance) triple
                     can be emitted / evaluated.

Both the solver and the evaluator import THIS module, so the guarantee the solver
provides and the coherence the evaluator reports cannot drift apart. The module
is intentionally free of torch / dataset imports -- it is cheap to import from
either side, and the relation-tier sets and semantic-class list are passed in by
the caller rather than imported here (avoids a cycle through csp_solver /
canon_dataset).
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

# Orientation tags returned by relation_pair_orientation.
ORIENT_FORWARD = "forward"    # subject satisfies domain, object satisfies range
ORIENT_REVERSED = "reversed"  # object satisfies domain, subject satisfies range
ORIENT_NONE = None            # no argument order satisfies domain/range


@dataclass
class MRCMTables:
    """Precomputed MRCM compatibility tables shared by solver and evaluator."""
    type_to_anchors: Dict[str, List[str]] = field(default_factory=dict)
    descendants: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    # rel -> (domain_root_ids, range_root_ids)
    domain_range: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = field(default_factory=dict)
    # (relation_label, type_a, type_b) -> bool. Tier-2 always True.
    valid_pair_for_relation: Dict[Tuple[str, str, str], bool] = field(default_factory=dict)


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


def load_tables(
    mrcm_path: Path,
    ancestors_path: Path,
    *,
    tier1_relations: Iterable[str],
    tier2_relations: Iterable[str],
    semantic_classes: Iterable[str],
    logger: Optional[logging.Logger] = None,
) -> MRCMTables:
    """Build the shared MRCM tables once per session."""
    logger = logger or logging.getLogger("mrcm_validity")
    tables = MRCMTables(type_to_anchors=copy.deepcopy(TYPE_ANCHORS))
    tables.descendants = _load_descendants(ancestors_path)

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

    # Tier-2 relations are unconstrained.
    for rel in tier2_relations:
        for ta in list(semantic_classes) + ["none"]:
            for tb in list(semantic_classes) + ["none"]:
                tables.valid_pair_for_relation[(rel, ta, tb)] = True

    logger.info(
        f"loaded MRCM tables: {len(tables.descendants)} descendant sets; "
        f"{len(tables.domain_range)} Tier-1 domain/range; "
        f"{len(tables.valid_pair_for_relation)} (relation, ta, tb) pairs"
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

    # Tier-2 relations are unconstrained and keep annotation order.
    assert relation_pair_valid("causes", "chemical", "disease", tables)
    assert relation_pair_orientation("causes", "chemical", "disease", tables) == ORIENT_FORWARD

    # Entity-level surface: a real chemical concept lies under 'chemical'; a
    # non-SNOMED type is unconstrained.
    assert concept_under_type("387207008", "chemical", tables)      # Ibuprofen (substance)
    assert concept_under_type("anything", "gene", tables)

    print(f"mrcm_validity self-check OK: {len(tables.valid_pair_for_relation)} pairs, "
          f"{len(tables.domain_range)} Tier-1 domain/range, "
          f"{len(tables.descendants)} descendant sets")


if __name__ == "__main__":
    _self_check()
