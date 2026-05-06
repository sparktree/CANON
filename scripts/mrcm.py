"""SNOMED CT Machine-Readable Concept Model formalization (CANON Phase 1.5).

Parses the three MRCM reference sets shipped with the SNOMED CT RF2 release
and converts them into CSP constraint templates keyed by the unified
relation labels declared in `relation_schema.TIER1_RELATIONS`.

Inputs (under Data/SNOMED_CT_RF2_*/Snapshot/Refset/Metadata/):
    der2_sssssssRefset_MRCMDomainSnapshot_*.txt
    der2_cissccRefset_MRCMAttributeDomainSnapshot_*.txt
    der2_ssccRefset_MRCMAttributeRangeSnapshot_*.txt

Output:
    outputs/phase1/mrcm_constraints.json -- structured constraint dict; this
    is what the CSP solver in Phase 3.5 consumes. Per-attribute SNOMED IDs
    are preserved so Phase 1.6's hierarchy graph can compute descendant
    closures at solve time.

Constraint shape per Tier-1 relation:
    {
      "<unified_relation_label>": {
        "snomed_attribute_id":   "<sctid>",
        "snomed_attribute_name": "<FSN>",
        "domain": [
            {"domain_concept_id": "<sctid>",
             "domain_constraint_ecl": "<< 404684003 |Clinical finding|",
             "domain_root_concept_ids": ["404684003"],
             "cardinality": "0..*",
             "in_group_cardinality": "0..1",
             "rule_strength": "Mandatory",
             "content_type": "All SNOMED CT content",
             "grouped": True},
            ...
        ],
        "range": [
            {"range_constraint_ecl": "<< 442083009 |Anatomical or acquired body structure|",
             "range_root_concept_ids": ["442083009"],
             "rule_strength": "Mandatory",
             "content_type": "All SNOMED CT content"},
            ...
        ]
      },
      ...
    }

The `*_root_concept_ids` lists are extracted by stripping ECL operators / FSN
labels and lifting bare SCTIDs. They serve as Phase 1.6 entry points: a CSP
check of the form "is concept C in the range of attribute A?" reduces to "is
C a descendant of any root_concept_id under the appropriate operator's
closure rule (`<<` = self-or-descendants, `<` = descendants-only, `=` = exact)?"
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

try:
    from config import MRCM_FILES, REPO_ROOT, SNOMED_FILES
    import relation_schema
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import MRCM_FILES, REPO_ROOT, SNOMED_FILES
    import relation_schema


OUTPUT_DIR = REPO_ROOT / "outputs" / "phase1"


# ---------------------------------------------------------------------------
# Unified-relation -> SNOMED attribute concept ID
# These IDs match the comment annotations in relation_schema.TIER1_RELATIONS.
# ---------------------------------------------------------------------------
RELATION_TO_ATTRIBUTE_ID: Dict[str, str] = {
    "causative-agent":        "246075003",
    "finding-site":           "363698007",
    "associated-morphology":  "116676008",
    "due-to":                 "42752001",
    "after":                  "255234002",
}


# ---------------------------------------------------------------------------
# MRCM controlled vocabularies (SNOMED metadata concept IDs)
# ---------------------------------------------------------------------------
_RULE_STRENGTH = {
    "723597001": "Mandatory CM rule",
    "723598006": "Optional CM rule",
}
_CONTENT_TYPE = {
    "723596005": "All SNOMED CT content",
    "723593002": "Precoordinated content only",
    "723594008": "Postcoordinated content only",
    "723595009": "New precoordinated content only",
}

# SNOMED description type IDs
_FSN_TYPE = "900000000000003001"
_SYN_TYPE = "900000000000013009"


# ---------------------------------------------------------------------------
# RF2 reading helpers
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
# ECL extraction
# ---------------------------------------------------------------------------
# Captures: optional operator (<<, <, <=, >=, >, =, ^), then SCTID, then optional |term|.
_ECL_TOKEN = re.compile(
    r"(?P<op><<\!?|>>\!?|<\!?|>\!?|<=|>=|=|\^|!=)?\s*"
    r"(?P<sctid>\d{6,18})"
    r"(?:\s*\|(?P<term>[^|]+)\|)?"
)


def parse_ecl_roots(expression: str) -> List[Dict[str, str]]:
    """Extract (operator, sctid, term) tuples from an ECL constraint string.

    The MRCM domain/range constraints we care about are simple unions of
    `<<sctid|term|`, `<sctid|term|`, or bare `=sctid|term|` clauses joined
    by AND/OR/MINUS. We don't need a full ECL parser at this stage --
    Phase 1.6's hierarchy graph will materialize the descendant closures.
    """
    if not expression:
        return []
    out: List[Dict[str, str]] = []
    for m in _ECL_TOKEN.finditer(expression):
        op = (m.group("op") or "=").strip()
        out.append(
            {
                "operator": op,
                "concept_id": m.group("sctid"),
                "term": (m.group("term") or "").strip(),
            }
        )
    return out


def _root_concept_ids(parsed: List[Dict[str, str]]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for tok in parsed:
        cid = tok["concept_id"]
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


# ---------------------------------------------------------------------------
# Description lookup -- we resolve attribute names lazily, only for the
# attribute IDs that actually appear in the MRCM refsets.
# ---------------------------------------------------------------------------
def _load_descriptions(needed: Set[str]) -> Dict[str, str]:
    if not needed:
        return {}
    desc_path = SNOMED_FILES["descriptions"]
    if not desc_path.exists():
        return {}
    out: Dict[str, str] = {}
    with desc_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if row.get("active") != "1":
                continue
            cid = row.get("conceptId", "")
            if cid not in needed:
                continue
            type_id = row.get("typeId", "")
            term = row.get("term", "")
            existing = out.get(cid)
            # Prefer FSN over Synonym; first-seen wins within a type.
            if existing is None or (type_id == _FSN_TYPE and not existing.endswith(")")):
                out[cid] = term
    return out


# ---------------------------------------------------------------------------
# MRCM parsers
# ---------------------------------------------------------------------------
def parse_mrcm_domain() -> Dict[str, Dict[str, object]]:
    """Return {domain_concept_id: {domain_constraint, parent_domain, ...}}."""
    domains: Dict[str, Dict[str, object]] = {}
    for row in _read_rf2(MRCM_FILES["domain"]):
        if not _active(row):
            continue
        cid = row["referencedComponentId"]
        domains[cid] = {
            "domain_concept_id": cid,
            "domain_constraint_ecl": row.get("domainConstraint", ""),
            "parent_domain": row.get("parentDomain", "") or None,
            "proximal_primitive_constraint": row.get("proximalPrimitiveConstraint", ""),
        }
    return domains


def parse_mrcm_attribute_domain() -> Dict[str, List[Dict[str, object]]]:
    """Return {attribute_concept_id: [domain_row, ...]}."""
    by_attr: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in _read_rf2(MRCM_FILES["attribute_domain"]):
        if not _active(row):
            continue
        attr = row["referencedComponentId"]
        by_attr[attr].append(
            {
                "domain_concept_id": row["domainId"],
                "grouped": row.get("grouped") == "1",
                "cardinality": row.get("attributeCardinality", ""),
                "in_group_cardinality": row.get("attributeInGroupCardinality", ""),
                "rule_strength": _RULE_STRENGTH.get(row.get("ruleStrengthId", ""), row.get("ruleStrengthId", "")),
                "content_type": _CONTENT_TYPE.get(row.get("contentTypeId", ""), row.get("contentTypeId", "")),
            }
        )
    return by_attr


def parse_mrcm_attribute_range() -> Dict[str, List[Dict[str, object]]]:
    """Return {attribute_concept_id: [range_row, ...]}."""
    by_attr: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in _read_rf2(MRCM_FILES["attribute_range"]):
        if not _active(row):
            continue
        attr = row["referencedComponentId"]
        ecl = row.get("rangeConstraint", "")
        parsed = parse_ecl_roots(ecl)
        by_attr[attr].append(
            {
                "range_constraint_ecl": ecl,
                "range_root_concept_ids": _root_concept_ids(parsed),
                "attribute_rule": row.get("attributeRule", ""),
                "rule_strength": _RULE_STRENGTH.get(row.get("ruleStrengthId", ""), row.get("ruleStrengthId", "")),
                "content_type": _CONTENT_TYPE.get(row.get("contentTypeId", ""), row.get("contentTypeId", "")),
            }
        )
    return by_attr


# ---------------------------------------------------------------------------
# Build the merged constraint dictionary
# ---------------------------------------------------------------------------
def build_constraints() -> Dict[str, object]:
    domains = parse_mrcm_domain()
    attr_domains = parse_mrcm_attribute_domain()
    attr_ranges = parse_mrcm_attribute_range()

    needed_names: Set[str] = set()
    needed_names.update(attr_domains.keys())
    needed_names.update(attr_ranges.keys())
    needed_names.update(domains.keys())
    for entries in attr_domains.values():
        for e in entries:
            needed_names.add(e["domain_concept_id"])
    for entries in attr_ranges.values():
        for e in entries:
            needed_names.update(e["range_root_concept_ids"])
    name_index = _load_descriptions(needed_names)

    def _name(cid: str) -> str:
        return name_index.get(cid, "")

    # Enrich each domain/range row with the resolved domain root concept IDs
    # and human-readable names.
    attributes_by_id: Dict[str, Dict[str, object]] = {}
    for attr_id in sorted(set(attr_domains) | set(attr_ranges)):
        domain_rows = []
        for d in attr_domains.get(attr_id, []):
            domain_meta = domains.get(d["domain_concept_id"], {})
            ecl = str(domain_meta.get("domain_constraint_ecl", ""))
            parsed = parse_ecl_roots(ecl)
            domain_rows.append(
                {
                    **d,
                    "domain_concept_name": _name(d["domain_concept_id"]),
                    "domain_constraint_ecl": ecl,
                    "domain_root_concept_ids": _root_concept_ids(parsed) or [d["domain_concept_id"]],
                }
            )
        range_rows = []
        for r in attr_ranges.get(attr_id, []):
            range_rows.append({
                **r,
                "range_root_concept_names": [_name(c) for c in r["range_root_concept_ids"]],
            })
        attributes_by_id[attr_id] = {
            "snomed_attribute_id": attr_id,
            "snomed_attribute_name": _name(attr_id),
            "domains": domain_rows,
            "ranges": range_rows,
        }

    # CSP-ready, keyed by unified Tier-1 relation label.
    relation_constraints: Dict[str, object] = {}
    for label, attr_id in RELATION_TO_ATTRIBUTE_ID.items():
        attr_block = attributes_by_id.get(attr_id)
        if attr_block is None:
            relation_constraints[label] = {
                "snomed_attribute_id": attr_id,
                "snomed_attribute_name": _name(attr_id),
                "domains": [],
                "ranges": [],
                "warning": "No active MRCM rows for this attribute in the loaded release.",
            }
            continue
        relation_constraints[label] = dict(attr_block)

    # Sanity: every Tier-1 relation in relation_schema must have an attribute mapping.
    missing = [r for r in relation_schema.TIER1_RELATIONS if r not in RELATION_TO_ATTRIBUTE_ID]
    extra = [r for r in RELATION_TO_ATTRIBUTE_ID if r not in relation_schema.TIER1_RELATIONS]

    return {
        "metadata": {
            "release_files": {k: str(v.name) for k, v in MRCM_FILES.items()},
            "domain_rows": len(domains),
            "attribute_domain_rows": sum(len(v) for v in attr_domains.values()),
            "attribute_range_rows": sum(len(v) for v in attr_ranges.values()),
            "tier1_relations_without_attribute_mapping": missing,
            "attribute_mappings_outside_tier1": extra,
        },
        "domains": {
            cid: {**dom, "domain_concept_name": _name(cid)}
            for cid, dom in domains.items()
        },
        "attributes_by_id": attributes_by_id,
        "relation_constraints": relation_constraints,
    }


# ---------------------------------------------------------------------------
# Disk + accessor API
# ---------------------------------------------------------------------------
JSON_PATH = OUTPUT_DIR / "mrcm_constraints.json"


def dump_json(constraints: Optional[Dict[str, object]] = None, path: Optional[Path] = None) -> Path:
    if constraints is None:
        constraints = build_constraints()
    if path is None:
        path = JSON_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(constraints, fh, indent=2, ensure_ascii=False)
    return path


def load_json(path: Optional[Path] = None) -> Dict[str, object]:
    if path is None:
        path = JSON_PATH
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_constraint(relation_label: str, constraints: Optional[Dict[str, object]] = None) -> Optional[Dict[str, object]]:
    if constraints is None:
        constraints = load_json()
    return constraints.get("relation_constraints", {}).get(relation_label)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _summarize(constraints: Dict[str, object]) -> None:
    meta = constraints["metadata"]
    print(
        f"[mrcm] domain_rows={meta['domain_rows']}  "
        f"attribute_domain_rows={meta['attribute_domain_rows']}  "
        f"attribute_range_rows={meta['attribute_range_rows']}"
    )
    if meta["tier1_relations_without_attribute_mapping"]:
        print(f"[mrcm] WARNING: Tier-1 relations missing attribute IDs: {meta['tier1_relations_without_attribute_mapping']}")
    print("[mrcm] CSP-ready constraints (one row per Tier-1 relation):")
    for label, block in constraints["relation_constraints"].items():
        attr_id = block.get("snomed_attribute_id", "")
        name = block.get("snomed_attribute_name", "")
        n_dom = len(block.get("domains", []))
        n_rng = len(block.get("ranges", []))
        warn = " [WARNING] no MRCM rows" if block.get("warning") else ""
        print(f"    {label:<22s} -> {attr_id} ({name})  domains={n_dom}  ranges={n_rng}{warn}")


def main(verbose: bool = True) -> Path:
    constraints = build_constraints()
    out = dump_json(constraints)
    if verbose:
        _summarize(constraints)
        print(f"[mrcm] wrote {out}")
    return out


if __name__ == "__main__":
    main()
