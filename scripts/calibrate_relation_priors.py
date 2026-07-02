"""Distant-supervision calibration of the Phase 1.4 Tier-1 relation priors.

The Phase 1.4 relation schema (relation_schema.py) assigns each ambiguous source
relation a soft distribution over unified relations, e.g. BC5CDR CID ->
{causes 0.60, causative-agent 0.30, associated-with 0.10}. The Tier-1 fraction
(the causative-agent 0.30) is load-bearing: it is the soft-label mass that gates
Phase 4.4 Tier-1 escalation. Until now those fractions were expert priors with no
empirical grounding.

This script grounds them. For every gold chemical-disease (and disease-disease)
pair the corpus annotates with a causal/associative source relation, it asks a
distant-supervision question: does SNOMED CT itself state the corresponding
Tier-1 attribute between the two mapped concepts (or their generalizations)?

    causative-agent (246075003): SNOMED triple is  finding --causative-agent--> substance
    due-to          (42752001) : SNOMED triple is  finding --due-to--> finding
    after           (255234002): SNOMED triple is  finding --after--> finding

The fraction of corpus pairs that are SNOMED-attestable is an empirical estimate
of how often the Tier-1 reading is ontologically grounded, i.e. a data-driven
target for the Tier-1 prior. Two estimates are reported per group:

    exact     : the two mapped concepts are directly related in SNOMED
    ancestor  : some generalization (self + ancestors) of each side is related,
                which credits corpus concepts that are more specific than the
                SNOMED concept carrying the stated attribute.

Read-only: consumes outputs/phase2/{mapped,relation_mapped} and the Phase 1
SNOMED artifacts, writes a summary JSON + CSV under outputs/phase1/. It does not
mutate the relation schema; a human folds the findings into relation_schema.py.

Usage:
    python scripts/calibrate_relation_priors.py
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, Set, Tuple

try:
    from config import REPO_ROOT, SNOMED_FILES
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, SNOMED_FILES


ATTR_CAUSATIVE_AGENT = "246075003"
ATTR_DUE_TO = "42752001"
ATTR_AFTER = "255234002"
_TARGET_ATTRS = {ATTR_CAUSATIVE_AGENT, ATTR_DUE_TO, ATTR_AFTER}

RELATION_MAPPED_DIR = REPO_ROOT / "outputs" / "phase2" / "relation_mapped"
ANCESTORS_PKL = REPO_ROOT / "outputs" / "phase1" / "snomed_ancestors.pkl"
OUT_JSON = REPO_ROOT / "outputs" / "phase1" / "relation_prior_calibration.json"
OUT_CSV = REPO_ROOT / "outputs" / "phase1" / "relation_prior_calibration.csv"

CORPORA = ("BioRED", "BC5CDR")
SPLITS = ("train", "dev", "test")


# ---------------------------------------------------------------------------
# SNOMED stated-attribute index
# ---------------------------------------------------------------------------
def load_attribute_index() -> Dict[str, Dict[str, Set[str]]]:
    """Return {attr_id: {sourceId: {destinationId, ...}}} for active target rows.

    Uses the inferred relationships snapshot (same source as Phase 2.5); the US
    stated-relationship snapshot carries only retired rows.
    """
    index: Dict[str, Dict[str, Set[str]]] = {a: defaultdict(set) for a in _TARGET_ATTRS}
    path = SNOMED_FILES["relationships"]
    with Path(path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("active") != "1":
                continue
            type_id = row.get("typeId", "")
            if type_id not in _TARGET_ATTRS:
                continue
            index[type_id][row["sourceId"]].add(row["destinationId"])
    return index


def load_ancestors() -> Dict[str, FrozenSet[str]]:
    with ANCESTORS_PKL.open("rb") as fh:
        blob = pickle.load(fh)
    anc = blob.get("ancestors", blob) if isinstance(blob, dict) else {}
    return {
        str(k): frozenset(str(x) for x in v)
        for k, v in anc.items()
        if not str(k).startswith("descendant:")
    }


def _self_and_ancestors(concept_id: str, ancestors: Dict[str, FrozenSet[str]]) -> Set[str]:
    out = {concept_id}
    out.update(ancestors.get(concept_id, frozenset()))
    return out


def attested(
    src_id: str,
    dst_id: str,
    attr_index: Dict[str, Set[str]],
    ancestors: Dict[str, FrozenSet[str]],
) -> Tuple[bool, bool]:
    """Return (exact, ancestor_aware) attestation of a directed src->dst attribute."""
    exact = dst_id in attr_index.get(src_id, ())
    src_set = _self_and_ancestors(src_id, ancestors)
    dst_set = _self_and_ancestors(dst_id, ancestors)
    anc_ok = False
    for s in src_set:
        dests = attr_index.get(s)
        if dests and not dst_set.isdisjoint(dests):
            anc_ok = True
            break
    return exact, (exact or anc_ok)


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------
def iter_relations() -> Iterator[dict]:
    """Yield one record per gold relation whose subject and object both carry a
    mapped SNOMED id, with fields needed for attestation."""
    for corpus in CORPORA:
        for split in SPLITS:
            path = RELATION_MAPPED_DIR / corpus / f"{split}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    doc = json.loads(line)
                    ents = doc.get("entities", [])
                    for rel in doc.get("relations", []):
                        si, oi = rel.get("subject_idx"), rel.get("object_idx")
                        if si is None or oi is None or si >= len(ents) or oi >= len(ents):
                            continue
                        se, oe = ents[si], ents[oi]
                        s_sid = se.get("mapped_snomed_id")
                        o_sid = oe.get("mapped_snomed_id")
                        if not s_sid or not o_sid:
                            continue
                        yield {
                            "corpus": corpus,
                            "source_relation": rel.get("source_relation_type", ""),
                            "subj_class": se.get("semantic_class", ""),
                            "obj_class": oe.get("semantic_class", ""),
                            "subj_snomed": str(s_sid),
                            "obj_snomed": str(o_sid),
                        }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _disease_chemical(rec: dict) -> Tuple[str, str] | None:
    """Return (disease_snomed, chemical_snomed) for a chem-disease pair, else None."""
    if rec["subj_class"] == "disease" and rec["obj_class"] == "chemical":
        return rec["subj_snomed"], rec["obj_snomed"]
    if rec["subj_class"] == "chemical" and rec["obj_class"] == "disease":
        return rec["obj_snomed"], rec["subj_snomed"]
    return None


def calibrate() -> dict:
    attr_index = load_attribute_index()
    ancestors = load_ancestors()
    ca = attr_index[ATTR_CAUSATIVE_AGENT]
    dt = attr_index[ATTR_DUE_TO]
    af = attr_index[ATTR_AFTER]

    # Per (corpus, source_relation, subj_class, obj_class) tallies.
    groups: Dict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "ca_exact": 0, "ca_anc": 0, "dt_anc": 0, "af_anc": 0}
    )

    for rec in iter_relations():
        key = (rec["corpus"], rec["source_relation"], rec["subj_class"], rec["obj_class"])
        g = groups[key]
        g["pairs"] += 1

        dc = _disease_chemical(rec)
        if dc is not None:
            disease, chemical = dc
            ex, an = attested(disease, chemical, ca, ancestors)
            g["ca_exact"] += int(ex)
            g["ca_anc"] += int(an)

        # due-to / after fire on disease-disease pairs; check both directions.
        if rec["subj_class"] == "disease" and rec["obj_class"] == "disease":
            s, o = rec["subj_snomed"], rec["obj_snomed"]
            _, dt1 = attested(s, o, dt, ancestors)
            _, dt2 = attested(o, s, dt, ancestors)
            g["dt_anc"] += int(dt1 or dt2)
            _, af1 = attested(s, o, af, ancestors)
            _, af2 = attested(o, s, af, ancestors)
            g["af_anc"] += int(af1 or af2)

    rows = []
    for key in sorted(groups):
        corpus, srel, sc, oc = key
        g = groups[key]
        n = g["pairs"]
        rows.append(
            {
                "corpus": corpus,
                "source_relation": srel,
                "subj_class": sc,
                "obj_class": oc,
                "pairs": n,
                "causative_agent_exact": g["ca_exact"],
                "causative_agent_exact_rate": round(g["ca_exact"] / n, 4) if n else 0.0,
                "causative_agent_anc": g["ca_anc"],
                "causative_agent_anc_rate": round(g["ca_anc"] / n, 4) if n else 0.0,
                "due_to_anc": g["dt_anc"],
                "due_to_anc_rate": round(g["dt_anc"] / n, 4) if n else 0.0,
                "after_anc": g["af_anc"],
                "after_anc_rate": round(g["af_anc"] / n, 4) if n else 0.0,
            }
        )

    snomed_counts = {
        "causative_agent_source_concepts": len(ca),
        "due_to_source_concepts": len(dt),
        "after_source_concepts": len(af),
    }
    return {"snomed_index": snomed_counts, "groups": rows}


def main() -> None:
    result = calibrate()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    fieldnames = list(result["groups"][0].keys()) if result["groups"] else []
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(result["groups"])

    print(f"SNOMED index: {result['snomed_index']}")
    print(f"{'corpus':7} {'source_relation':22} {'subj':8} {'obj':8} "
          f"{'pairs':>6} {'CA_ex':>6} {'CA_anc':>7} {'DT_anc':>7} {'AF_anc':>7}")
    for r in result["groups"]:
        # Focus the console view on chem-disease and disease-disease groups.
        if not (
            {r["subj_class"], r["obj_class"]} == {"chemical", "disease"}
            or r["subj_class"] == r["obj_class"] == "disease"
        ):
            continue
        print(f"{r['corpus']:7} {r['source_relation']:22} {r['subj_class']:8} "
              f"{r['obj_class']:8} {r['pairs']:>6} "
              f"{r['causative_agent_exact_rate']:>6} {r['causative_agent_anc_rate']:>7} "
              f"{r['due_to_anc_rate']:>7} {r['after_anc_rate']:>7}")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
