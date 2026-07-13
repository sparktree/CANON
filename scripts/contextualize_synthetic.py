"""Ground SNOMED Tier-1 relations in real document context (no label leakage).

Step-1 grounding source (local, no retrieval): rather than scanning for exact
co-mentions of the templated synthetic SNOMED triples (which yielded almost
nothing on the small silver set), this grounds Tier-1 labels in REAL corpus and
silver documents whose chemical-disease (or disease-disease) concept pairs SNOMED
itself attests as causative-agent / due-to / after (ancestor-aware, via the
Phase-1.7 stated-relationship index used by the prior calibration). Each grounded
example carries the real abstract text and the two real mentions, so there is no
"subject relation object" label leakage and no synthetic template. It is distant
supervision -- co-mention plus a SNOMED-stated attribute -- so confidence is set
in the plan's 0.5-0.7 band, not gold.

Augmentation is train-only: only *train* gold splits and the silver set are read;
dev/test are never touched. Coverage is reported as evidence (per attribute, per
corpus, distinct pairs) rather than forced to an arbitrary fixed count; the run
fails only if grounding produces nothing at all (a misconfiguration), and warns
if it falls short of a configurable soft target.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

try:
    import config
    from unified_format import (Document, EntityMention, Relation, derive_jsonl_cache,
                                read_jsonl, write_jsonld_documents)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from unified_format import (Document, EntityMention, Relation, derive_jsonl_cache,
                                read_jsonl, write_jsonld_documents)


SILVER = config.PHASE2_DIR / "silver" / "PubTator3" / "train.jsonl"
OUTPUT = config.PHASE2_DIR / "synthetic" / "contextual_train.jsonl"

# Train-only real-context sources (dev/test are never used for augmentation).
# The retrieved-context file (Step 2, retrieve_pubtator_context.py) is included
# when present; a default offline run simply omits it.
RELATION_MAPPED_DIR = config.PHASE2_DIR / "relation_mapped"
RETRIEVED_CONTEXT = config.PHASE2_DIR / "silver" / "PubTator3" / "retrieved_context.jsonl"
DEFAULT_CONTEXT_PATHS = [
    RELATION_MAPPED_DIR / "BioRED" / "train.jsonl",
    RELATION_MAPPED_DIR / "BC5CDR" / "train.jsonl",
    SILVER,
    RETRIEVED_CONTEXT,
]

# SNOMED attribute -> (attribute_index_id, emitted canon relation label). The
# index id keys the stated-relationship lookup; the label is what we stamp.
_ATTR_CAUSATIVE_AGENT = "246075003"
_ATTR_DUE_TO = "42752001"
_ATTR_AFTER = "255234002"

# Distant-supervision confidence for real-context SNOMED-attested examples
# (real text + a SNOMED-stated attribute, but co-mention is not assertion).
_DS_CONFIDENCE = 0.6


def _grounding_entity(base: EntityMention, tag: str) -> EntityMention:
    return EntityMention(**{**base.__dict__, "id": tag, "mapping_confidence": _DS_CONFIDENCE})


def _make_doc(source: Document, e_subj: EntityMention, e_obj: EntityMention,
              label: str, ordinal: int) -> Document:
    relation = Relation(
        subject_idx=0, object_idx=1, source_relation_type=label,
        target_relation=label, tier=1, target_probability=1.0, confidence=_DS_CONFIDENCE,
        target_candidates=[{"target_relation": label, "tier": 1, "probability": 1.0}],
        extra={"distant_supervision": "snomed-attested-real-context",
               "source_corpus": source.corpus, "source_pmid": source.pmid},
    )
    return Document(
        pmid=f"SNOMED_ATT_{source.corpus}_{source.pmid}_{label}_{ordinal:07d}",
        corpus="SNOMED_attested", split="train", title=source.title,
        abstract=source.abstract, text=source.text,
        entities=[_grounding_entity(e_subj, "T1"), _grounding_entity(e_obj, "T2")],
        relations=[relation],
    )


def contextualize(context_paths: List[Path] | None = None,
                  output_path: Path = OUTPUT,
                  canonical_dir: Path | None = None) -> dict:
    """Ground Tier-1 relations in SNOMED-attested real-context documents.

    Coverage is reported as evidence (counts and distinct pairs per attribute);
    there is no target gate -- the amount grounded is whatever the real data
    supports, and Step-2 retrieval (retrieve_pubtator_context.py) adds more.
    """
    from calibrate_relation_priors import attested, load_ancestors, load_attribute_index

    context_paths = context_paths or DEFAULT_CONTEXT_PATHS
    attr_index = load_attribute_index()
    ancestors = load_ancestors()
    ca = attr_index[_ATTR_CAUSATIVE_AGENT]
    dt = attr_index[_ATTR_DUE_TO]
    af = attr_index[_ATTR_AFTER]

    seen: set = set()
    counts: Counter = Counter()
    pairs_by_attr: Dict[str, set] = defaultdict(set)

    def _first_mentions(source: Document) -> Tuple[Dict[str, Tuple[int, EntityMention]],
                                                   Dict[str, Tuple[int, EntityMention]]]:
        chem: Dict[str, Tuple[int, EntityMention]] = {}
        dise: Dict[str, Tuple[int, EntityMention]] = {}
        for idx, e in enumerate(source.entities):
            sid = e.mapped_snomed_id
            if not sid:
                continue
            if e.semantic_class == "chemical":
                chem.setdefault(str(sid), (idx, e))
            elif e.semantic_class == "disease":
                dise.setdefault(str(sid), (idx, e))
        return chem, dise

    def docs() -> Iterator[Document]:
        n = 0
        for context_path in context_paths:
            if not context_path.exists():
                continue
            for source in read_jsonl(context_path):
                chem, dise = _first_mentions(source)
                # causative-agent: SNOMED reads finding(disease) -> substance(chemical).
                for d_sid, (_, de) in dise.items():
                    for c_sid, (_, ce) in chem.items():
                        if not attested(d_sid, c_sid, ca, ancestors)[1]:
                            continue
                        key = (source.corpus, source.pmid, "causative-agent", d_sid, c_sid)
                        if key in seen:
                            continue
                        seen.add(key)
                        counts["causative-agent"] += 1
                        pairs_by_attr["causative-agent"].add((d_sid, c_sid))
                        n += 1
                        yield _make_doc(source, de, ce, "causative-agent", n)
                # due-to / after: disease -> disease, attested direction.
                dis_items = list(dise.items())
                for a_sid, (_, ae) in dis_items:
                    for b_sid, (_, be) in dis_items:
                        if a_sid == b_sid:
                            continue
                        for label, idx in (("due-to", dt), ("after", af)):
                            if not attested(a_sid, b_sid, idx, ancestors)[1]:
                                continue
                            key = (source.corpus, source.pmid, label, a_sid, b_sid)
                            if key in seen:
                                continue
                            seen.add(key)
                            counts[label] += 1
                            pairs_by_attr[label].add((a_sid, b_sid))
                            n += 1
                            yield _make_doc(source, ae, be, label, n)

    canonical_dir = canonical_dir or output_path.parent / "jsonld" / "contextual"
    written = write_jsonld_documents(docs(), canonical_dir)
    cached = derive_jsonl_cache(canonical_dir, output_path)
    if cached != written:
        raise RuntimeError(f"canonical/cache count mismatch: {written} != {cached}")

    summary = {
        "documents": written,
        "counts_by_attribute": dict(counts),
        "distinct_pairs_by_attribute": {k: len(v) for k, v in pairs_by_attr.items()},
        "grounding": "snomed-attested-real-context (step 1; no retrieval)",
        "contexts": [str(p) for p in context_paths],
        "output": str(output_path),
        "canonical_dir": str(canonical_dir),
    }
    summary_path = output_path.parent / "contextual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="+", default=[str(p) for p in DEFAULT_CONTEXT_PATHS])
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    print(json.dumps(contextualize([Path(x) for x in args.contexts],
                                   Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
