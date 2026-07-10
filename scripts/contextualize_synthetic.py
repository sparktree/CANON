"""Ground SNOMED Tier-1 triples in real PubTator/PubMed document context."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


SYNTHETIC = config.PHASE2_DIR / "synthetic" / "train.jsonl"
SILVER = config.PHASE2_DIR / "silver" / "PubTator3" / "train.jsonl"
OUTPUT = config.PHASE2_DIR / "synthetic" / "contextual_train.jsonl"


def contextualize(synthetic_path: Path = SYNTHETIC, context_paths: List[Path] | None = None,
                  output_path: Path = OUTPUT, min_causative: int = 3000,
                  canonical_dir: Path | None = None) -> dict:
    context_paths = context_paths or [SILVER]
    triples: Dict[Tuple[str, str], str] = {}
    for doc in read_jsonl(synthetic_path):
        for rel in doc.relations:
            a = doc.entities[rel.subject_idx].mapped_snomed_id
            b = doc.entities[rel.object_idx].mapped_snomed_id
            if a and b and rel.target_relation:
                triples[(str(a), str(b))] = rel.target_relation

    seen = set()
    counts = Counter()

    def docs() -> Iterator[Document]:
        for context_path in context_paths:
            if not context_path.exists():
                continue
            for source in read_jsonl(context_path):
                by_concept: Dict[str, List[Tuple[int, EntityMention]]] = {}
                for idx, entity in enumerate(source.entities):
                    if entity.mapped_snomed_id:
                        by_concept.setdefault(str(entity.mapped_snomed_id), []).append((idx, entity))
                present = list(by_concept)
                for a in present:
                    for b in present:
                        if a == b:
                            continue
                        label = triples.get((a, b))
                        if label is None:
                            continue
                        ea = by_concept[a][0][1]
                        eb = by_concept[b][0][1]
                        key = (source.corpus, source.pmid, a, b, label)
                        if key in seen:
                            continue
                        seen.add(key)
                        counts[label] += 1
                        entities = [
                            EntityMention(**{**ea.__dict__, "id": "T1", "mapping_confidence": 0.55}),
                            EntityMention(**{**eb.__dict__, "id": "T2", "mapping_confidence": 0.55}),
                        ]
                        relation = Relation(
                            subject_idx=0, object_idx=1, source_relation_type=label,
                            target_relation=label, tier=1, target_probability=1.0,
                            confidence=0.55,
                            target_candidates=[{"target_relation": label, "tier": 1, "probability": 1.0}],
                            extra={"distant_supervision": "snomed-stated-plus-pubtator-context",
                                   "source_pmid": source.pmid},
                        )
                        yield Document(
                            pmid=f"SNOMED_CTX_{source.pmid}_{len(seen):07d}",
                            corpus="SNOMED_contextual", split="train", title=source.title,
                            abstract=source.abstract, text=source.text,
                            entities=entities, relations=[relation],
                        )

    canonical_dir = canonical_dir or output_path.parent / "jsonld" / "contextual"
    written = write_jsonld_documents(docs(), canonical_dir)
    cached = derive_jsonl_cache(canonical_dir, output_path)
    if cached != written:
        raise RuntimeError(f"canonical/cache count mismatch: {written} != {cached}")
    summary = {"documents": written, "counts_by_attribute": dict(counts),
               "minimum_causative_agent": min_causative,
               "causative_agent_requirement_met": counts["causative-agent"] >= min_causative,
               "contexts": [str(p) for p in context_paths], "output": str(output_path),
               "canonical_dir": str(canonical_dir)}
    summary_path = output_path.parent / "contextual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if counts["causative-agent"] < min_causative:
        raise RuntimeError(
            f"real-context causative-agent coverage {counts['causative-agent']} < {min_causative}; "
            "add retrieved PubMed/PubTator context before production training")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=str(SYNTHETIC))
    parser.add_argument("--contexts", nargs="+", default=[str(SILVER)])
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--min-causative", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(contextualize(Path(args.synthetic), [Path(x) for x in args.contexts],
                                   Path(args.output), args.min_causative), indent=2))


if __name__ == "__main__":
    main()
