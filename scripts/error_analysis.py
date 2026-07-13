"""Build the blinded Phase 4.6 manual error-review artifact."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


CATEGORIES = (
    "mapping-failure",
    "coverage-gap",
    "constraint-over-restriction",
    "linguistic-failure",
)


def _entity_type(entity: dict):
    return entity.get("ner_type") or {
        "chemical": "substance", "disease": "clinical_finding",
    }.get(entity.get("semantic_class"), entity.get("semantic_class"))


def _read(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _has_error(gold: dict, prediction: dict, assignment: str) -> bool:
    predicted = prediction.get(assignment, {})
    gold_entities = {
        (e.get("span_start"), e.get("span_end"), _entity_type(e))
        for e in gold.get("entities", [])
    }
    pred_entities = {
        (e.get("span_start"), e.get("span_end"), e.get("type"))
        for e in predicted.get("entities", [])
    }
    if gold_entities != pred_entities:
        return True
    for idx, entity in enumerate(predicted.get("entities", [])):
        if idx >= len(gold.get("entities", [])):
            return True
        match = next((g for g in gold["entities"]
                      if (g.get("span_start"), g.get("span_end")) ==
                      (entity.get("span_start"), entity.get("span_end"))), None)
        if match and match.get("mapped_snomed_id") and str(entity.get("concept")) != str(match["mapped_snomed_id"]):
            return True
    gold_relations = set()
    for relation in gold.get("relations", []):
        if relation.get("target_relation") and relation.get("subject_idx", -1) < len(gold.get("entities", [])) and relation.get("object_idx", -1) < len(gold.get("entities", [])):
            subject = gold["entities"][relation["subject_idx"]]
            obj = gold["entities"][relation["object_idx"]]
            gold_relations.add((subject.get("original_code"), obj.get("original_code"),
                                relation["target_relation"]))
    gold_by_span = {(e.get("span_start"), e.get("span_end")): e for e in gold.get("entities", [])}
    predicted_relations = set()
    entities = predicted.get("entities", [])
    for pair in predicted.get("pairs", []):
        if pair.get("relation") == "no-relation" or pair.get("i", -1) >= len(entities) or pair.get("j", -1) >= len(entities):
            continue
        subject = gold_by_span.get((entities[pair["i"]].get("span_start"), entities[pair["i"]].get("span_end")))
        obj = gold_by_span.get((entities[pair["j"]].get("span_start"), entities[pair["j"]].get("span_end")))
        if subject and obj:
            predicted_relations.add((subject.get("original_code"), obj.get("original_code"),
                                     pair.get("relation")))
    return predicted_relations != gold_relations


def sample_errors(predictions_path: Path, gold_path: Path, output_path: Path,
                  size: int = 100, seed: int = 42, assignment: str = "csp") -> Dict[str, int]:
    predictions = {(row.get("corpus"), str(row.get("pmid"))): row
                   for row in _read(predictions_path)}
    candidates = []
    for gold in _read(gold_path):
        key = (gold.get("corpus"), str(gold.get("pmid")))
        prediction = predictions.get(key)
        if prediction is None or not _has_error(gold, prediction, assignment):
            continue
        candidates.append({
            "corpus": key[0],
            "pmid": key[1],
            "title": gold.get("title", ""),
            "abstract": gold.get("abstract", ""),
            "gold_entities": gold.get("entities", []),
            "gold_relations": gold.get("relations", []),
            "neural": prediction.get("neural", {}),
            "csp": prediction.get("csp", {}),
            "csp_status": prediction.get("csp_status"),
            "review_category": None,
            "allowed_categories": CATEGORIES,
            "review_notes": "",
        })
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:size]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(selected, 1):
            row["error_id"] = f"ERR-{idx:04d}"
            fh.write(json.dumps(row) + "\n")
    return {"eligible_documents": len(candidates), "sampled_documents": len(selected)}


def summarize(review_path: Path) -> dict:
    counts = {category: 0 for category in CATEGORIES}
    unreviewed = invalid = 0
    for row in _read(review_path):
        category = row.get("review_category")
        if category is None:
            unreviewed += 1
        elif category in counts:
            counts[category] += 1
        else:
            invalid += 1
    return {"categories": counts, "unreviewed": unreviewed, "invalid": invalid}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("sample")
    create.add_argument("--predictions", required=True)
    create.add_argument("--gold", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--size", type=int, default=100, choices=range(50, 101))
    create.add_argument("--assignment", choices=("neural", "csp"), default="csp")
    report = sub.add_parser("summarize")
    report.add_argument("--input", required=True)
    args = parser.parse_args()
    if args.command == "sample":
        result = sample_errors(Path(args.predictions), Path(args.gold), Path(args.output),
                               args.size, assignment=args.assignment)
    else:
        result = summarize(Path(args.input))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
