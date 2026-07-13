"""CANON Phase 4 task and ontological-coherence evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import config
    import mrcm_validity
    from canon_dataset import _ner_class
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
    from ctd_attestation import attested, load_direct_evidence
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import mrcm_validity
    from canon_dataset import _ner_class
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
    from ctd_attestation import attested, load_direct_evidence


def _load_jsonl(path: Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _prf(pred: Set[tuple], gold: Set[tuple]) -> Dict[str, float]:
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def evaluate(predictions_path: Path, gold_path: Path, output_path: Path,
             assignment_key: str = "csp", *, use_ctd_attestation: bool = True) -> dict:
    predictions = {(r["corpus"], str(r["pmid"])): r for r in _load_jsonl(predictions_path)}
    gold_docs = _load_jsonl(gold_path)
    tables = mrcm_validity.load_tables(
        config.MRCM_CONSTRAINTS_JSON, config.SNOMED_ANCESTORS_PKL,
        tier1_relations=TIER1_RELATIONS, tier2_relations=TIER2_RELATIONS,
        semantic_classes=("clinical_finding", "substance", "pharmaceutical_product",
                          "gene", "variant", "species", "cell_line"),
        semantic_network_path=config.UMLS_SEMANTIC_NETWORK_FILES["srstre1"],
    )
    with config.SNOMED_HIERARCHY_PKL.open("rb") as fh:
        hierarchy_blob = pickle.load(fh)
    hierarchy = hierarchy_blob.get("graph", hierarchy_blob) if isinstance(hierarchy_blob, dict) else hierarchy_blob
    ctd_lookup = (load_direct_evidence(config.CTD_CHEMICALS_DISEASES_FILE,
                                       config.CTD_DIRECT_EVIDENCE_JSON)
                  if use_ctd_attestation and config.CTD_CHEMICALS_DISEASES_FILE.exists() else None)

    ner_pred: Set[tuple] = set()
    ner_gold: Set[tuple] = set()
    relaxed_tp = 0
    norm_exact = norm_parent = norm_semantic = norm_total = 0
    rel_pred: Set[tuple] = set()
    rel_gold: Set[tuple] = set()
    per_rel_pred: Dict[str, Set[tuple]] = defaultdict(set)
    per_rel_gold: Dict[str, Set[tuple]] = defaultdict(set)
    checks = Counter()
    fully_coherent = documents_evaluated = 0
    fallback_docs = 0
    attestation_counts = defaultdict(Counter)
    override_effects = defaultdict(Counter)

    for gold in gold_docs:
        key = (gold.get("corpus", ""), str(gold.get("pmid", "")))
        pred_row = predictions.get(key)
        doc_key = f"{key[0]}:{key[1]}"
        for entity in gold.get("entities", []):
            ner_gold.add((doc_key, entity["span_start"], entity["span_end"], _ner_class(entity)))
        for relation in gold.get("relations", []):
            if not relation.get("target_relation"):
                continue
            s = gold["entities"][relation["subject_idx"]].get("original_code")
            o = gold["entities"][relation["object_idx"]].get("original_code")
            item = (doc_key, s, o, relation["target_relation"])
            rel_gold.add(item)
            per_rel_gold[relation["target_relation"]].add(item)
        if pred_row is None:
            continue
        documents_evaluated += 1
        if pred_row.get("csp_status") != "sat":
            fallback_docs += 1
        assignment = pred_row.get(assignment_key, {})
        pred_entities = assignment.get("entities", [])
        neural_assignment = pred_row.get("neural", {})
        neural_entities = neural_assignment.get("entities", [])
        gold_by_span = {(e["span_start"], e["span_end"]): e for e in gold.get("entities", [])}
        unmatched_gold = set(range(len(gold.get("entities", []))))
        doc_valid = True
        for entity in pred_entities:
            span = (entity.get("span_start"), entity.get("span_end"))
            ptype = entity.get("type")
            ner_pred.add((doc_key, span[0], span[1], ptype))
            for gold_idx in sorted(unmatched_gold):
                g = gold["entities"][gold_idx]
                if _overlaps(span, (g["span_start"], g["span_end"])) and ptype == _ner_class(g):
                    relaxed_tp += 1
                    unmatched_gold.remove(gold_idx)
                    break
            matched = gold_by_span.get(span)
            if matched and matched.get("snomed_active") is True and matched.get("mapped_snomed_id"):
                norm_total += 1
                predicted_cid = str(entity.get("concept"))
                gold_cid = str(matched.get("mapped_snomed_id"))
                is_exact = predicted_cid == gold_cid
                norm_exact += int(is_exact)
                is_parent = is_exact or hierarchy.has_edge(predicted_cid, gold_cid) or hierarchy.has_edge(gold_cid, predicted_cid)
                norm_parent += int(is_parent)
                pred_tops = set(hierarchy.nodes.get(predicted_cid, {}).get("top_level_hierarchies", []))
                gold_tops = set(hierarchy.nodes.get(gold_cid, {}).get("top_level_hierarchies", []))
                norm_semantic += int(is_exact or bool(pred_tops & gold_tops))
            if entity.get("concept"):
                valid = mrcm_validity.concept_under_type(str(entity["concept"]), ptype, tables)
                checks["entity_total"] += 1
                checks["entity_valid"] += int(valid)
                doc_valid &= valid

        for pair in assignment.get("pairs", []):
            label = pair.get("relation")
            if label == "no-relation" or pair.get("i", -1) >= len(pred_entities) or pair.get("j", -1) >= len(pred_entities):
                continue
            a, b = pred_entities[pair["i"]], pred_entities[pair["j"]]
            ga = gold_by_span.get((a.get("span_start"), a.get("span_end")))
            gb = gold_by_span.get((b.get("span_start"), b.get("span_end")))
            if ga and gb:
                item = (doc_key, ga.get("original_code"), gb.get("original_code"), label)
                rel_pred.add(item)
                per_rel_pred[label].add(item)
                if assignment_key == "csp" and pair.get("event", "unchanged") != "unchanged":
                    neural_label = "no-relation"
                    for neural_pair in neural_assignment.get("pairs", []):
                        if neural_pair.get("i") == pair.get("i") and neural_pair.get("j") == pair.get("j"):
                            neural_label = neural_pair.get("relation", "no-relation")
                            break
                    csp_correct = item in rel_gold
                    neural_item = (doc_key, ga.get("original_code"), gb.get("original_code"), neural_label)
                    neural_correct = neural_item in rel_gold
                    effect = ("flip_to_correct" if csp_correct and not neural_correct else
                              "flip_to_incorrect" if neural_correct and not csp_correct else
                              "no_correctness_change")
                    override_effects[pair.get("event", "relation-override")][effect] += 1
                if ctd_lookup and {ga.get("semantic_class"), gb.get("semantic_class")} == {"chemical", "disease"}:
                    chemical = ga if ga.get("semantic_class") == "chemical" else gb
                    disease = ga if ga.get("semantic_class") == "disease" else gb
                    if label in {"causes", "causative-agent", "treats"}:
                        bucket = attestation_counts[key[0]]
                        bucket[f"{label}_total"] += 1
                        bucket[f"{label}_attested"] += int(attested(
                            label, str(chemical.get("original_code") or ""),
                            str(disease.get("original_code") or ""), ctd_lookup))
            if label in TIER1_RELATIONS:
                valid = mrcm_validity.relation_pair_valid(label, a.get("type"), b.get("type"), tables)
                checks["tier1_total"] += 1
                checks["tier1_valid"] += int(valid)
                doc_valid &= valid
            elif label in TIER2_RELATIONS and tables.tier2_to_sn.get(label):
                valid = mrcm_validity.relation_pair_valid_sn(
                    label, a.get("stys", []), b.get("stys", []), tables)
                checks["tier2_total"] += 1
                checks["tier2_valid"] += int(valid)
                doc_valid &= valid
            else:
                checks["unconstrained_relations"] += 1
        fully_coherent += int(doc_valid)

        if assignment_key == "csp":
            for idx, entity in enumerate(pred_entities):
                if idx >= len(neural_entities):
                    continue
                neural_entity = neural_entities[idx]
                if (entity.get("concept"), entity.get("type")) == (
                        neural_entity.get("concept"), neural_entity.get("type")):
                    continue
                matched = gold_by_span.get((entity.get("span_start"), entity.get("span_end")))
                if not matched:
                    override_effects["entity-rectification"]["unscored"] += 1
                    continue
                gold_concept = str(matched.get("mapped_snomed_id") or "")
                gold_type = _ner_class(matched)
                csp_correct = (str(entity.get("concept") or "") == gold_concept and
                               entity.get("type") == gold_type)
                neural_correct = (str(neural_entity.get("concept") or "") == gold_concept and
                                  neural_entity.get("type") == gold_type)
                effect = ("flip_to_correct" if csp_correct and not neural_correct else
                          "flip_to_incorrect" if neural_correct and not csp_correct else
                          "no_correctness_change")
                override_effects["entity-rectification"][effect] += 1

    ner = _prf(ner_pred, ner_gold)
    relaxed_precision = relaxed_tp / len(ner_pred) if ner_pred else 0.0
    relaxed_recall = relaxed_tp / len(ner_gold) if ner_gold else 0.0
    relaxed_f1 = (2 * relaxed_precision * relaxed_recall / (relaxed_precision + relaxed_recall)
                  if relaxed_precision + relaxed_recall else 0.0)
    relation = _prf(rel_pred, rel_gold)
    per_relation = {label: _prf(per_rel_pred[label], per_rel_gold[label])
                    for label in sorted(set(per_rel_pred) | set(per_rel_gold))}
    macro_f1 = sum(row["f1"] for row in per_relation.values()) / len(per_relation) if per_relation else 0.0
    tier1_labels = [label for label in per_relation if label in TIER1_RELATIONS]
    tier2_labels = [label for label in per_relation if label in TIER2_RELATIONS]

    def rate(name: str) -> Optional[float]:
        total = checks[f"{name}_total"]
        return checks[f"{name}_valid"] / total if total else None

    constrained_total = checks["entity_total"] + checks["tier1_total"] + checks["tier2_total"]
    constrained_valid = checks["entity_valid"] + checks["tier1_valid"] + checks["tier2_valid"]
    summary = {
        "assignment": assignment_key,
        "documents_evaluated": documents_evaluated,
        "fallback_documents": fallback_docs,
        "ner_strict": ner,
        "ner_relaxed": {"precision": relaxed_precision, "recall": relaxed_recall, "f1": relaxed_f1},
        "normalization": {"exact_accuracy": norm_exact / norm_total if norm_total else None,
                          "parent_accuracy": norm_parent / norm_total if norm_total else None,
                          "semantic_accuracy": norm_semantic / norm_total if norm_total else None,
                          "evaluated": norm_total},
        "relation": {**relation, "macro_f1": macro_f1,
                     "tier1_macro_f1": (sum(per_relation[x]["f1"] for x in tier1_labels) / len(tier1_labels)
                                         if tier1_labels else None),
                     "tier2_macro_f1": (sum(per_relation[x]["f1"] for x in tier2_labels) / len(tier2_labels)
                                         if tier2_labels else None),
                     "per_relation": per_relation},
        "coherence": {
            "validity_rate_overall": constrained_valid / constrained_total if constrained_total else None,
            "validity_rate_entity_level": rate("entity"),
            "validity_rate_relation_tier1_mrcm": rate("tier1"),
            "validity_rate_relation_tier2_sn": rate("tier2"),
            "fully_coherent_document_rate": fully_coherent / documents_evaluated if documents_evaluated else None,
            "constrained_checks": constrained_total,
            "unconstrained_relations": checks["unconstrained_relations"],
        },
        "ctd_attestation": {
            corpus: {**dict(counts), **{
                f"{label}_rate": (counts[f"{label}_attested"] / counts[f"{label}_total"]
                                   if counts[f"{label}_total"] else None)
                for label in ("causes", "causative-agent", "treats")}}
            for corpus, counts in attestation_counts.items()
        },
        "csp_override_effects": {
            category: dict(counts) for category, counts in sorted(override_effects.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", default=str(config.PHASE2_SPLITS_DIR / "test.jsonl"))
    parser.add_argument("--assignment", choices=("neural", "csp"), default="csp")
    parser.add_argument("--output", default=str(config.OUTPUTS_ROOT / "phase4" / "evaluation.json"))
    parser.add_argument("--skip-ctd-attestation", action="store_true")
    args = parser.parse_args()
    result = evaluate(Path(args.predictions), Path(args.gold), Path(args.output), args.assignment,
                      use_ctd_attestation=not args.skip_ctd_attestation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
