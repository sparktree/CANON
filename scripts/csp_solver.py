"""CANON Phase 3.5 -- Z3-based constraint-satisfaction solver.

Takes a Stage-2 (or Stage-3) trained MultiTaskModel, runs it over a JSONL
split, then for each document solves a constraint-satisfaction problem that
chooses the maximum-confidence assignment of (entity_type, concept,
relation) variables consistent with SNOMED MRCM constraints.

Constraints
-----------
* type-concept compatibility  : the assigned concept must lie under the
  semantic anchor(s) for the entity's NER type. For 'disease' both
  404684003 (Clinical finding) and 64572001 (Disease) are valid anchors;
  for 'chemical' both 105590001 (Substance) and 373873005 (Pharmaceutical
  product) are valid; non-SNOMED types have no constraint.
* relation domain/range       : Tier-1 relations require subject/object
  semantic types compatible with the MRCM domain/range entries from
  outputs/phase1/mrcm_constraints.json. Tier-2 relations with a Semantic
  Network predicate require a valid subject-STY/predicate/object-STY edge.

Encoding
--------
Per document we declare:
    entity_type[i]  in  {0, ..., k_T-1}
    concept[i]      in  {0, ..., k_C-1}
    relation[i,j]   in  {0, ..., k_R-1}
We encode boolean compatibility tables ahead of time (avoids quantifier-
heavy descendant predicates inside Z3).

CLI
---
    python scripts/csp_solver.py [--smoke-test] [--split dev|test]
        [--max-docs 50] [--top-k-types 2] [--top-k-concepts 10]
        [--top-k-relations 3] [--timeout-ms 5000]

Outputs land at outputs/phase3/csp_predictions/{split}.jsonl plus summary.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:
    import config
    from canon_dataset import (
        CanonDocDataset,
        BIO_LABEL_TO_ID,
        BIO_ID_TO_LABEL,
        NO_RELATION_ID,
        NUM_RELATION_LABELS,
        RELATION_LABELS,
        SEMANTIC_CLASSES,
        SEMANTIC_CLASS_TO_ID,
        collate_docs,
        load_soft_lookup,
    )
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
    import concept_sty
    import mrcm_validity
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from canon_dataset import (
        CanonDocDataset,
        BIO_LABEL_TO_ID,
        BIO_ID_TO_LABEL,
        NO_RELATION_ID,
        NUM_RELATION_LABELS,
        RELATION_LABELS,
        SEMANTIC_CLASSES,
        SEMANTIC_CLASS_TO_ID,
        collate_docs,
        load_soft_lookup,
    )
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
    import concept_sty
    import mrcm_validity


# MRCM validity logic lives in mrcm_validity so the solver (here) and the Phase
# 4.3 coherence evaluator share one implementation and cannot diverge. These are
# re-exported for back-compat: existing importers (train_stage3) keep working.
TYPE_ANCHORS = mrcm_validity.TYPE_ANCHORS
TIER1_ATTRIBUTE_IDS = mrcm_validity.TIER1_ATTRIBUTE_IDS
ConstraintTables = mrcm_validity.MRCMTables
concept_under_type = mrcm_validity.concept_under_type


def decode_entities_from_bio(bio: Sequence[int]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    start = -1
    current = ""
    for idx, tag_id in enumerate(bio):
        label = BIO_ID_TO_LABEL.get(int(tag_id), "O")
        if label == "O" or label.startswith("B-"):
            if start >= 0:
                spans.append((start, idx, current))
                start, current = -1, ""
            if label.startswith("B-"):
                start, current = idx, label[2:]
        elif label.startswith("I-"):
            sem_class = label[2:]
            if start < 0 or current != sem_class:
                if start >= 0:
                    spans.append((start, idx, current))
                start, current = idx, sem_class
    if start >= 0:
        spans.append((start, len(bio), current))
    return spans


def load_constraint_tables(
    mrcm_path: Path,
    ancestors_path: Path,
    *,
    semantic_network_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> ConstraintTables:
    """Build the shared MRCM + Semantic Network tables once per session.

    Thin wrapper over mrcm_validity.load_tables that supplies the CSP-side tier
    sets and semantic classes. Signature preserved for existing importers
    (train_stage3). The SN edge set (Tier-2 constraint) is loaded from SRSTRE1;
    defaults to config's UMLS NET path when present, so callers get Tier-2
    coverage without extra wiring. The solver and the Phase 4.3 evaluator build
    their tables from this identical implementation.
    """
    if semantic_network_path is None:
        srstre1 = config.UMLS_SEMANTIC_NETWORK_FILES.get("srstre1")
        if srstre1 is not None and Path(srstre1).exists():
            semantic_network_path = srstre1
    return mrcm_validity.load_tables(
        mrcm_path,
        ancestors_path,
        tier1_relations=TIER1_RELATIONS,
        tier2_relations=TIER2_RELATIONS,
        semantic_classes=SEMANTIC_CLASSES,
        semantic_network_path=semantic_network_path,
        logger=logger or logging.getLogger("csp_solver"),
    )


# ---------------------------------------------------------------------------
# Z3 encoding
# ---------------------------------------------------------------------------


def solve_document(
    doc_predictions: Dict,
    tables: ConstraintTables,
    *,
    timeout_ms: int = 5000,
    score_scale: int = 1000,
    escalation_epsilon: float = 0.05,
) -> Dict:
    """Solve one document.

    doc_predictions schema (built by predict_split):
      {
        "pmid": str,
        "entities": [
           {"type_candidates": [{"sem_class": str, "score": float}, ...],
            "concept_candidates": [{"id": str, "score": float}, ...] | None}, ...
        ],
        "pairs": [
           {"i": int, "j": int,
            "rel_candidates": [{"label": str, "score": float}, ...]}, ...
        ]
      }
    """
    try:
        import z3
    except ImportError:
        return {"status": "z3-missing", "assignment": doc_predictions}

    opt = z3.Optimize()
    opt.set("timeout", int(timeout_ms))

    entities = doc_predictions.get("entities", [])
    pairs = doc_predictions.get("pairs", [])

    # Variables.
    type_vars: List[z3.ArithRef] = []
    concept_vars: List[Optional[z3.ArithRef]] = []
    for i, ent in enumerate(entities):
        kT = max(1, len(ent.get("type_candidates", [])))
        tv = z3.Int(f"type_{i}")
        opt.add(tv >= 0, tv < kT)
        type_vars.append(tv)
        c_cands = ent.get("concept_candidates")
        if c_cands:
            cv = z3.Int(f"concept_{i}")
            opt.add(cv >= 0, cv < len(c_cands))
            concept_vars.append(cv)
        else:
            concept_vars.append(None)

    rel_vars: List[Tuple[int, int, z3.ArithRef, List[Dict]]] = []
    for pair in pairs:
        i = pair["i"]
        j = pair["j"]
        cands = pair.get("rel_candidates", [])
        if not cands:
            continue
        rv = z3.Int(f"rel_{i}_{j}")
        opt.add(rv >= 0, rv < len(cands))
        rel_vars.append((i, j, rv, cands))

    # Type-concept compatibility.
    for i, ent in enumerate(entities):
        c_cands = ent.get("concept_candidates")
        cv = concept_vars[i]
        if not c_cands or cv is None:
            continue
        type_cands = ent.get("type_candidates", [])
        for t_idx, type_cand in enumerate(type_cands):
            sem_class = type_cand["sem_class"]
            allowed_idx = [
                k for k, c in enumerate(c_cands)
                if concept_under_type(c["id"], sem_class, tables)
            ]
            if not allowed_idx:
                # Force this type choice to be infeasible.
                opt.add(z3.Implies(type_vars[i] == t_idx, z3.BoolVal(False)))
            else:
                opt.add(
                    z3.Implies(
                        type_vars[i] == t_idx,
                        z3.Or([cv == k for k in allowed_idx]),
                    )
                )

    # Relation domain/range compatibility (Tier-1 only).
    for (i, j, rv, cands) in rel_vars:
        type_cands_i = entities[i].get("type_candidates", [])
        type_cands_j = entities[j].get("type_candidates", [])
        for r_idx, rcand in enumerate(cands):
            rel_label = rcand["label"]
            if rel_label not in TIER1_RELATIONS:
                continue
            for ti_idx, ti in enumerate(type_cands_i):
                ta = ti["sem_class"]
                for tj_idx, tj in enumerate(type_cands_j):
                    tb = tj["sem_class"]
                    if not tables.valid_pair_for_relation.get((rel_label, ta, tb), False):
                        opt.add(
                            z3.Implies(
                                z3.And(
                                    rv == r_idx,
                                    type_vars[i] == ti_idx,
                                    type_vars[j] == tj_idx,
                                ),
                                z3.BoolVal(False),
                            )
                        )

    # Tier-2 UMLS Semantic Network compatibility at candidate-concept STY
    # granularity. Missing STYs are invalid for constrained predicates, which
    # leaves the always-present no-relation candidate as the honest abstention.
    def _sty_options(ent_idx: int):
        ent = entities[ent_idx]
        cands = ent.get("concept_candidates") or []
        if cands:
            return [(idx, list(c.get("stys", []))) for idx, c in enumerate(cands)]
        return [(None, list(ent.get("stys", [])))]

    for (i, j, rv, cands) in rel_vars:
        for r_idx, rcand in enumerate(cands):
            rel_label = rcand["label"]
            if rel_label not in TIER2_RELATIONS or not tables.tier2_to_sn.get(rel_label):
                continue
            for ci, stys_i in _sty_options(i):
                for cj, stys_j in _sty_options(j):
                    if mrcm_validity.relation_pair_valid_sn(rel_label, stys_i, stys_j, tables):
                        continue
                    terms = [rv == r_idx]
                    if ci is not None and concept_vars[i] is not None:
                        terms.append(concept_vars[i] == ci)
                    if cj is not None and concept_vars[j] is not None:
                        terms.append(concept_vars[j] == cj)
                    opt.add(z3.Implies(z3.And(*terms), z3.BoolVal(False)))

    # Objective: maximize sum of integer-scaled scores.
    score_terms = []
    for i, ent in enumerate(entities):
        type_cands = ent.get("type_candidates", [])
        for k, cand in enumerate(type_cands):
            score_terms.append(z3.If(type_vars[i] == k, int(round(cand["score"] * score_scale)), 0))
        cv = concept_vars[i]
        c_cands = ent.get("concept_candidates")
        if cv is not None and c_cands:
            for k, cand in enumerate(c_cands):
                score_terms.append(z3.If(cv == k, int(round(cand["score"] * score_scale)), 0))
    for (i, j, rv, cands) in rel_vars:
        for k, cand in enumerate(cands):
            bonus = escalation_epsilon if cand.get("label") in TIER1_RELATIONS else 0.0
            score_terms.append(z3.If(
                rv == k, int(round((cand["score"] + bonus) * score_scale)), 0))

    if score_terms:
        opt.maximize(z3.Sum(score_terms))

    t0 = time.time()
    result = opt.check()
    elapsed = time.time() - t0

    if result != z3.sat:
        return {
            "status": "fallback",
            "elapsed_ms": round(elapsed * 1000, 2),
            "assignment": _neural_argmax(entities, pairs),
        }

    model = opt.model()
    out_entities = []
    for i, ent in enumerate(entities):
        t_idx = model[type_vars[i]].as_long() if type_vars[i] is not None else 0
        type_cands = ent.get("type_candidates", []) or [{"sem_class": "none", "score": 0.0}]
        chosen_type = type_cands[t_idx]["sem_class"] if t_idx < len(type_cands) else type_cands[0]["sem_class"]
        cv = concept_vars[i]
        chosen_concept = None
        c_idx = -1
        c_cands = ent.get("concept_candidates")
        if cv is not None and c_cands:
            c_idx = model[cv].as_long()
            chosen_concept = c_cands[c_idx]["id"] if c_idx < len(c_cands) else None
        chosen_cand = c_cands[c_idx] if cv is not None and c_cands else None
        out_entities.append({
            "type": chosen_type,
            "concept": chosen_concept,
            "stys": list(chosen_cand.get("stys", [])) if chosen_cand else list(ent.get("stys", [])),
            "span_start": ent.get("span_start"), "span_end": ent.get("span_end"),
            "surface": ent.get("surface", ""),
        })

    out_pairs = []
    for (i, j, rv, cands) in rel_vars:
        r_idx = model[rv].as_long()
        rel_label = cands[r_idx]["label"] if r_idx < len(cands) else "no-relation"
        # Record the MRCM orientation via the shared predicate so a canonical
        # (finding, attribute, substance) triple can be read off regardless of
        # the corpus's annotation order. i/j/relation are left in annotation
        # order for existing consumers; canonical_subject/object give the
        # SNOMED-canonical direction (equal to i/j unless orientation flips).
        ti, tj = out_entities[i]["type"], out_entities[j]["type"]
        if rel_label in TIER2_RELATIONS:
            orient = mrcm_validity.relation_pair_orientation_sn(
                rel_label, out_entities[i].get("stys", []), out_entities[j].get("stys", []), tables)
        else:
            orient = mrcm_validity.relation_pair_orientation(rel_label, ti, tj, tables)
        if orient == mrcm_validity.ORIENT_REVERSED:
            canon_subj, canon_obj = j, i
        else:
            canon_subj, canon_obj = i, j
        neural_label = cands[0]["label"] if cands else "no-relation"
        event = "unchanged"
        if rel_label != neural_label:
            if rel_label in TIER1_RELATIONS and neural_label in TIER2_RELATIONS:
                event = "tier1-escalation"
            elif neural_label in TIER1_RELATIONS:
                event = "mrcm-rejection"
            elif neural_label in TIER2_RELATIONS:
                event = "tier2-sn-rejection"
        out_pairs.append({
            "i": i, "j": j, "relation": rel_label,
            "orientation": orient,
            "canonical_subject": canon_subj, "canonical_object": canon_obj,
            "event": event,
        })

    return {
        "status": "sat",
        "elapsed_ms": round(elapsed * 1000, 2),
        "assignment": {"entities": out_entities, "pairs": out_pairs},
    }


def _neural_argmax(entities: List[Dict], pairs: List[Dict]) -> Dict:
    out_entities = []
    for ent in entities:
        tcs = ent.get("type_candidates", [])
        ccs = ent.get("concept_candidates")
        chosen_type = tcs[0]["sem_class"] if tcs else "none"
        chosen_concept = ccs[0]["id"] if ccs else None
        chosen_cand = ccs[0] if ccs else None
        out_entities.append({
            "type": chosen_type, "concept": chosen_concept,
            "stys": list(chosen_cand.get("stys", [])) if chosen_cand else list(ent.get("stys", [])),
            "span_start": ent.get("span_start"), "span_end": ent.get("span_end"),
            "surface": ent.get("surface", ""),
        })
    out_pairs = []
    for pair in pairs:
        cands = pair.get("rel_candidates", [])
        chosen = cands[0]["label"] if cands else "no-relation"
        out_pairs.append({"i": pair["i"], "j": pair["j"], "relation": chosen})
    return {"entities": out_entities, "pairs": out_pairs}


# ---------------------------------------------------------------------------
# Inference: run the model, emit per-doc neural predictions, hand to solver.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def model_predict(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    *,
    top_k_types: int,
    top_k_concepts: int,
    top_k_relations: int,
    sty_lookup: Optional[Dict[str, List[str]]] = None,
    oracle_inputs: bool = False,
) -> List[Dict]:
    """Produce neural candidates; end-to-end CRF spans are the default input."""
    model.eval()
    docs: List[Dict] = []
    cid_lookup = model.norm_head.concept_ids if model.has_norm else []
    for batch in loader:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        ner_out = model(batch, active_heads=("ner",))
        decoded = ner_out["raw"]["ner"]["decoded"]
        emissions = torch.softmax(ner_out["raw"]["ner"]["logits"], dim=-1)

        predicted_spans: List[List[Tuple[int, int, str]]] = []
        type_candidates_by_doc: List[List[List[Dict]]] = []
        for b, sequence in enumerate(decoded):
            spans = batch["entity_token_spans"][b] if oracle_inputs else decode_entities_from_bio(sequence)
            cleaned = []
            per_types = []
            for start, end, sem_class in spans:
                valid_tokens = [t for t in range(start, end)
                                if t < batch["offset_mapping"].size(1)
                                and tuple(batch["offset_mapping"][b, t].tolist()) != (0, 0)]
                if not valid_tokens:
                    continue
                start, end = min(valid_tokens), max(valid_tokens) + 1
                scored = []
                for cls in SEMANTIC_CLASSES:
                    b_id = BIO_LABEL_TO_ID[f"B-{cls}"]
                    i_id = BIO_LABEL_TO_ID[f"I-{cls}"]
                    score = emissions[b, start:end, [b_id, i_id]].max(dim=-1).values.mean().item()
                    scored.append({"sem_class": cls, "score": float(score)})
                scored.sort(key=lambda row: row["score"], reverse=True)
                if oracle_inputs:
                    scored.sort(key=lambda row: row["sem_class"] != sem_class)
                chosen_class = scored[0]["sem_class"] if scored else sem_class
                cleaned.append((start, end, chosen_class))
                per_types.append(scored[:max(1, top_k_types)])
            predicted_spans.append(cleaned)
            type_candidates_by_doc.append(per_types)

        inference_batch = dict(batch)
        inference_batch["entity_token_spans"] = predicted_spans
        pair_indices = []
        pair_classes = []
        for spans in predicted_spans:
            pairs = [(i, j) for i in range(len(spans)) for j in range(len(spans)) if i != j]
            pair_indices.append(torch.tensor(pairs, dtype=torch.long) if pairs else torch.zeros((0, 2), dtype=torch.long))
            classes = [[SEMANTIC_CLASS_TO_ID.get(spans[i][2], SEMANTIC_CLASS_TO_ID["none"]),
                        SEMANTIC_CLASS_TO_ID.get(spans[j][2], SEMANTIC_CLASS_TO_ID["none"])]
                       for i, j in pairs]
            pair_classes.append(torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0, 2), dtype=torch.long))
        inference_batch["pair_indices"] = pair_indices
        inference_batch["pair_semantic_classes"] = pair_classes
        for key in ("bio_labels", "norm_targets", "norm_entity_idx", "norm_weights",
                    "pair_labels", "pair_targets", "pair_bag_ids", "pair_weights"):
            inference_batch.pop(key, None)
        out = model(inference_batch, active_heads=("norm", "rel"))

        # Norm logits per surviving in-scope entity (need to align with batch order).
        norm_scores = out["raw"].get("norm", {}).get("scores")
        norm_index = out["raw"].get("norm", {}).get("span_index", [])
        # Build (b, span_idx) -> norm row mapping.
        norm_row_for: Dict[Tuple[int, int], int] = {}
        if norm_scores is not None:
            for r, key in enumerate(norm_index or []):
                norm_row_for[tuple(key)] = r

        rel_logits_per_doc = out["raw"].get("rel", {}).get("per_doc_logits", [])

        B = batch["input_ids"].size(0)
        for b in range(B):
            ent_token_spans = predicted_spans[b]
            raw_doc = batch["raw_docs"][b]
            text_value = raw_doc.get("text") or ((raw_doc.get("title", "") + " " + raw_doc.get("abstract", "")).strip())
            entities_pred = []
            for s_idx, (start, end, sem_class) in enumerate(ent_token_spans):
                type_cands = type_candidates_by_doc[b][s_idx]
                start_char = int(batch["offset_mapping"][b, start, 0].item())
                end_char = int(batch["offset_mapping"][b, end - 1, 1].item())

                concept_cands = None
                if sem_class in TYPE_ANCHORS and norm_scores is not None:
                    row = norm_row_for.get((b, s_idx))
                    if row is not None and row < norm_scores.size(0) and cid_lookup:
                        scores_row = norm_scores[row]
                        topk = min(top_k_concepts, scores_row.size(0))
                        topv, topi = torch.topk(scores_row, topk)
                        topv_n = torch.softmax(topv, dim=-1).cpu().tolist()
                        idx_list = topi.cpu().tolist()
                        concept_cands = [
                            {"id": cid_lookup[idx], "score": float(score),
                             "stys": list((sty_lookup or {}).get(cid_lookup[idx], []))}
                            for idx, score in zip(idx_list, topv_n)
                            if idx < len(cid_lookup)
                        ]
                entities_pred.append({
                    "type_candidates": type_cands,
                    "concept_candidates": concept_cands,
                    "surface": text_value[start_char:end_char],
                    "span_start": start_char,
                    "span_end": end_char,
                    "stys": concept_sty.FALLBACK_STYS.get(sem_class, []),
                })

            pairs_pred = []
            if b < len(rel_logits_per_doc):
                logits = rel_logits_per_doc[b]
                pair_idx = inference_batch["pair_indices"][b]
                if logits.numel():
                    probs = torch.softmax(logits, dim=-1)
                    topk = min(top_k_relations, probs.size(-1))
                    topv, topi = torch.topk(probs, topk, dim=-1)
                    pair_idx_cpu = pair_idx.cpu().tolist()
                    for p in range(probs.size(0)):
                        if p >= len(pair_idx_cpu):
                            break
                        ii, jj = pair_idx_cpu[p]
                        cand_list = []
                        selected = set()
                        for k in range(topk):
                            label = RELATION_LABELS[int(topi[p, k].item())]
                            cand_list.append({"label": label, "score": float(topv[p, k].item())})
                            selected.add(label)
                        required = set(TIER1_RELATIONS) | {"no-relation"}
                        for label in sorted(required - selected):
                            label_idx = RELATION_LABELS.index(label)
                            cand_list.append({"label": label, "score": float(probs[p, label_idx].item())})
                        cand_list.sort(key=lambda c: c["score"], reverse=True)
                        pairs_pred.append({"i": ii, "j": jj, "rel_candidates": cand_list})

            docs.append({
                "pmid": batch["pmids"][b],
                "corpus": batch["corpora"][b],
                "entities": entities_pred,
                "pairs": pairs_pred,
            })
    return docs


def predict_split(
    model: Any,
    split_path: Path,
    output_path: Path,
    tokenizer,
    soft,
    *,
    tables: ConstraintTables,
    device: torch.device,
    top_k_types: int,
    top_k_concepts: int,
    top_k_relations: int,
    timeout_ms: int,
    max_docs: Optional[int],
    max_pairs: int,
    logger: logging.Logger,
    sty_lookup: Optional[Dict[str, List[str]]] = None,
    oracle_inputs: bool = False,
) -> Dict:
    pad_id = tokenizer.pad_token_id or 0
    ds = CanonDocDataset(
        split_path,
        tokenizer,
        soft,
        max_length=512,
        max_docs=max_docs,
        max_pairs=max_pairs,
        seed=0,
    )

    def coll(b):
        return collate_docs(b, pad_token_id=pad_id)

    loader = DataLoader(ds, batch_size=4, collate_fn=coll)
    neural_records = model_predict(
        model, loader, device,
        top_k_types=top_k_types,
        top_k_concepts=top_k_concepts,
        top_k_relations=top_k_relations,
        sty_lookup=sty_lookup,
        oracle_inputs=oracle_inputs,
    )

    n_sat = n_fb = 0
    n_overrides = 0
    event_counts: Dict[str, int] = {}
    elapsed_total = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in neural_records:
            neural_assignment = _neural_argmax(rec["entities"], rec["pairs"])
            sol = solve_document(rec, tables, timeout_ms=timeout_ms)
            if sol["status"] == "sat":
                n_sat += 1
                csp_assignment = sol["assignment"]
                # Count overrides
                for n_e, c_e in zip(neural_assignment["entities"], csp_assignment["entities"]):
                    if n_e.get("concept") != c_e.get("concept"):
                        n_overrides += 1
                for pair in csp_assignment.get("pairs", []):
                    event = pair.get("event", "unchanged")
                    event_counts[event] = event_counts.get(event, 0) + 1
            else:
                n_fb += 1
                csp_assignment = sol.get("assignment", neural_assignment)
            elapsed_total += sol.get("elapsed_ms", 0.0)
            row = {
                "pmid": rec["pmid"],
                "corpus": rec["corpus"],
                "neural": neural_assignment,
                "csp": csp_assignment,
                "csp_status": sol["status"],
                "elapsed_ms": sol.get("elapsed_ms", 0.0),
            }
            fh.write(json.dumps(row) + "\n")

    n_total = n_sat + n_fb
    summary = {
        "split": split_path.name,
        "documents": n_total,
        "csp_sat": n_sat,
        "csp_fallback": n_fb,
        "concept_overrides": n_overrides,
        "fallback_rate": n_fb / n_total if n_total else 0.0,
        "override_rate_per_doc": n_overrides / n_total if n_total else 0.0,
        "avg_solve_ms": elapsed_total / n_total if n_total else 0.0,
        "relation_events": event_counts,
    }
    summary_path = output_path.parent / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"summary: {summary}")
    return summary


def setup_logging(log_path: Path, name: str = "csp_solver") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [{name}] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    from transformers import AutoTokenizer
    from heads import MultiTaskModel
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--top-k-types", type=int, default=2)
    parser.add_argument("--top-k-concepts", type=int, default=10)
    parser.add_argument("--top-k-relations", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--model-dir", default=str(config.STAGE2_DIR / "best"))
    parser.add_argument("--independent-stage1-dir", default=None,
                        help="Compose separately fine-tuned NER/norm/relation Stage-1 models.")
    parser.add_argument("--concept-index-dir", default=str(config.CONCEPT_INDEX_DIR))
    parser.add_argument("--oracle-inputs", action="store_true",
                        help="Diagnostic upper bound: use gold spans/types instead of CRF predictions.")
    parser.add_argument("--output-dir", default=str(config.CSP_PREDICTIONS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "log.txt", name="3.5")

    if args.smoke_test:
        if args.max_docs is None:
            args.max_docs = 50
        args.top_k_concepts = min(args.top_k_concepts, 5)
        args.top_k_types = min(args.top_k_types, 2)
        args.timeout_ms = min(args.timeout_ms, 1000)
        args.split = "dev"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} smoke={args.smoke_test} split={args.split} max_docs={args.max_docs}")

    tables = load_constraint_tables(
        config.MRCM_CONSTRAINTS_JSON,
        config.SNOMED_ANCESTORS_PKL,
        logger=logger,
    )

    concept_index_dir = Path(args.concept_index_dir)
    concept_ids_path = concept_index_dir / "concept_ids.json"
    concept_emb_path = concept_index_dir / "concept_emb.safetensors"
    with concept_ids_path.open() as fh:
        num_concepts = len(json.load(fh))

    def load_head_model(head: str, encoder_dir: Path):
        if not (encoder_dir / "config.json").is_file():
            raise FileNotFoundError(f"trained model is required; missing {encoder_dir / 'config.json'}")
        flags = {name: name == head for name in ("ner", "norm", "rel")}
        loaded = MultiTaskModel(str(encoder_dir), num_concepts=num_concepts, **flags)
        if head == "norm":
            loaded.norm_head.load_concept_index(concept_ids_path, concept_emb_path)
        head_state = encoder_dir / "head_state.pt"
        if not head_state.exists():
            raise FileNotFoundError(f"trained head state is required; missing {head_state}")
        state = torch.load(head_state, map_location="cpu", weights_only=True)
        own = {name: parameter for name, parameter in loaded.named_parameters()
               if not name.startswith("encoder.")}
        copied = 0
        for name, value in state.items():
            if name in own and own[name].shape == value.shape:
                own[name].data.copy_(value)
                copied += 1
        if copied == 0:
            raise RuntimeError(f"no compatible {head} parameters in {head_state}")
        logger.info(f"loaded independent {head} head ({copied} tensors) from {encoder_dir}")
        return loaded

    if args.independent_stage1_dir:
        stage1_dir = Path(args.independent_stage1_dir)
        models = {head: load_head_model(head, stage1_dir / head / "best")
                  for head in ("ner", "norm", "rel")}

        class IndependentHeads:
            has_norm = True

            def __init__(self, head_models):
                self.models = head_models
                self.norm_head = head_models["norm"].norm_head

            def to(self, target):
                for item in self.models.values():
                    item.to(target)
                return self

            def eval(self):
                for item in self.models.values():
                    item.eval()
                return self

            def __call__(self, batch, *, active_heads=None):
                requested = active_heads or ("ner", "norm", "rel")
                merged = {"losses": {}, "raw": {}}
                for head in requested:
                    result = self.models[head](batch, active_heads=(head,))
                    merged["raw"].update(result["raw"])
                    merged["losses"].update(result["losses"])
                return merged

        model = IndependentHeads(models)
        tokenizer = AutoTokenizer.from_pretrained(str(stage1_dir / "ner" / "best"))
    else:
        encoder_dir = Path(args.model_dir)
        if not (encoder_dir / "config.json").is_file():
            raise FileNotFoundError(f"trained model is required; missing {encoder_dir / 'config.json'}")
        tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
        model = MultiTaskModel(str(encoder_dir), num_concepts=num_concepts)
        model.norm_head.load_concept_index(concept_ids_path, concept_emb_path)
        head_state = encoder_dir / "head_state.pt"
        if not head_state.exists():
            raise FileNotFoundError(f"trained head state is required; missing {head_state}")
        try:
            state = torch.load(head_state, map_location="cpu", weights_only=True)
            own = {name: parameter for name, parameter in model.named_parameters()
                   if not name.startswith("encoder.")}
            for name, value in state.items():
                if name in own and own[name].shape == value.shape:
                    own[name].data.copy_(value)
            logger.info(f"loaded head_state.pt from {head_state}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to load {head_state}: {exc}") from exc

    soft = load_soft_lookup(config.SOFT_MAPPING_LOOKUP)
    sty_lookup = concept_sty.load_lookup(config.CONCEPT_STY_LOOKUP_JSON)
    if not sty_lookup:
        raise FileNotFoundError(
            f"concept STY lookup is required; run build_concept_index.py: {config.CONCEPT_STY_LOOKUP_JSON}")

    model.to(device)

    split_path = config.PHASE2_SPLITS_DIR / f"{args.split}.jsonl"
    output_path = output_dir / f"{args.split}.jsonl"
    summary = predict_split(
        model, split_path, output_path, tokenizer, soft,
        tables=tables, device=device,
        top_k_types=args.top_k_types,
        top_k_concepts=args.top_k_concepts,
        top_k_relations=args.top_k_relations,
        timeout_ms=args.timeout_ms,
        max_docs=args.max_docs,
        max_pairs=args.max_pairs,
        logger=logger,
        sty_lookup=sty_lookup,
        oracle_inputs=args.oracle_inputs,
    )
    logger.info(f"summary -> {output_dir/'summary.json'}")


if __name__ == "__main__":
    main()
