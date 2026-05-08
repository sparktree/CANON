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
  outputs/phase1/mrcm_constraints.json. Tier-2 relations are unconstrained.

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
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    import config
    from canon_dataset import (
        CanonDocDataset,
        NO_RELATION_ID,
        NUM_RELATION_LABELS,
        RELATION_LABELS,
        SEMANTIC_CLASSES,
        collate_docs,
        load_soft_lookup,
    )
    from heads import MultiTaskModel
    from train_stage1 import decode_entities_from_bio
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from canon_dataset import (
        CanonDocDataset,
        NO_RELATION_ID,
        NUM_RELATION_LABELS,
        RELATION_LABELS,
        SEMANTIC_CLASSES,
        collate_docs,
        load_soft_lookup,
    )
    from heads import MultiTaskModel
    from train_stage1 import decode_entities_from_bio
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS


# Type-to-anchor mapping (per plan: both anchors valid for disease + chemical).
TYPE_ANCHORS: Dict[str, List[str]] = {
    "disease":  ["404684003", "64572001"],
    "chemical": ["105590001", "373873005"],
}

# Tier-1 SNOMED attribute SCTIDs (must match mrcm_constraints.json keys / attribute_id field).
TIER1_ATTRIBUTE_IDS: Dict[str, str] = {
    "causative-agent":       "246075003",
    "finding-site":          "363698007",
    "associated-morphology": "116676008",
    "due-to":                "42752001",
    "after":                 "255234002",
}


@dataclass
class ConstraintTables:
    """Precomputed boolean compatibility tables for the CSP solver."""
    type_to_anchors: Dict[str, List[str]] = field(default_factory=dict)
    descendants: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    valid_concept_for_type: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    # (relation_label, type_a, type_b) -> bool. Tier-2 always True.
    valid_pair_for_relation: Dict[Tuple[str, str, str], bool] = field(default_factory=dict)


def load_constraint_tables(
    mrcm_path: Path,
    ancestors_path: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> ConstraintTables:
    """Build the lookup tables once per session."""
    logger = logger or logging.getLogger("csp_solver")
    tables = ConstraintTables(type_to_anchors=copy.deepcopy(TYPE_ANCHORS))

    with Path(ancestors_path).open("rb") as fh:
        anc_blob = pickle.load(fh)
    anc_dict = anc_blob.get("ancestors", {}) if isinstance(anc_blob, dict) else anc_blob
    for k, v in anc_dict.items():
        key = str(k)
        if key.startswith("descendant:"):
            tables.descendants[key.split(":", 1)[1]] = frozenset(str(x) for x in v)

    with Path(mrcm_path).open("r", encoding="utf-8") as fh:
        mrcm = json.load(fh)
    relation_constraints = mrcm.get("relation_constraints", {})

    # Tier-1 pair compatibility: (relation, type_a, type_b) -> bool.
    for rel, entries in relation_constraints.items():
        if rel not in TIER1_RELATIONS:
            continue
        attr_id = TIER1_ATTRIBUTE_IDS.get(rel)
        if attr_id is None:
            continue
        domain_anchors: List[str] = []
        range_anchors: List[str] = []
        for entry in entries:
            d = str(entry.get("domain_root") or entry.get("domain") or "")
            r = str(entry.get("range_root") or entry.get("range") or "")
            if d:
                domain_anchors.append(d)
            if r:
                range_anchors.append(r)
        for type_a, anchors_a in TYPE_ANCHORS.items():
            for type_b, anchors_b in TYPE_ANCHORS.items():
                ok = any(a in domain_anchors for a in anchors_a) and any(
                    a in range_anchors for a in anchors_b
                )
                tables.valid_pair_for_relation[(rel, type_a, type_b)] = bool(ok)
        # type_a or type_b not in TYPE_ANCHORS -> default invalid (CSP solver
        # forces no-relation for those pairs anyway).

    for rel in TIER2_RELATIONS:
        for ta in list(SEMANTIC_CLASSES) + ["none"]:
            for tb in list(SEMANTIC_CLASSES) + ["none"]:
                tables.valid_pair_for_relation[(rel, ta, tb)] = True

    if logger is not None:
        logger.info(
            f"loaded MRCM constraints: {len(tables.descendants)} descendant sets; "
            f"{len(tables.valid_pair_for_relation)} (relation, ta, tb) pairs"
        )
    return tables


def concept_under_type(concept_id: str, sem_class: str, tables: ConstraintTables) -> bool:
    anchors = tables.type_to_anchors.get(sem_class)
    if not anchors:
        # NER-only types (gene, variant, species, cell_line) have no concept constraint.
        return True
    for anc in anchors:
        ds = tables.descendants.get(anc)
        if ds is not None and (concept_id == anc or concept_id in ds):
            return True
    return False


# ---------------------------------------------------------------------------
# Z3 encoding
# ---------------------------------------------------------------------------


def solve_document(
    doc_predictions: Dict,
    tables: ConstraintTables,
    *,
    timeout_ms: int = 5000,
    score_scale: int = 1000,
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
            score_terms.append(z3.If(rv == k, int(round(cand["score"] * score_scale)), 0))

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
        c_cands = ent.get("concept_candidates")
        if cv is not None and c_cands:
            c_idx = model[cv].as_long()
            chosen_concept = c_cands[c_idx]["id"] if c_idx < len(c_cands) else None
        out_entities.append({"type": chosen_type, "concept": chosen_concept})

    out_pairs = []
    for (i, j, rv, cands) in rel_vars:
        r_idx = model[rv].as_long()
        rel_label = cands[r_idx]["label"] if r_idx < len(cands) else "no-relation"
        out_pairs.append({"i": i, "j": j, "relation": rel_label})

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
        out_entities.append({"type": chosen_type, "concept": chosen_concept})
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
    model: MultiTaskModel,
    loader: DataLoader,
    device: torch.device,
    *,
    top_k_types: int,
    top_k_concepts: int,
    top_k_relations: int,
) -> List[Dict]:
    """Produce one neural-prediction record per document."""
    model.eval()
    docs: List[Dict] = []
    cid_lookup = model.norm_head.concept_ids if model.has_norm else []
    for batch in loader:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out = model(batch, active_heads=("ner", "norm", "rel"))

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
            ent_token_spans = batch["entity_token_spans"][b]
            ent_dicts = batch["entity_original"][b]
            entities_pred = []
            for s_idx, (start, end, sem_class) in enumerate(ent_token_spans):
                # Type candidates: the gold semantic_class first (we have it from
                # the dataset survival path) plus 'none' fallback. We do NOT use
                # the CRF decode here because span identification already happened
                # by the dataloader; the CSP only needs a candidate for the type
                # variable scope.
                type_cands = [{"sem_class": sem_class, "score": 1.0}]
                if top_k_types > 1:
                    type_cands.append({"sem_class": "none", "score": 0.05})

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
                            {"id": cid_lookup[idx], "score": float(score)}
                            for idx, score in zip(idx_list, topv_n)
                            if idx < len(cid_lookup)
                        ]
                entities_pred.append({
                    "type_candidates": type_cands,
                    "concept_candidates": concept_cands,
                    "surface": ent_dicts[s_idx].get("surface_text") if s_idx < len(ent_dicts) else "",
                    "span_start": ent_dicts[s_idx].get("span_start") if s_idx < len(ent_dicts) else None,
                    "span_end": ent_dicts[s_idx].get("span_end") if s_idx < len(ent_dicts) else None,
                    "gold_sem_class": sem_class,
                })

            pairs_pred = []
            if b < len(rel_logits_per_doc):
                logits = rel_logits_per_doc[b]
                pair_idx = batch["pair_indices"][b]
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
                        for k in range(topk):
                            label = RELATION_LABELS[int(topi[p, k].item())]
                            cand_list.append({"label": label, "score": float(topv[p, k].item())})
                        pairs_pred.append({"i": ii, "j": jj, "rel_candidates": cand_list})

            docs.append({
                "pmid": batch["pmids"][b],
                "corpus": batch["corpora"][b],
                "entities": entities_pred,
                "pairs": pairs_pred,
            })
    return docs


def predict_split(
    model: MultiTaskModel,
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
    )

    n_sat = n_fb = 0
    n_overrides = 0
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

    encoder_dir = Path(args.model_dir)
    if not (encoder_dir / "config.json").is_file():
        logger.info(f"model dir {encoder_dir} missing; falling back to SapBERT encoder")
        encoder_dir = Path(config.SAPBERT_ENCODER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    soft = load_soft_lookup(config.SOFT_MAPPING_LOOKUP)

    with open(config.CONCEPT_INDEX_IDS) as fh:
        num_concepts = len(json.load(fh))
    model = MultiTaskModel(str(encoder_dir), num_concepts=num_concepts)
    model.norm_head.load_concept_index(config.CONCEPT_INDEX_IDS, config.CONCEPT_INDEX_EMB)
    head_state = encoder_dir / "head_state.pt"
    if head_state.exists():
        try:
            sd = torch.load(head_state, map_location="cpu", weights_only=True)
            own = {n: p for n, p in model.named_parameters() if not n.startswith("encoder.")}
            for k, v in sd.items():
                if k in own and own[k].shape == v.shape:
                    own[k].data.copy_(v)
            logger.info(f"loaded head_state.pt from {head_state}")
        except Exception as exc:  # noqa: BLE001
            logger.info(f"failed to load head_state.pt: {exc}")
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
    )
    logger.info(f"summary -> {output_dir/'summary.json'}")


if __name__ == "__main__":
    main()
