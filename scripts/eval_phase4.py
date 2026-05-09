"""Phase 4 evaluation harness for CANON shrunk-scope ablation runs.

Modes
-----
preflight
    Pair an encoder snapshot (.safetensors) with a head bundle (.pt) and a
    concept index, run evaluate_all on dev, dump metrics JSON.

coherence_sweep
    Run model inference once on dev, then evaluate three configurations
    (off / hard) by post-processing the same neural top-k outputs through
    different assignment schemes. Reports per-triple validity, per-doc
    full coherence, and concept-flip counts vs gold.

Usage
-----
    python scripts/eval_phase4.py --mode preflight \\
        --encoder-base outputs/phase3/stage2_epoch8/best \\
        --head-state outputs/phase3/stage2_epoch8/best/head_state.pt \\
        --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \\
        --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \\
        --dev-path outputs/phase2/splits/dev.jsonl \\
        --output outputs/phase4/stage2_epoch8_preflight.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    import config
    from utils import choose_torch_device
    from canon_dataset import CanonDocDataset, collate_docs, load_soft_lookup
    from heads import MultiTaskModel
    from train_stage2 import evaluate_all, aggregate_dev_metric
    from csp_solver import (
        ConstraintTables,
        load_constraint_tables,
        model_predict,
        solve_document,
        concept_under_type,
        TYPE_ANCHORS,
        TIER1_ATTRIBUTE_IDS,
    )
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from utils import choose_torch_device
    from canon_dataset import CanonDocDataset, collate_docs, load_soft_lookup
    from heads import MultiTaskModel
    from train_stage2 import evaluate_all, aggregate_dev_metric
    from csp_solver import (
        ConstraintTables,
        load_constraint_tables,
        model_predict,
        solve_document,
        concept_under_type,
        TYPE_ANCHORS,
        TIER1_ATTRIBUTE_IDS,
    )
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS


def setup_logger(output_path: Path) -> logging.Logger:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_phase4")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_path.parent / (output_path.stem + ".log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_ancestors(logger: logging.Logger):
    try:
        with open(config.SNOMED_ANCESTORS_PKL, "rb") as fh:
            blob = pickle.load(fh)
        anc_dict = blob.get("ancestors", {}) if isinstance(blob, dict) else blob
        return {k: set(v) for k, v in anc_dict.items() if not str(k).startswith("descendant:")}
    except Exception as exc:
        logger.warning(f"ancestors load failed ({exc}); ancestor-match metric will be omitted")
        return None


def run_preflight(args, logger: logging.Logger) -> dict:
    model, tokenizer, device, pad_id, num_concepts = _build_model_for_inference(args, logger)
    dev_loader = _load_dev_dataloader(args, tokenizer, pad_id)
    ancestors = load_ancestors(logger)

    t0 = time.time()
    metrics = evaluate_all(model, dev_loader, device, ancestors=ancestors)
    metrics["aggregate"] = aggregate_dev_metric(metrics)
    metrics["elapsed_sec"] = round(time.time() - t0, 2)
    metrics["device"] = str(device)
    metrics["concept_index_size"] = num_concepts
    return metrics


def _load_constraint_tables_local(mrcm_path: Path, ancestors_path: Path,
                                  logger: logging.Logger) -> ConstraintTables:
    """Build CSP tables matching the current mrcm_constraints.json schema.

    Replaces csp_solver.load_constraint_tables which targets an older flat-list
    format; the live JSON puts domain/range entries inside dicts under
    relation_constraints[rel]['domains'] and ['ranges'].
    """
    import copy
    from canon_dataset import SEMANTIC_CLASSES

    tables = ConstraintTables(type_to_anchors=copy.deepcopy(TYPE_ANCHORS))

    with ancestors_path.open("rb") as fh:
        anc_blob = pickle.load(fh)
    anc_dict = anc_blob.get("ancestors", {}) if isinstance(anc_blob, dict) else anc_blob
    for k, v in anc_dict.items():
        key = str(k)
        if key.startswith("descendant:"):
            tables.descendants[key.split(":", 1)[1]] = frozenset(str(x) for x in v)

    with mrcm_path.open("r", encoding="utf-8") as fh:
        mrcm = json.load(fh)
    relation_constraints = mrcm.get("relation_constraints", {})

    for rel, rel_entry in relation_constraints.items():
        if rel not in TIER1_RELATIONS:
            continue
        if not isinstance(rel_entry, dict):
            continue
        domain_anchors: list = []
        range_anchors: list = []
        for d in rel_entry.get("domains", []):
            for cid in d.get("domain_root_concept_ids", []):
                domain_anchors.append(str(cid))
        for r in rel_entry.get("ranges", []):
            for cid in r.get("range_root_concept_ids", []):
                range_anchors.append(str(cid))

        for type_a, anchors_a in TYPE_ANCHORS.items():
            for type_b, anchors_b in TYPE_ANCHORS.items():
                ok = any(a in domain_anchors for a in anchors_a) and \
                     any(a in range_anchors for a in anchors_b)
                tables.valid_pair_for_relation[(rel, type_a, type_b)] = bool(ok)

    for rel in TIER2_RELATIONS:
        for ta in list(SEMANTIC_CLASSES) + ["none"]:
            for tb in list(SEMANTIC_CLASSES) + ["none"]:
                tables.valid_pair_for_relation[(rel, ta, tb)] = True

    logger.info(f"loaded MRCM: {len(tables.descendants)} descendant sets, "
                f"{len(tables.valid_pair_for_relation)} (rel, ta, tb) pairs")
    return tables


def _build_model_for_inference(args, logger: logging.Logger):
    """Shared loader used by both preflight and coherence_sweep."""
    device = choose_torch_device(args.device)
    logger.info(f"device={device}")

    encoder_base = Path(args.encoder_base)
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_base))
    pad_id = tokenizer.pad_token_id or 0

    concept_ids_path = Path(args.concept_index_ids)
    concept_emb_path = Path(args.concept_index_emb)
    with concept_ids_path.open() as fh:
        num_concepts = len(json.load(fh))
    logger.info(f"concept_index n={num_concepts} from {concept_ids_path.name}")

    model = MultiTaskModel(str(encoder_base), num_concepts=num_concepts)

    if args.encoder_weights:
        logger.info(f"overriding encoder weights from {args.encoder_weights}")
        enc_sd = load_file(args.encoder_weights)
        missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
        logger.info(f"encoder load: missing={len(missing)} unexpected={len(unexpected)}")

    if args.head_state:
        head_files = args.head_state if isinstance(args.head_state, list) else [args.head_state]
        own = {n: p for n, p in model.named_parameters() if not n.startswith("encoder.")}
        for path in head_files:
            logger.info(f"loading head state from {path}")
            head_sd = torch.load(path, map_location="cpu", weights_only=True)
            loaded = 0
            for k, v in head_sd.items():
                if k in own and own[k].shape == v.shape:
                    own[k].data.copy_(v)
                    loaded += 1
            logger.info(f"  copied {loaded}/{len(head_sd)} params from {Path(path).name}")

    model.norm_head.load_concept_index(concept_ids_path, concept_emb_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device, pad_id, num_concepts


def _load_dev_dataloader(args, tokenizer, pad_id):
    soft = load_soft_lookup(config.SOFT_MAPPING_LOOKUP)
    dev_ds = CanonDocDataset(
        Path(args.dev_path), tokenizer, soft,
        max_length=args.max_length,
        max_docs=args.max_dev_docs,
        neg_ratio=args.neg_ratio,
        max_pairs=args.max_pairs,
        seed=43,
    )
    return DataLoader(
        dev_ds, batch_size=args.batch_size,
        collate_fn=lambda b: collate_docs(b, pad_token_id=pad_id),
    )


def _load_dev_gold_index(dev_path: Path) -> dict:
    """pmid -> {entities: [{snomed_id, sem_class, span_start, span_end, surface}], relations: [...]}"""
    index = {}
    with dev_path.open() as fh:
        for line in fh:
            doc = json.loads(line)
            pmid = doc.get("pmid")
            if pmid is None:
                continue
            ents = []
            for e in doc.get("entities", []):
                ents.append({
                    "snomed_id": e.get("mapped_snomed_id"),
                    "sem_class": e.get("semantic_class"),
                    "span_start": e.get("span_start"),
                    "span_end": e.get("span_end"),
                    "surface": e.get("surface_text"),
                    "non_snomed": e.get("non_snomed", False),
                })
            rels = [{"i": r.get("subject_idx"), "j": r.get("object_idx"),
                     "relation": r.get("target_relation"), "tier": r.get("tier")}
                    for r in doc.get("relations", [])]
            index[str(pmid)] = {"entities": ents, "relations": rels}
    return index


def _gold_concept_for_predicted_entity(pred_ent, gold_ents):
    """Match a predicted entity to its gold counterpart by exact span overlap on (start, end)."""
    s, e = pred_ent.get("span_start"), pred_ent.get("span_end")
    if s is None or e is None:
        return None
    for ge in gold_ents:
        if ge.get("span_start") == s and ge.get("span_end") == e:
            return ge.get("snomed_id")
    return None


def _triple_is_valid(sub_type, sub_concept, rel_label, obj_type, obj_concept,
                     tables: ConstraintTables) -> bool:
    """A predicted triple is MRCM-valid iff:
       (1) sub_concept is under sub_type's anchor (or sub is non-SNOMED),
       (2) obj_concept is under obj_type's anchor (or obj is non-SNOMED),
       (3) for tier-1 relations: (rel, sub_type, obj_type) is in valid_pair_for_relation;
           tier-2 relations are unconstrained.
    """
    if sub_concept is not None and not concept_under_type(sub_concept, sub_type, tables):
        return False
    if obj_concept is not None and not concept_under_type(obj_concept, obj_type, tables):
        return False
    if rel_label in TIER1_RELATIONS:
        return tables.valid_pair_for_relation.get((rel_label, sub_type, obj_type), False)
    return True


def _evaluate_assignment(neural_records, mode: str, tables: ConstraintTables,
                         gold_index: dict, timeout_ms: int, logger: logging.Logger) -> dict:
    """Assign per-doc predictions under one mode {off, hard} and aggregate metrics."""
    n_docs = 0
    n_docs_fully_coherent = 0
    n_triples = 0
    n_valid_triples = 0
    n_tier1_triples = 0
    n_tier1_valid = 0
    n_entities = 0
    n_correct_concepts = 0
    n_csp_status_sat = 0
    n_csp_status_fb = 0
    n_csp_status_skip = 0
    n_overrides = 0
    flips_corr_to_incorr = 0
    flips_incorr_to_corr = 0
    flips_both_wrong = 0
    elapsed_total = 0.0

    for rec in neural_records:
        # Neural argmax baseline (always computed for flip diff vs CSP)
        neural_entities = []
        for ent in rec["entities"]:
            tcs = ent.get("type_candidates", [])
            ccs = ent.get("concept_candidates")
            n_t = tcs[0]["sem_class"] if tcs else "none"
            n_c = ccs[0]["id"] if ccs else None
            neural_entities.append({"type": n_t, "concept": n_c})
        neural_pairs = []
        for pair in rec["pairs"]:
            cands = pair.get("rel_candidates", [])
            neural_pairs.append({"i": pair["i"], "j": pair["j"],
                                 "relation": cands[0]["label"] if cands else "no-relation"})

        if mode == "off":
            assignment = {"entities": neural_entities, "pairs": neural_pairs}
            n_csp_status_skip += 1
        elif mode == "hard":
            sol = solve_document(rec, tables, timeout_ms=timeout_ms)
            if sol["status"] == "sat":
                n_csp_status_sat += 1
            else:
                n_csp_status_fb += 1
            assignment = sol.get("assignment", {"entities": neural_entities, "pairs": neural_pairs})
            elapsed_total += sol.get("elapsed_ms", 0.0)
        else:
            raise ValueError(f"unknown mode {mode}")

        # Compute coherence on this assignment.
        ent_assign = assignment.get("entities", [])
        pair_assign = assignment.get("pairs", [])
        doc_all_valid = True
        for pair in pair_assign:
            i, j = pair["i"], pair["j"]
            rel_label = pair.get("relation", "no-relation")
            if rel_label == "no-relation":
                continue
            if i >= len(ent_assign) or j >= len(ent_assign):
                continue
            sub = ent_assign[i]
            obj = ent_assign[j]
            valid = _triple_is_valid(
                sub.get("type"), sub.get("concept"),
                rel_label,
                obj.get("type"), obj.get("concept"),
                tables,
            )
            n_triples += 1
            is_tier1 = rel_label in TIER1_RELATIONS
            if is_tier1:
                n_tier1_triples += 1
                if valid:
                    n_tier1_valid += 1
            if valid:
                n_valid_triples += 1
            else:
                doc_all_valid = False
        if doc_all_valid and n_triples >= 0:
            n_docs_fully_coherent += 1
        n_docs += 1

        # Concept correctness vs gold + flip counts.
        gold_doc = gold_index.get(str(rec.get("pmid")), {})
        gold_ents = gold_doc.get("entities", [])
        for s_idx, ent in enumerate(rec["entities"]):
            if s_idx >= len(ent_assign):
                continue
            gold_cid = _gold_concept_for_predicted_entity(ent, gold_ents)
            if gold_cid is None:
                continue
            n_entities += 1
            csp_concept = ent_assign[s_idx].get("concept")
            neural_concept = neural_entities[s_idx].get("concept")
            if csp_concept == gold_cid:
                n_correct_concepts += 1
            if mode == "hard" and csp_concept != neural_concept:
                n_overrides += 1
                neural_correct = (neural_concept == gold_cid)
                csp_correct = (csp_concept == gold_cid)
                if neural_correct and not csp_correct:
                    flips_corr_to_incorr += 1
                elif csp_correct and not neural_correct:
                    flips_incorr_to_corr += 1
                else:
                    flips_both_wrong += 1

    metrics = {
        "mode": mode,
        "documents": n_docs,
        "predicted_triples": n_triples,
        "valid_triples": n_valid_triples,
        "per_triple_validity": (n_valid_triples / n_triples) if n_triples else None,
        "tier1_triples": n_tier1_triples,
        "tier1_valid": n_tier1_valid,
        "tier1_validity": (n_tier1_valid / n_tier1_triples) if n_tier1_triples else None,
        "fully_coherent_documents": n_docs_fully_coherent,
        "per_document_coherence": (n_docs_fully_coherent / n_docs) if n_docs else None,
        "entities_with_gold": n_entities,
        "concept_correct": n_correct_concepts,
        "concept_top1_acc": (n_correct_concepts / n_entities) if n_entities else None,
        "csp_sat": n_csp_status_sat,
        "csp_fallback": n_csp_status_fb,
        "csp_skipped_off_mode": n_csp_status_skip,
        "avg_solve_ms": (elapsed_total / max(n_docs, 1)) if mode == "hard" else None,
    }
    if mode == "hard":
        metrics.update({
            "n_overrides": n_overrides,
            "flips_correct_to_incorrect": flips_corr_to_incorr,
            "flips_incorrect_to_correct": flips_incorr_to_corr,
            "flips_both_wrong": flips_both_wrong,
            "net_flip_benefit": flips_incorr_to_corr - flips_corr_to_incorr,
        })
    logger.info(f"[{mode}] {metrics}")
    return metrics


def run_coherence_sweep(args, logger: logging.Logger) -> dict:
    model, tokenizer, device, pad_id, num_concepts = _build_model_for_inference(args, logger)
    dev_loader = _load_dev_dataloader(args, tokenizer, pad_id)

    # One-time inference produces neural records consumed by every mode.
    logger.info("running model_predict on dev (one pass)...")
    t0 = time.time()
    neural_records = model_predict(
        model, dev_loader, device,
        top_k_types=args.top_k_types,
        top_k_concepts=args.top_k_concepts,
        top_k_relations=args.top_k_relations,
    )
    inference_sec = time.time() - t0
    logger.info(f"model_predict: {len(neural_records)} docs, {inference_sec:.1f}s")

    tables = _load_constraint_tables_local(
        Path(config.MRCM_CONSTRAINTS_JSON),
        Path(config.SNOMED_ANCESTORS_PKL),
        logger=logger,
    )
    gold_index = _load_dev_gold_index(Path(args.dev_path))
    logger.info(f"loaded gold index for {len(gold_index)} dev pmids")

    sweep = {}
    for mode in args.modes:
        sweep[mode] = _evaluate_assignment(
            neural_records, mode, tables, gold_index,
            timeout_ms=args.timeout_ms, logger=logger,
        )

    return {
        "sweep": sweep,
        "inference_sec": round(inference_sec, 2),
        "device": str(device),
        "concept_index_size": num_concepts,
        "top_k_types": args.top_k_types,
        "top_k_concepts": args.top_k_concepts,
        "top_k_relations": args.top_k_relations,
    }


def run_ootb_norm(args, logger: logging.Logger) -> dict:
    """Zero-shot bi-encoder normalization with an off-the-shelf encoder.

    Pipeline:
      1. load encoder + tokenizer + (rebuilt) concept index
      2. for each gold entity with non-null mapped_snomed_id, encode the
         surface text (mean-pool excluding special tokens), L2-norm
      3. cosine-sim against the candidate concept embedding matrix, argmax
      4. report top-1 accuracy + ancestor-match accuracy
    """
    from transformers import AutoModel
    import torch.nn.functional as F

    device = choose_torch_device(args.device)
    logger.info(f"device={device} encoder={args.encoder_base}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.encoder_base))
    encoder = AutoModel.from_pretrained(str(args.encoder_base)).to(device)
    encoder.eval()

    concept_ids_path = Path(args.concept_index_ids)
    concept_emb_path = Path(args.concept_index_emb)
    with concept_ids_path.open() as fh:
        concept_ids = json.load(fh)
    cid_to_row = {cid: i for i, cid in enumerate(concept_ids)}
    concept_emb = load_file(str(concept_emb_path))["embeddings"].to(device).float()
    concept_emb_n = F.normalize(concept_emb, dim=-1)
    logger.info(f"concept_index n={len(concept_ids)} dim={concept_emb.shape[-1]}")

    ancestors = load_ancestors(logger) or {}

    dev_path = Path(args.dev_path)
    n_total = 0
    n_top1 = 0
    n_ancestor = 0
    n_skipped_oob = 0  # gold concept not in this index

    surface_buffer: list = []
    gold_buffer: list = []

    @torch.inference_mode()
    def flush_batch():
        nonlocal n_total, n_top1, n_ancestor, n_skipped_oob
        if not surface_buffer:
            return
        toks = tokenizer(surface_buffer, padding=True, truncation=True,
                         max_length=64, return_tensors="pt").to(device)
        out = encoder(**toks)
        h = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).float()
        # mean-pool excluding pads (special tokens are kept; matches build_concept_index style)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = F.normalize(pooled, dim=-1)
        sims = pooled @ concept_emb_n.T   # (B, N)
        argmax = sims.argmax(dim=-1).cpu().tolist()
        for pred_row, gold_cid in zip(argmax, gold_buffer):
            pred_cid = concept_ids[pred_row]
            n_total += 1
            if pred_cid == gold_cid:
                n_top1 += 1
                n_ancestor += 1
            else:
                gold_set = ancestors.get(str(gold_cid))
                if gold_set and pred_cid in gold_set:
                    n_ancestor += 1
        surface_buffer.clear()
        gold_buffer.clear()

    BATCH = 64
    docs_seen = 0
    with dev_path.open() as fh:
        for line in fh:
            doc = json.loads(line)
            docs_seen += 1
            if args.max_dev_docs is not None and docs_seen > args.max_dev_docs:
                break
            for ent in doc.get("entities", []):
                gold_cid = ent.get("mapped_snomed_id")
                surface = ent.get("surface_text") or ""
                if not gold_cid or not surface:
                    continue
                if str(gold_cid) not in cid_to_row:
                    n_skipped_oob += 1
                    continue
                surface_buffer.append(surface)
                gold_buffer.append(str(gold_cid))
                if len(surface_buffer) >= BATCH:
                    flush_batch()
    flush_batch()

    metrics = {
        "encoder": str(args.encoder_base),
        "concept_index_size": len(concept_ids),
        "evaluated_entities": n_total,
        "skipped_oob_gold": n_skipped_oob,
        "top1": (n_top1 / n_total) if n_total else None,
        "ancestor": (n_ancestor / n_total) if n_total else None,
    }
    logger.info(f"ootb_norm: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "coherence_sweep", "ootb_norm"], default="preflight")
    parser.add_argument("--encoder-base", required=True,
                        help="HF folder with config.json + tokenizer (provides architecture + tokenizer)")
    parser.add_argument("--encoder-weights", default=None,
                        help="Optional .safetensors of encoder-only weights to override --encoder-base")
    parser.add_argument("--head-state", nargs="+", default=None,
                        help="One or more .pt files with non-encoder head parameters to load. "
                             "Multiple files are loaded in order; later files override earlier "
                             "(but head keys don't overlap across stage1 ner/norm/rel files).")
    parser.add_argument("--concept-index-ids", required=True)
    parser.add_argument("--concept-index-emb", required=True)
    parser.add_argument("--dev-path", default=str(config.PHASE2_SPLITS_DIR / "dev.jsonl"))
    parser.add_argument("--max-dev-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--neg-ratio", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k-types", type=int, default=2,
                        help="coherence_sweep: candidate types per entity")
    parser.add_argument("--top-k-concepts", type=int, default=10,
                        help="coherence_sweep: candidate concepts per entity")
    parser.add_argument("--top-k-relations", type=int, default=3,
                        help="coherence_sweep: candidate relations per pair")
    parser.add_argument("--timeout-ms", type=int, default=5000,
                        help="coherence_sweep: per-doc CSP solve timeout")
    parser.add_argument("--modes", nargs="+", default=["off", "hard"],
                        help="coherence_sweep: which assignment schemes to evaluate")
    args = parser.parse_args()

    output_path = Path(args.output)
    logger = setup_logger(output_path)
    logger.info(f"args: {vars(args)}")

    if args.mode == "preflight":
        metrics = run_preflight(args, logger)
    elif args.mode == "coherence_sweep":
        metrics = run_coherence_sweep(args, logger)
    elif args.mode == "ootb_norm":
        metrics = run_ootb_norm(args, logger)
    else:
        raise ValueError(f"unknown mode {args.mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"metrics -> {output_path}")
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
