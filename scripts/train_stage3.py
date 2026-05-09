"""CANON Phase 3.6 -- Stage 3 CSP-feedback fine-tuning.

1. Load Stage-2 best.
2. Run CSP solver on the training set, collect overrides where CSP changed
   the concept assignment.
3. Build a new soft-mapping JSON (Stage 2's lookup, boosted toward CSP's
   choice with factor 2.0 and renormalized per MeSH ID).
4. Retrain 5 epochs from Stage-2 weights using the new soft labels (encoder
   lr 5e-6, tau frozen at 0.2).

CLI
---
    python scripts/train_stage3.py [--smoke-test]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    import config
    from utils import choose_torch_device
    from canon_dataset import (
        CanonDocDataset,
        collate_docs,
        load_soft_lookup,
    )
    from heads import MultiTaskModel
    from csp_solver import (
        ConstraintTables,
        load_constraint_tables,
        model_predict,
        solve_document,
        TYPE_ANCHORS,
    )
    from train_stage1 import setup_logging
    from train_stage2 import JointLoss, evaluate_all, aggregate_dev_metric
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from utils import choose_torch_device
    from canon_dataset import (
        CanonDocDataset,
        collate_docs,
        load_soft_lookup,
    )
    from heads import MultiTaskModel
    from csp_solver import (
        ConstraintTables,
        load_constraint_tables,
        model_predict,
        solve_document,
        TYPE_ANCHORS,
    )
    from train_stage1 import setup_logging
    from train_stage2 import JointLoss, evaluate_all, aggregate_dev_metric


DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_ENCODER_LR = 5e-6
DEFAULT_HEAD_LR = 1e-5
DEFAULT_BOOST = 2.0
DEFAULT_TAU = 0.2
DEFAULT_MAX_LENGTH = 512
DEFAULT_MAX_PAIRS = 64

SMOKE_MAX_DOCS = 100
SMOKE_BATCH = 4
SMOKE_EPOCHS = 1
SMOKE_TIMEOUT_MS = 500


def collect_overrides(
    model: MultiTaskModel,
    train_path: Path,
    tokenizer,
    soft,
    tables: ConstraintTables,
    device: torch.device,
    *,
    max_docs: Optional[int],
    timeout_ms: int,
    top_k_concepts: int,
    max_pairs: int,
    logger: logging.Logger,
) -> Dict[str, Dict[str, int]]:
    """Run model + CSP on training docs and tabulate {original_code: {snomed_id: count}}.

    Each entity that survived the dataset's truncation contributes one vote:
    the CSP-chosen concept (which equals the neural top-1 when CSP doesn't
    override). We weight overrides higher when boosting the soft lookup.
    """
    pad_id = tokenizer.pad_token_id or 0
    ds = CanonDocDataset(train_path, tokenizer, soft, max_length=DEFAULT_MAX_LENGTH,
                         max_docs=max_docs, max_pairs=max_pairs, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=lambda b: collate_docs(b, pad_token_id=pad_id))

    neural = model_predict(
        model, loader, device,
        top_k_types=2,
        top_k_concepts=top_k_concepts,
        top_k_relations=3,
    )

    # We need original_code per surviving entity. Re-walk the dataset to
    # extract those alongside the neural records.
    ds_for_codes = CanonDocDataset(train_path, tokenizer, soft, max_length=DEFAULT_MAX_LENGTH,
                                   max_docs=max_docs, max_pairs=max_pairs, seed=0)
    code_records: List[List[Optional[str]]] = []
    for feat in ds_for_codes:
        per_doc_codes = []
        for ent in feat.entity_original:
            sclass = ent.get("semantic_class")
            if sclass in TYPE_ANCHORS:
                code = ent.get("original_code")
                per_doc_codes.append(code if code else None)
            else:
                per_doc_codes.append(None)
        code_records.append(per_doc_codes)

    counts: Dict[str, Dict[str, int]] = {}
    overrides_logged = 0
    for rec, codes in zip(neural, code_records):
        sol = solve_document(rec, tables, timeout_ms=timeout_ms)
        assignment = sol.get("assignment", {})
        ents = assignment.get("entities", [])
        for ent_idx, ent in enumerate(ents):
            if ent_idx >= len(codes):
                break
            code = codes[ent_idx]
            cid = ent.get("concept")
            if not code or not cid:
                continue
            counts.setdefault(code, {})
            counts[code][cid] = counts[code].get(cid, 0) + 1
            # Compare to neural argmax to count overrides.
            neural_cands = rec["entities"][ent_idx].get("concept_candidates") or []
            if neural_cands and neural_cands[0]["id"] != cid:
                overrides_logged += 1

    logger.info(f"collected overrides: {overrides_logged} entities differed; {len(counts)} codes touched")
    return counts


def build_updated_soft_lookup(
    base_soft: Dict[str, List[Dict]],
    counts: Dict[str, Dict[str, int]],
    *,
    boost: float,
    output_path: Path,
    logger: logging.Logger,
) -> Dict[str, List[Dict]]:
    """Boost CSP-preferred concepts in the original soft lookup."""
    updated = copy.deepcopy(base_soft)
    boosted_codes = 0
    for code, cand_counts in counts.items():
        if code not in updated:
            continue
        # Pick the CSP-preferred concept = max-count.
        if not cand_counts:
            continue
        chosen, _ = max(cand_counts.items(), key=lambda kv: kv[1])
        cands = updated[code]
        # Find the matching candidate; if absent, append.
        found = False
        for cand in cands:
            if str(cand.get("snomed_id")) == str(chosen):
                cand["prob"] = float(cand.get("prob", 0.0)) * boost
                found = True
                break
        if not found:
            cands.append({"snomed_id": chosen, "term": "", "prob": 0.5,
                          "hop_dist": 0, "sim_string": 0.0,
                          "sim_ontological": 0.0, "sim_ic": 0.0})
        # Renormalize.
        total = sum(float(c.get("prob", 0.0)) for c in cands) or 1.0
        for cand in cands:
            cand["prob"] = float(cand.get("prob", 0.0)) / total
        updated[code] = cands
        boosted_codes += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(updated, fh)
    logger.info(f"updated soft lookup: boosted {boosted_codes} codes -> {output_path}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--encoder-lr", type=float, default=DEFAULT_ENCODER_LR)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--boost", type=float, default=DEFAULT_BOOST)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--top-k-concepts", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--train-path", default=str(config.PHASE2_SPLITS_DIR / "train.jsonl"))
    parser.add_argument("--dev-path", default=str(config.PHASE2_SPLITS_DIR / "dev.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Optional train-doc cap outside smoke-test mode.")
    parser.add_argument("--max-dev-docs", type=int, default=None,
                        help="Optional dev-doc cap outside smoke-test mode.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--stage2-dir", default=str(config.STAGE2_DIR / "best"))
    parser.add_argument("--output-dir", default=str(config.STAGE3_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "log.txt", "3.6")

    smoke = args.smoke_test
    epochs = SMOKE_EPOCHS if smoke else args.epochs
    batch_size = SMOKE_BATCH if smoke else args.batch_size
    max_docs = SMOKE_MAX_DOCS if smoke else args.max_docs
    timeout_ms = SMOKE_TIMEOUT_MS if smoke else args.timeout_ms

    logger.info(f"epochs={epochs} batch={batch_size} max_docs={max_docs} smoke={smoke}")

    device = choose_torch_device(args.device)
    logger.info(f"device={device}")
    encoder_dir = Path(args.stage2_dir)
    if not (encoder_dir / "config.json").is_file():
        encoder_dir = config.SAPBERT_ENCODER_DIR
        logger.info(f"stage2 dir missing; falling back to {encoder_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    base_soft = load_soft_lookup(config.SOFT_MAPPING_LOOKUP)

    tables = load_constraint_tables(
        config.MRCM_CONSTRAINTS_JSON,
        config.SNOMED_ANCESTORS_PKL,
        logger=logger,
    )

    with open(config.CONCEPT_INDEX_IDS) as fh:
        num_concepts = len(json.load(fh))
    model = MultiTaskModel(str(encoder_dir), num_concepts=num_concepts)
    model.norm_head.load_concept_index(config.CONCEPT_INDEX_IDS, config.CONCEPT_INDEX_EMB)
    head_state = encoder_dir / "head_state.pt"
    if head_state.exists():
        sd = torch.load(head_state, map_location="cpu", weights_only=True)
        own = {n: p for n, p in model.named_parameters() if not n.startswith("encoder.")}
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].data.copy_(v)
        logger.info(f"loaded {head_state}")
    model.to(device)

    # 1+2: collect CSP overrides.
    train_path = Path(args.train_path)
    counts = collect_overrides(
        model, train_path, tokenizer, base_soft, tables, device,
        max_docs=max_docs, timeout_ms=timeout_ms,
        top_k_concepts=args.top_k_concepts, max_pairs=args.max_pairs, logger=logger,
    )

    # 3: write updated soft mapping.
    updated_soft_path = config.STAGE3_SOFT_MAPPING
    updated_soft = build_updated_soft_lookup(
        base_soft, counts, boost=args.boost, output_path=updated_soft_path, logger=logger,
    )

    # 4: retrain.
    pad_id = tokenizer.pad_token_id or 0
    train_ds = CanonDocDataset(train_path, tokenizer, updated_soft,
                               max_length=args.max_length, max_docs=max_docs,
                               max_pairs=args.max_pairs, seed=42)
    dev_ds = CanonDocDataset(Path(args.dev_path), tokenizer, updated_soft,
                             max_length=args.max_length,
                             max_docs=args.max_dev_docs if not smoke else max_docs,
                             max_pairs=args.max_pairs, seed=43)
    coll = lambda b: collate_docs(b, pad_token_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=coll)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=coll)

    if model.has_norm:
        model.norm_head.tau = args.tau
    model.freeze_encoder(False)
    joint = JointLoss(["ner", "norm", "rel"]).to(device)
    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
        {"params": list(joint.parameters()), "lr": args.head_lr},
    ], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    approx = max(1, (max_docs or 40000) // batch_size)
    total_steps = epochs * approx
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    metric_log = output_dir / "dev_metrics.jsonl"
    metric_log.write_text("")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        t0 = time.time()
        for batch in train_loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(batch)
                if not out["losses"]:
                    continue
                loss, _ = joint(out["losses"])
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            steps += 1
        avg = running / max(steps, 1)
        metrics = evaluate_all(model, dev_loader, device)
        agg = aggregate_dev_metric(metrics)
        metrics.update({"epoch": epoch, "train_loss": avg, "aggregate": agg,
                        "elapsed_sec": round(time.time() - t0, 2)})
        with metric_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")
        logger.info(f"epoch {epoch}/{epochs} loss={avg:.4f} agg={agg:.4f} {metrics}")

        if agg > best_score:
            best_score = agg
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True, parents=True)
            model.encoder.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if not n.startswith("encoder.")},
                       str(best_dir / "head_state.pt"))
            with (best_dir / "train_state.json").open("w", encoding="utf-8") as fh:
                json.dump({"epoch": epoch, "aggregate": agg}, fh, indent=2)

    summary = {
        "epochs": epochs,
        "best_aggregate": best_score,
        "smoke_test": bool(smoke),
        "device": str(device),
        "boosted_codes": len(counts),
        "stage3_soft_mapping": str(updated_soft_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"done. best={best_score:.4f}")


if __name__ == "__main__":
    main()
