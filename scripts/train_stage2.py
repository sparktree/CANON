"""CANON Phase 3.4 -- Stage 2 joint multi-task training.

Loads the three Stage-1 best checkpoints (NER / Norm / Rel) and continues
training all heads + encoder jointly with Kendall-style learned uncertainty
weights and a tau anneal on the concept-norm soft-label temperature.

CLI
---
    python scripts/train_stage2.py [--smoke-test]

Outputs at outputs/phase3/stage2/{best, log.txt, training_summary.json,
dev_metrics.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    from train_stage1 import (
        evaluate,
        ner_micro_f1,
        relation_macro_f1,
        decode_entities_from_bio,
        setup_logging,
    )
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
    from train_stage1 import (
        evaluate,
        ner_micro_f1,
        relation_macro_f1,
        decode_entities_from_bio,
        setup_logging,
    )


DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 8
DEFAULT_ENCODER_LR = 1e-5
DEFAULT_HEAD_LR = 3e-5
DEFAULT_WARMUP = 0.05
DEFAULT_MAX_LENGTH = 512
DEFAULT_NEG_RATIO = 2.0
DEFAULT_MAX_PAIRS = 64
DEFAULT_TAU_START = 1.0
DEFAULT_TAU_END = 0.2
DEFAULT_PATIENCE = 3

SMOKE_MAX_DOCS = 100
SMOKE_BATCH = 4
SMOKE_EPOCHS = 2


class JointLoss(nn.Module):
    """Kendall et al. 2018 multi-task uncertainty weighting.

    L = sum_i (1/(2 sigma_i^2)) * L_i + log sigma_i
    Parameterized via log_sigma to keep sigma > 0.
    """

    def __init__(self, task_names: List[str]) -> None:
        super().__init__()
        self.task_names = task_names
        for t in task_names:
            self.register_parameter(f"log_sigma_{t}", nn.Parameter(torch.zeros(())))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = None
        components: Dict[str, float] = {}
        for t, l in losses.items():
            log_sigma = getattr(self, f"log_sigma_{t}")
            term = 0.5 * torch.exp(-2.0 * log_sigma) * l + log_sigma
            total = term if total is None else total + term
            components[t] = float(l.detach())
            components[f"log_sigma_{t}"] = float(log_sigma.detach())
        if total is None:
            total = torch.tensor(0.0, requires_grad=True)
        return total, components


def tau_for_epoch(epoch: int, total: int, t_start: float, t_end: float) -> float:
    if total <= 1:
        return t_end
    frac = (epoch - 1) / (total - 1)
    return t_start + frac * (t_end - t_start)


def aggregate_dev_metric(metrics: Dict[str, float]) -> float:
    return (
        0.4 * metrics.get("ner_f1", 0.0)
        + 0.3 * metrics.get("norm_top1", 0.0)
        + 0.3 * metrics.get("rel_macro_f1", 0.0)
    )


def evaluate_all(model: MultiTaskModel, loader: DataLoader, device: torch.device,
                 ancestors: Optional[Dict[str, set]] = None) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    # NER
    preds_ner, golds_ner = [], []
    # Norm
    norm_exact = norm_total = 0
    norm_anc = 0
    # Rel
    preds_rel, golds_rel = [], []
    cid_lookup = model.norm_head.concept_ids if model.has_norm else []
    with torch.inference_mode():
        for batch in loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(batch, active_heads=("ner", "norm", "rel"))

            if "ner" in out["raw"]:
                decoded = out["raw"]["ner"]["decoded"]
                mask = batch["attention_mask"].cpu()
                gold_bio = batch["bio_labels"].cpu()
                for b, seq in enumerate(decoded):
                    L = int(mask[b].sum())
                    preds_ner.append(decode_entities_from_bio(seq[:L]))
                    golds_ner.append(decode_entities_from_bio(gold_bio[b][:L].tolist()))

            if "norm" in out["raw"]:
                scores = out["raw"]["norm"].get("scores")
                if scores is not None and scores.numel() > 0:
                    argmax = scores.argmax(dim=-1).cpu().tolist()
                    row = 0
                    for b in range(len(batch["norm_targets"])):
                        ent_idx_list = batch["norm_entity_idx"][b]
                        target_list = batch["norm_targets"][b]
                        for k, _ in enumerate(ent_idx_list):
                            if k >= len(target_list):
                                continue
                            if row >= scores.size(0):
                                break
                            pred_cid = cid_lookup[argmax[row]] if argmax[row] < len(cid_lookup) else None
                            tp = target_list[k]
                            if not tp:
                                row += 1
                                continue
                            gold_cid = max(tp.items(), key=lambda kv: kv[1])[0]
                            norm_total += 1
                            if pred_cid == gold_cid:
                                norm_exact += 1
                                norm_anc += 1
                            elif ancestors and pred_cid is not None:
                                gold_set = ancestors.get(str(gold_cid), set())
                                if pred_cid in gold_set or pred_cid == gold_cid:
                                    norm_anc += 1
                            row += 1

            if "rel" in out["raw"]:
                for b, logits in enumerate(out["raw"]["rel"]["per_doc_logits"]):
                    if logits.numel() == 0:
                        continue
                    pred = logits.argmax(dim=-1).cpu().tolist()
                    gold = batch["pair_labels"][b].cpu().tolist()
                    preds_rel.extend(pred)
                    golds_rel.extend(gold)

    if preds_ner:
        p, r, f = ner_micro_f1(preds_ner, golds_ner)
        metrics.update({"ner_precision": p, "ner_recall": r, "ner_f1": f})
    if norm_total:
        metrics.update({"norm_top1": norm_exact / norm_total,
                        "norm_ancestor": norm_anc / norm_total,
                        "norm_evaluated": float(norm_total)})
    if preds_rel:
        metrics.update({"rel_macro_f1": relation_macro_f1(preds_rel, golds_rel),
                        "rel_pairs": float(len(preds_rel))})
    return metrics


def load_stage1_state(model: MultiTaskModel, stage1_dir: Path, logger: logging.Logger) -> None:
    """Optionally warm-start each head from its Stage-1 checkpoint.

    The state file is `head_state.pt` written by train_stage1; missing keys
    are tolerated (Stage 2 falls back to random init for those heads).
    """
    for head_name in ("ner", "norm", "rel"):
        ck = stage1_dir / head_name / "best" / "head_state.pt"
        if not ck.exists():
            logger.info(f"stage1 ckpt missing for head={head_name}; skipping warm start")
            continue
        try:
            sd = torch.load(ck, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            logger.info(f"failed to load {ck}: {exc}")
            continue
        own = {n: p for n, p in model.named_parameters() if not n.startswith("encoder.")}
        loaded = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].data.copy_(v)
                loaded += 1
        logger.info(f"warm-started {loaded} params from {ck}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--encoder-lr", type=float, default=DEFAULT_ENCODER_LR)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--neg-ratio", type=float, default=DEFAULT_NEG_RATIO)
    parser.add_argument("--tau-start", type=float, default=DEFAULT_TAU_START)
    parser.add_argument("--tau-end", type=float, default=DEFAULT_TAU_END)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--train-path", default=str(config.PHASE2_SPLITS_DIR / "train.jsonl"))
    parser.add_argument("--dev-path", default=str(config.PHASE2_SPLITS_DIR / "dev.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Optional train-doc cap outside smoke-test mode.")
    parser.add_argument("--max-dev-docs", type=int, default=None,
                        help="Optional dev-doc cap outside smoke-test mode.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--encoder-dir", default=str(config.SAPBERT_ENCODER_DIR))
    parser.add_argument("--concept-index-ids", default=str(config.CONCEPT_INDEX_IDS))
    parser.add_argument("--concept-index-emb", default=str(config.CONCEPT_INDEX_EMB))
    parser.add_argument("--stage1-dir", default=str(config.STAGE1_DIR))
    parser.add_argument("--output-dir", default=str(config.STAGE2_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "log.txt", "3.4")

    smoke = args.smoke_test
    epochs = SMOKE_EPOCHS if smoke else args.epochs
    batch_size = SMOKE_BATCH if smoke else args.batch_size
    max_docs = SMOKE_MAX_DOCS if smoke else args.max_docs

    logger.info(f"epochs={epochs} batch={batch_size} max_docs={max_docs} smoke={smoke}")

    device = choose_torch_device(args.device)
    logger.info(f"device={device}")
    encoder_dir = Path(args.encoder_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    pad_id = tokenizer.pad_token_id or 0
    soft = load_soft_lookup(config.SOFT_MAPPING_LOOKUP)

    train_ds = CanonDocDataset(
        Path(args.train_path),
        tokenizer, soft, max_length=args.max_length,
        max_docs=max_docs, neg_ratio=args.neg_ratio, max_pairs=args.max_pairs, seed=42)
    dev_ds = CanonDocDataset(
        Path(args.dev_path),
        tokenizer, soft, max_length=args.max_length,
        max_docs=args.max_dev_docs if not smoke else max_docs,
        neg_ratio=args.neg_ratio, max_pairs=args.max_pairs, seed=43)

    def coll(b):
        return collate_docs(b, pad_token_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=coll)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=coll)

    concept_ids_path = Path(args.concept_index_ids)
    concept_emb_path = Path(args.concept_index_emb)
    with open(concept_ids_path) as fh:
        num_concepts = len(json.load(fh))
    model = MultiTaskModel(str(encoder_dir), num_concepts=num_concepts)
    model.norm_head.load_concept_index(concept_ids_path, concept_emb_path)
    load_stage1_state(model, Path(args.stage1_dir), logger)
    model.to(device)
    model.freeze_encoder(False)

    joint_loss = JointLoss(["ner", "norm", "rel"]).to(device)

    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.head_lr},
            {"params": list(joint_loss.parameters()), "lr": args.head_lr},
        ],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )
    approx_steps = max(1, (max_docs or 40000) // batch_size)
    total_steps = epochs * approx_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    # Ancestors for ancestor-match metric
    ancestors = None
    if not smoke:
        try:
            import pickle
            with open(config.SNOMED_ANCESTORS_PKL, "rb") as fh:
                blob = pickle.load(fh)
            anc_dict = blob.get("ancestors", {}) if isinstance(blob, dict) else blob
            ancestors = {k: set(v) for k, v in anc_dict.items() if not str(k).startswith("descendant:")}
        except Exception as exc:  # noqa: BLE001
            logger.info(f"ancestors load failed: {exc}")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -1.0
    no_improve = 0
    metric_log = output_dir / "dev_metrics.jsonl"
    metric_log.write_text("")

    for epoch in range(1, epochs + 1):
        model.train()
        tau = tau_for_epoch(epoch, epochs, args.tau_start, args.tau_end)
        if model.has_norm:
            model.norm_head.tau = tau
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
                loss, comps = joint_loss(out["losses"])
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
        metrics = evaluate_all(model, dev_loader, device, ancestors=ancestors)
        agg = aggregate_dev_metric(metrics)
        metrics.update({"epoch": epoch, "tau": tau, "train_loss": avg,
                        "aggregate": agg, "elapsed_sec": round(time.time() - t0, 2)})
        with metric_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")
        logger.info(f"epoch {epoch}/{epochs} tau={tau:.3f} loss={avg:.4f} agg={agg:.4f} {metrics}")

        if agg > best_score:
            best_score = agg
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True, parents=True)
            model.encoder.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            torch.save(
                {n: p.detach().cpu() for n, p in model.named_parameters() if not n.startswith("encoder.")},
                str(best_dir / "head_state.pt"),
            )
            torch.save(
                {n: p.detach().cpu() for n, p in joint_loss.named_parameters()},
                str(best_dir / "log_sigmas.pt"),
            )
            with (best_dir / "train_state.json").open("w", encoding="utf-8") as fh:
                json.dump({"epoch": epoch, "aggregate": agg, "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}}, fh, indent=2)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience and not smoke:
                logger.info(f"early stop at epoch {epoch}; no improvement for {no_improve} epochs")
                break

    summary = {
        "epochs": epochs,
        "best_aggregate": best_score,
        "smoke_test": bool(smoke),
        "device": str(device),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"done. best aggregate={best_score:.4f}")


if __name__ == "__main__":
    main()
