"""CANON Phase 3.3 -- Stage 1 per-head training.

Trains one head at a time on top of the SapBERT-pretrained encoder.
Schedule per head: 5 epochs encoder-frozen, then 5 epochs joint fine-tune.

CLI
---
    python scripts/train_stage1.py --head {ner,norm,rel} [--smoke-test]

Outputs land at outputs/phase3/stage1/<head>/{best,log.txt,training_summary.json,
dev_metrics.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    import config
    from canon_dataset import (
        BIO_ID_TO_LABEL,
        CanonDocDataset,
        NUM_BIO_LABELS,
        NUM_RELATION_LABELS,
        NO_RELATION_ID,
        RELATION_LABELS,
        collate_docs,
        load_soft_lookup,
    )
    from heads import MultiTaskModel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from canon_dataset import (
        BIO_ID_TO_LABEL,
        CanonDocDataset,
        NUM_BIO_LABELS,
        NUM_RELATION_LABELS,
        NO_RELATION_ID,
        RELATION_LABELS,
        collate_docs,
        load_soft_lookup,
    )
    from heads import MultiTaskModel


HEADS = ("ner", "norm", "rel")
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_ENCODER_LR = 1e-5
DEFAULT_HEAD_LR = 5e-5
DEFAULT_WARMUP = 0.05
DEFAULT_MAX_LENGTH = 512
DEFAULT_NEG_RATIO = 2.0
DEFAULT_MAX_PAIRS = 64

SMOKE_MAX_DOCS = 100
SMOKE_BATCH = 4
SMOKE_EPOCHS = 2


def setup_logging(log_path: Path, name: str) -> logging.Logger:
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


def build_dataset(jsonl_path: Path, tokenizer, soft, max_docs: Optional[int],
                  max_length: int, max_pairs: int, neg_ratio: float, seed: int) -> CanonDocDataset:
    return CanonDocDataset(
        jsonl_path,
        tokenizer,
        soft,
        max_length=max_length,
        max_docs=max_docs,
        neg_ratio=neg_ratio,
        max_pairs=max_pairs,
        seed=seed,
    )


def collate_fn_factory(pad_token_id: int):
    def _fn(batch):
        return collate_docs(batch, pad_token_id=pad_token_id)
    return _fn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def decode_entities_from_bio(bio: List[int]) -> List[Tuple[int, int, str]]:
    """Greedy BIO -> entity spans extraction."""
    spans: List[Tuple[int, int, str]] = []
    cur_start = -1
    cur_class = ""
    for i, tag_id in enumerate(bio):
        label = BIO_ID_TO_LABEL.get(int(tag_id), "O")
        if label == "O":
            if cur_start >= 0:
                spans.append((cur_start, i, cur_class))
                cur_start = -1
                cur_class = ""
        elif label.startswith("B-"):
            if cur_start >= 0:
                spans.append((cur_start, i, cur_class))
            cur_start = i
            cur_class = label[2:]
        else:  # I-
            cls = label[2:]
            if cur_start < 0 or cls != cur_class:
                # treat as B-
                if cur_start >= 0:
                    spans.append((cur_start, i, cur_class))
                cur_start = i
                cur_class = cls
    if cur_start >= 0:
        spans.append((cur_start, len(bio), cur_class))
    return spans


def ner_micro_f1(preds: List[List[Tuple[int, int, str]]],
                 golds: List[List[Tuple[int, int, str]]]) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        ps = set((a, b, c) for (a, b, c) in p)
        gs = set((a, b, c) for (a, b, c) in g)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def relation_macro_f1(pred: List[int], gold: List[int]) -> float:
    """Macro F1 over the 12 positive labels (no-relation excluded)."""
    f1s = []
    for cls in range(NUM_RELATION_LABELS):
        if cls == NO_RELATION_ID:
            continue
        tp = sum(1 for p, g in zip(pred, gold) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred, gold) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred, gold) if p != cls and g == cls)
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# Evaluation passes
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate(model: MultiTaskModel, loader: DataLoader, device: torch.device,
             head: str, ancestors: Optional[Dict[str, set]] = None) -> Dict[str, float]:
    model.eval()
    if head == "ner":
        preds, golds = [], []
        for batch in loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(batch, active_heads=("ner",))
            decoded = out["raw"]["ner"]["decoded"]
            mask = batch["attention_mask"].cpu()
            gold_bio = batch["bio_labels"].cpu()
            for b, seq in enumerate(decoded):
                L = int(mask[b].sum())
                preds.append(decode_entities_from_bio(seq[:L]))
                golds.append(decode_entities_from_bio(gold_bio[b][:L].tolist()))
        p, r, f = ner_micro_f1(preds, golds)
        return {"ner_precision": p, "ner_recall": r, "ner_f1": f}
    if head == "norm":
        exact = 0
        ancestor = 0
        total = 0
        for batch in loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(batch, active_heads=("norm",))
            scores = out["raw"]["norm"].get("scores")
            if scores is None or scores.numel() == 0:
                continue
            argmax = scores.argmax(dim=-1).cpu().tolist()
            cid_lookup = model.norm_head.concept_ids
            # Walk through targets in the batch dim order to align with rows
            row = 0
            for b in range(len(batch["norm_targets"])):
                ent_idx_list = batch["norm_entity_idx"][b]
                target_list = batch["norm_targets"][b]
                for k, _ent_i in enumerate(ent_idx_list):
                    if k >= len(target_list):
                        continue
                    if row >= scores.size(0):
                        break
                    pred_cid = cid_lookup[argmax[row]] if argmax[row] < len(cid_lookup) else None
                    target_p = target_list[k]
                    if not target_p:
                        row += 1
                        continue
                    gold_cid = max(target_p.items(), key=lambda kv: kv[1])[0]
                    total += 1
                    if pred_cid == gold_cid:
                        exact += 1
                        ancestor += 1
                    elif ancestors is not None and pred_cid is not None:
                        gold_anc = ancestors.get(str(gold_cid), set())
                        if pred_cid in gold_anc or pred_cid == gold_cid:
                            ancestor += 1
                    row += 1
        return {
            "norm_top1": exact / total if total else 0.0,
            "norm_ancestor": ancestor / total if total else 0.0,
            "norm_evaluated": float(total),
        }
    if head == "rel":
        preds, golds = [], []
        for batch in loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(batch, active_heads=("rel",))
            for b, logits in enumerate(out["raw"]["rel"]["per_doc_logits"]):
                if logits.numel() == 0:
                    continue
                pred = logits.argmax(dim=-1).cpu().tolist()
                gold = batch["pair_labels"][b].cpu().tolist()
                preds.extend(pred)
                golds.extend(gold)
        f = relation_macro_f1(preds, golds)
        return {"rel_macro_f1": f, "rel_pairs": float(len(preds))}
    raise ValueError(f"Unknown head: {head}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_head(args: argparse.Namespace) -> None:
    head = args.head
    output_dir = Path(args.output_dir) / head
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "log.txt", f"3.3:{head}")

    smoke = args.smoke_test
    epochs = SMOKE_EPOCHS if smoke else args.epochs
    batch_size = SMOKE_BATCH if smoke else args.batch_size
    # --max-docs CLI wins; smoke_test caps to SMOKE_MAX_DOCS otherwise; full run = unbounded.
    if getattr(args, "max_docs", None) is not None:
        max_docs = args.max_docs
    elif smoke:
        max_docs = SMOKE_MAX_DOCS
    else:
        max_docs = None
    half_epochs = max(1, epochs // 2)

    device_arg = getattr(args, "device", "auto")
    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = Path(getattr(args, "train_path", None) or (config.PHASE2_SPLITS_DIR / "train.jsonl"))
    dev_path   = Path(getattr(args, "dev_path",   None) or (config.PHASE2_SPLITS_DIR / "dev.jsonl"))

    logger.info(f"head={head} epochs={epochs} batch={batch_size} max_docs={max_docs} smoke={smoke}")
    logger.info(f"device={device} train={train_path} dev={dev_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(config.SAPBERT_ENCODER_DIR))
    pad_id = tokenizer.pad_token_id or 0
    soft = load_soft_lookup(config.SOFT_MAPPING_LOOKUP)

    train_ds = build_dataset(
        train_path,
        tokenizer, soft, max_docs, args.max_length, args.max_pairs, args.neg_ratio, seed=42)
    dev_ds = build_dataset(
        dev_path,
        tokenizer, soft, max_docs, args.max_length, args.max_pairs, args.neg_ratio, seed=43)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn_factory(pad_id))
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, collate_fn=collate_fn_factory(pad_id))

    # Model setup -- only the active head is constructed, others omitted.
    head_flags = {"ner": False, "norm": False, "rel": False}
    head_flags[head] = True
    with open(config.CONCEPT_INDEX_IDS) as fh:
        num_concepts = len(json.load(fh))
    model = MultiTaskModel(
        str(config.SAPBERT_ENCODER_DIR),
        num_concepts=num_concepts,
        ner=head_flags["ner"],
        norm=head_flags["norm"],
        rel=head_flags["rel"],
    )
    if head == "norm":
        model.norm_head.load_concept_index(config.CONCEPT_INDEX_IDS, config.CONCEPT_INDEX_EMB)
    model.to(device)

    # Ancestors for ancestor-match metric (norm head only).
    ancestors = None
    if head == "norm" and not smoke:
        try:
            import pickle
            with open(config.SNOMED_ANCESTORS_PKL, "rb") as fh:
                blob = pickle.load(fh)
            anc_dict = blob.get("ancestors", {}) if isinstance(blob, dict) else blob
            ancestors = {k: set(v) for k, v in anc_dict.items() if not str(k).startswith("descendant:")}
        except Exception as exc:  # noqa: BLE001
            logger.info(f"ancestors load failed: {exc}; skipping ancestor metric")
            ancestors = None

    # Optimizer setup. Distinct LRs for encoder and head (head LR ignored when frozen).
    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)

    # Steps-per-epoch is unknown for IterableDataset; use a generous estimate.
    approx_steps_per_epoch = max(1, (max_docs or 40000) // batch_size)
    total_steps = epochs * approx_steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_metric = -1.0
    metric_key = {"ner": "ner_f1", "norm": "norm_top1", "rel": "rel_macro_f1"}[head]
    metric_log_path = output_dir / "dev_metrics.jsonl"
    metric_log_path.write_text("")

    for epoch in range(1, epochs + 1):
        if epoch <= half_epochs:
            model.freeze_encoder(True)
            phase = "frozen"
        else:
            model.freeze_encoder(False)
            phase = "joint"
        model.train()

        t0 = time.time()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(batch, active_heads=(head,))
                losses = out["losses"]
                if not losses:
                    continue
                loss = sum(losses.values())
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach())
            steps += 1
        avg_loss = running_loss / max(steps, 1)
        metrics = evaluate(model, dev_loader, device, head, ancestors=ancestors)
        metrics["epoch"] = epoch
        metrics["phase"] = phase
        metrics["train_loss"] = avg_loss
        metrics["elapsed_sec"] = round(time.time() - t0, 2)
        with metric_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")
        logger.info(f"epoch {epoch}/{epochs} ({phase}) loss={avg_loss:.4f} {metrics}")

        score = metrics.get(metric_key, 0.0)
        if score > best_metric:
            best_metric = score
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True, parents=True)
            model.encoder.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            torch.save(
                {n: p.detach().cpu() for n, p in model.named_parameters() if not n.startswith("encoder.")},
                str(best_dir / "head_state.pt"),
            )
            with (best_dir / "train_state.json").open("w", encoding="utf-8") as fh:
                json.dump({"head": head, "epoch": epoch, "metric": metric_key, "score": best_metric}, fh, indent=2)

    summary = {
        "head": head,
        "epochs": epochs,
        "best_metric_key": metric_key,
        "best_metric": best_metric,
        "smoke_test": bool(smoke),
        "device": str(device),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"done. best {metric_key}={best_metric:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=HEADS, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--encoder-lr", type=float, default=DEFAULT_ENCODER_LR)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--neg-ratio", type=float, default=DEFAULT_NEG_RATIO)
    parser.add_argument("--output-dir", default=str(config.STAGE1_DIR))
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap dataset documents per epoch (full run if omitted).")
    parser.add_argument("--train-path", default=None,
                        help="Override train JSONL path (default: outputs/phase2/splits/train.jsonl).")
    parser.add_argument("--dev-path", default=None,
                        help="Override dev JSONL path (default: outputs/phase2/splits/dev.jsonl).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Force device; default auto picks CUDA when available.")
    args = parser.parse_args()
    train_head(args)


if __name__ == "__main__":
    main()
