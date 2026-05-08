"""SapBERT-style contrastive pre-training of BioLinkBERT (CANON Phase 3.1).

Construction:
  * Source corpus -- active SNOMED CT descriptions (FSN + Synonym typeIds).
    Grouped by conceptId. FSN semantic-tag suffixes ("(disorder)" etc.)
    are stripped because they are metadata, not natural language.
  * Positive pairs -- two distinct descriptions of the same concept.
  * In-batch negatives -- every other concept's descriptions in the batch.
  * Loss -- multi-similarity (Wang et al., CVPR 2019) with hard mining.
            Same loss family SapBERT uses; no extra dependency.
  * Pooler -- [CLS] token, then L2-normalize.

Output:
  * Per-epoch checkpoints under outputs/phase3/sapbert/checkpoints/epoch_NN/
  * Final fine-tuned encoder under outputs/phase3/sapbert/encoder/
  * Train log at outputs/phase3/sapbert/log.txt

Both the per-epoch and final directories are HuggingFace-loadable
(`AutoModel.from_pretrained(...)`) so Phase 3.2 can pick the encoder up
without any custom loader.

Portability (Indiana BigRed200 + Slate)
---------------------------------------
The script honors three environment variables for HPC use:
  CANON_BIOLINKBERT  -- path to BioLinkBERT weights (default: WORKSPACE_ROOT/BioLinkBERT/)
  CANON_DATA_ROOT    -- path to corpora directory (default: WORKSPACE_ROOT/Data/)
  CANON_OUTPUTS_ROOT -- path to phase outputs directory (default: REPO_ROOT/outputs/)
                       use a Slate path on BigRed200 to keep ~4 GB of checkpoints
                       off the small home filesystem.

Single-GPU and CPU runs share the same code; mixed precision auto-enables on CUDA.
A smoke-test mode (`--smoke-test`) caps everything to 1 epoch / 1 K pairs / batch 8
so the orchestration is testable on a laptop without a GPU.

CLI
---
Production (BigRed200 H100, ~3-4 hours):
    python scripts/sapbert_pretrain.py --epochs 8 --batch-size 128 --lr 2e-5
Smoke test (any machine, ~5 min):
    python scripts/sapbert_pretrain.py --smoke-test

The companion slurm/sapbert.slurm script wraps the production invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

try:
    from config import BIOLINKBERT_DIR, REPO_ROOT, SNOMED_FILES, relative_to_repo
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import BIOLINKBERT_DIR, REPO_ROOT, SNOMED_FILES, relative_to_repo


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FSN_TYPE_ID = "900000000000003001"   # Fully specified name
SYN_TYPE_ID = "900000000000013009"   # Synonym

OUTPUTS_ROOT = Path(os.environ.get("CANON_OUTPUTS_ROOT", str(REPO_ROOT / "outputs")))
OUTPUT_DIR      = OUTPUTS_ROOT / "phase3" / "sapbert"
ENCODER_OUT     = OUTPUT_DIR / "encoder"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH        = OUTPUT_DIR / "log.txt"
SUMMARY_JSON    = OUTPUT_DIR / "training_summary.json"

DEFAULT_BATCH_SIZE        = 128     # input batch is 2x this (two terms per concept)
DEFAULT_EPOCHS            = 8
DEFAULT_LR                = 2e-5
DEFAULT_MAX_LENGTH        = 64      # SNOMED descriptions are short
DEFAULT_PAIRS_PER_CONCEPT = 10
DEFAULT_WARMUP_RATIO      = 0.05
DEFAULT_NUM_WORKERS       = 0       # IterableDataset works simplest single-process

SMOKE_TEST_PAIRS  = 1000
SMOKE_TEST_BATCH  = 8

# Multi-similarity loss hyperparams (SapBERT defaults).
MS_ALPHA = 2.0
MS_BETA  = 50.0
MS_BASE  = 0.5

_FSN_SUFFIX = re.compile(r"\s*\([^)]+\)\s*$")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _strip_fsn_tag(term: str) -> str:
    return _FSN_SUFFIX.sub("", term).strip()


def load_snomed_descriptions(desc_path: Path, logger: logging.Logger) -> Dict[str, List[str]]:
    """Group active descriptions by conceptId.

    Returns {concept_id: [unique_term, ...]} for concepts with at least 2
    distinct terms (single-term concepts can't form a positive pair).
    """
    if not desc_path.exists():
        raise FileNotFoundError(f"SNOMED descriptions not found at {desc_path}")
    concept_to_terms: Dict[str, List[str]] = {}
    seen_per_concept: Dict[str, set] = {}
    n_rows = 0
    n_kept = 0
    with desc_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            n_rows += 1
            if row.get("active") != "1":
                continue
            type_id = row.get("typeId", "")
            if type_id not in (FSN_TYPE_ID, SYN_TYPE_ID):
                continue
            cid = row.get("conceptId", "")
            term = (row.get("term") or "").strip()
            if not cid or not term:
                continue
            if type_id == FSN_TYPE_ID:
                term = _strip_fsn_tag(term)
            if not term:
                continue
            seen = seen_per_concept.setdefault(cid, set())
            if term in seen:
                continue
            seen.add(term)
            concept_to_terms.setdefault(cid, []).append(term)
            n_kept += 1
    pair_eligible = {c: t for c, t in concept_to_terms.items() if len(t) >= 2}
    logger.info(
        f"  scanned {n_rows:,} description rows, kept {n_kept:,} unique active terms "
        f"across {len(concept_to_terms):,} concepts; "
        f"{len(pair_eligible):,} concepts have >= 2 distinct terms"
    )
    return pair_eligible


def count_pairs_per_epoch(
    concept_to_terms: Dict[str, List[str]],
    pairs_per_concept: int,
) -> int:
    return sum(
        min(len(t) * (len(t) - 1) // 2, pairs_per_concept)
        for t in concept_to_terms.values()
    )


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

class SnomedPairDataset(IterableDataset):
    """Iterable stream of (term_a, term_b, concept_id) triples for one epoch.

    Each iteration shuffles concepts then yields up to `pairs_per_concept`
    random unique pairs per concept. Stops when `pairs_per_epoch` is reached.

    Single-process (DataLoader num_workers=0) is intended; IterableDataset
    sharding across workers is overkill given how cheap this generation is.
    """

    def __init__(
        self,
        concept_to_terms: Dict[str, List[str]],
        pairs_per_epoch: int,
        max_pairs_per_concept: int,
        seed: int,
    ) -> None:
        self.concept_terms = list(concept_to_terms.items())
        self.pairs_per_epoch = pairs_per_epoch
        self.max_pairs_per_concept = max_pairs_per_concept
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        rng = random.Random(self.seed)
        concepts = list(self.concept_terms)
        rng.shuffle(concepts)
        n = 0
        for concept_id, terms in concepts:
            n_max = min(len(terms) * (len(terms) - 1) // 2, self.max_pairs_per_concept)
            for _ in range(n_max):
                a, b = rng.sample(terms, 2)
                yield a, b, concept_id
                n += 1
                if n >= self.pairs_per_epoch:
                    return


def collate_pairs(
    batch: List[Tuple[str, str, str]],
    tokenizer,
    max_length: int,
):
    terms: List[str] = []
    labels: List[int] = []
    label_to_idx: Dict[str, int] = {}
    for a, b, cid in batch:
        if cid not in label_to_idx:
            label_to_idx[cid] = len(label_to_idx)
        idx = label_to_idx[cid]
        terms.extend([a, b])
        labels.extend([idx, idx])
    encoded = tokenizer(
        terms,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Multi-similarity loss
# ---------------------------------------------------------------------------

def multi_similarity_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = MS_ALPHA,
    beta: float = MS_BETA,
    base: float = MS_BASE,
) -> torch.Tensor:
    """Wang et al. CVPR 2019 multi-similarity loss; SapBERT's choice.

    Args:
        embeddings: (B, D) L2-normalized.
        labels:     (B,) integer class labels (concept ids re-indexed per batch).
    """
    sim = embeddings @ embeddings.t()  # (B, B)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    pos_mask = labels_eq & ~self_mask
    neg_mask = ~labels_eq

    losses: List[torch.Tensor] = []
    for i in range(labels.size(0)):
        pos_sim = sim[i][pos_mask[i]]
        neg_sim = sim[i][neg_mask[i]]
        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            continue
        max_neg = neg_sim.max()
        min_pos = pos_sim.min()
        sel_pos = pos_sim[pos_sim < max_neg + base]
        sel_neg = neg_sim[neg_sim > min_pos - base]
        if sel_pos.numel() == 0 or sel_neg.numel() == 0:
            continue
        pos_loss = (1.0 / alpha) * torch.log1p(
            torch.exp(-alpha * (sel_pos - base)).sum()
        )
        neg_loss = (1.0 / beta) * torch.log1p(
            torch.exp(beta * (sel_neg - base)).sum()
        )
        losses.append(pos_loss + neg_loss)
    if not losses:
        # No anchors had non-trivial pos+neg pairs after hard mining (rare;
        # happens occasionally on small AMP fp16 batches where similarities
        # collapse into ties). Return a graph-connected zero so the AMP
        # GradScaler can still record inf checks for this step.
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def encode_batch(model, encoded, device) -> torch.Tensor:
    encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
    out = model(**encoded)
    cls = out.last_hidden_state[:, 0]
    return F.normalize(cls, dim=-1)


def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, device, log_interval, logger
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    t0 = time.time()
    for step, (encoded, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                emb = encode_batch(model, encoded, device)
                loss = multi_similarity_loss(emb, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            emb = encode_batch(model, encoded, device)
            loss = multi_similarity_loss(emb, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % log_interval == 0:
            recent = losses[-log_interval:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  step {step + 1:>5d}: loss={avg:.4f}  lr={lr:.2e}  "
                f"elapsed={elapsed:.0f}s"
            )
    epoch_avg = sum(losses) / max(len(losses), 1)
    return {"avg_loss": epoch_avg, "steps": len(losses), "elapsed_s": time.time() - t0}


def save_checkpoint(model, tokenizer, dst: Path, train_state: Dict[str, object]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # save_pretrained handles model bin + config; tokenizer files written separately
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)
    if train_state:
        (dst / "train_state.json").write_text(
            json.dumps(train_state, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SapBERT-style pre-training (CANON Phase 3.1)")
    p.add_argument("--biolinkbert-dir", default=str(BIOLINKBERT_DIR),
                   help="Path to BioLinkBERT (or PubMedBERT) HF model directory.")
    p.add_argument("--snomed-descriptions", default=str(SNOMED_FILES["descriptions"]),
                   help="Path to sct2_Description_Snapshot-en_*.txt.")
    p.add_argument("--output-dir",      default=str(ENCODER_OUT),
                   help="Where to write the final fine-tuned encoder.")
    p.add_argument("--checkpoints-dir", default=str(CHECKPOINTS_DIR),
                   help="Where to write per-epoch checkpoints.")
    p.add_argument("--epochs",            type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",        type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",                type=float, default=DEFAULT_LR)
    p.add_argument("--max-length",        type=int,   default=DEFAULT_MAX_LENGTH)
    p.add_argument("--pairs-per-concept", type=int,   default=DEFAULT_PAIRS_PER_CONCEPT)
    p.add_argument("--max-pairs",         type=int,   default=None,
                   help="Cap pairs per epoch (default: full coverage).")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--num-workers",       type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--mixed-precision",   default="auto", choices=("auto", "yes", "no"))
    p.add_argument("--log-interval",      type=int,   default=50)
    p.add_argument("--smoke-test",        action="store_true",
                   help="Quick local sanity run: 1 epoch, ~1 K pairs, batch 8.")
    return p.parse_args()


def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("canon.sapbert")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [3.1] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = SMOKE_TEST_BATCH
        args.max_pairs = SMOKE_TEST_PAIRS
        args.log_interval = 5

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info(f"phase 3.1 SapBERT pre-training; smoke_test={args.smoke_test}")
    logger.info(f"  biolinkbert_dir       = {args.biolinkbert_dir}")
    logger.info(f"  snomed_descriptions   = {args.snomed_descriptions}")
    logger.info(f"  output_dir            = {args.output_dir}")
    logger.info(f"  checkpoints_dir       = {args.checkpoints_dir}")
    logger.info(f"  epochs                = {args.epochs}")
    logger.info(f"  batch_size (concepts) = {args.batch_size}  (effective input: {args.batch_size * 2})")
    logger.info(f"  lr                    = {args.lr}")
    logger.info(f"  pairs_per_concept     = {args.pairs_per_concept}")

    # Heavy imports deferred until logging is up.
    from transformers import (  # noqa: E402
        AutoModel, AutoTokenizer, get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (
        args.mixed_precision == "yes"
        or (args.mixed_precision == "auto" and device.type == "cuda")
    )
    logger.info(f"  device                = {device}  (mixed_precision={use_amp})")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    logger.info("loading SNOMED descriptions ...")
    concept_to_terms = load_snomed_descriptions(Path(args.snomed_descriptions), logger)
    pairs_per_epoch = count_pairs_per_epoch(concept_to_terms, args.pairs_per_concept)
    if args.max_pairs:
        pairs_per_epoch = min(pairs_per_epoch, args.max_pairs)
    logger.info(f"  pairs_per_epoch       = {pairs_per_epoch:,}")

    logger.info("loading model + tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.biolinkbert_dir)
    model = AutoModel.from_pretrained(args.biolinkbert_dir)
    model.to(device)

    steps_per_epoch = max(pairs_per_epoch // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(total_steps * DEFAULT_WARMUP_RATIO), 1)
    logger.info(f"  steps_per_epoch       = {steps_per_epoch:,}")
    logger.info(f"  total_steps           = {total_steps:,}  (warmup={warmup_steps})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    epoch_logs: List[Dict[str, object]] = []

    for epoch in range(args.epochs):
        logger.info(f"epoch {epoch + 1}/{args.epochs} starting")
        dataset = SnomedPairDataset(
            concept_to_terms,
            pairs_per_epoch=pairs_per_epoch,
            max_pairs_per_concept=args.pairs_per_concept,
            seed=args.seed + epoch,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_pairs(b, tokenizer, args.max_length),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        epoch_stats = train_one_epoch(
            model, loader, optimizer, scheduler, scaler, device,
            args.log_interval, logger,
        )
        logger.info(
            f"epoch {epoch + 1}: avg_loss={epoch_stats['avg_loss']:.4f}  "
            f"steps={epoch_stats['steps']}  elapsed={epoch_stats['elapsed_s']:.0f}s"
        )
        ckpt_dir = Path(args.checkpoints_dir) / f"epoch_{epoch + 1:02d}"
        save_checkpoint(model, tokenizer, ckpt_dir, {
            "epoch": epoch + 1,
            "avg_loss": epoch_stats["avg_loss"],
            "steps": epoch_stats["steps"],
            "elapsed_s": epoch_stats["elapsed_s"],
            "biolinkbert_dir": args.biolinkbert_dir,
            "smoke_test": args.smoke_test,
        })
        logger.info(f"  saved checkpoint -> {relative_to_repo(ckpt_dir)}")
        epoch_logs.append({
            "epoch": epoch + 1,
            "avg_loss": epoch_stats["avg_loss"],
            "elapsed_s": epoch_stats["elapsed_s"],
        })

    save_checkpoint(model, tokenizer, Path(args.output_dir), {
        "epochs": args.epochs,
        "biolinkbert_dir": args.biolinkbert_dir,
        "smoke_test": args.smoke_test,
        "final_avg_loss": epoch_logs[-1]["avg_loss"] if epoch_logs else None,
    })
    logger.info(f"final encoder -> {relative_to_repo(Path(args.output_dir))}")

    summary = {
        "status": "completed",
        "smoke_test": args.smoke_test,
        "biolinkbert_dir": relative_to_repo(Path(args.biolinkbert_dir)),
        "snomed_descriptions": relative_to_repo(Path(args.snomed_descriptions)),
        "epochs": args.epochs,
        "batch_size_concepts": args.batch_size,
        "lr": args.lr,
        "pairs_per_concept": args.pairs_per_concept,
        "pairs_per_epoch": pairs_per_epoch,
        "concepts_with_pairs": len(concept_to_terms),
        "device": str(device),
        "mixed_precision": use_amp,
        "epoch_logs": epoch_logs,
        "outputs": {
            "encoder": relative_to_repo(Path(args.output_dir)),
            "checkpoints": relative_to_repo(Path(args.checkpoints_dir)),
            "log": relative_to_repo(LOG_PATH),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"summary -> {relative_to_repo(SUMMARY_JSON)}")
    logger.info("done")


if __name__ == "__main__":
    main()
