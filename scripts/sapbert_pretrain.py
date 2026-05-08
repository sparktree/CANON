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
Production (BigRed200 A100, ~3-4 hours):
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

def precompute_term_tokens(
    concept_to_terms: Dict[str, List[str]],
    tokenizer,
    max_length: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, List[List[int]]], Dict[str, List[int]]]:
    """Tokenize every unique term once; index concepts by term position.

    Returns:
        term_tokens: dict with keys 'input_ids' and 'attention_mask', each a
                     List[List[int]] of length N_unique (unpadded; collate pads).
        concept_to_idxs: dict[concept_id -> list of term indices into term_tokens].
    """
    unique_terms: List[str] = []
    term_to_idx: Dict[str, int] = {}
    concept_to_idxs: Dict[str, List[int]] = {}
    for cid, terms in concept_to_terms.items():
        idxs: List[int] = []
        for t in terms:
            j = term_to_idx.get(t)
            if j is None:
                j = len(unique_terms)
                term_to_idx[t] = j
                unique_terms.append(t)
            idxs.append(j)
        concept_to_idxs[cid] = idxs
    logger.info(f"  pre-tokenizing {len(unique_terms):,} unique terms (max_length={max_length}) ...")
    t0 = time.time()
    encoded = tokenizer(
        unique_terms,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    logger.info(f"  tokenization done in {time.time() - t0:.1f}s")
    return (
        {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]},
        concept_to_idxs,
    )


class SnomedPairIndexDataset(IterableDataset):
    """Iterable stream of (idx_a, idx_b, concept_id) triples for one epoch.

    Indices reference a pre-tokenized term pool, so collate avoids per-batch
    tokenizer calls. Each iteration shuffles concepts then yields up to
    `max_pairs_per_concept` random unique index pairs per concept; stops when
    `pairs_per_epoch` is reached.

    Single-process (DataLoader num_workers=0) is intended; IterableDataset
    sharding across workers is overkill given how cheap this generation is.
    """

    def __init__(
        self,
        concept_to_idxs: Dict[str, List[int]],
        pairs_per_epoch: int,
        max_pairs_per_concept: int,
        seed: int,
    ) -> None:
        self.items = list(concept_to_idxs.items())
        self.pairs_per_epoch = pairs_per_epoch
        self.max_pairs_per_concept = max_pairs_per_concept
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[int, int, str]]:
        rng = random.Random(self.seed)
        items = list(self.items)
        rng.shuffle(items)
        n = 0
        for concept_id, idxs in items:
            if len(idxs) < 2:
                continue
            n_max = min(len(idxs) * (len(idxs) - 1) // 2, self.max_pairs_per_concept)
            for _ in range(n_max):
                a, b = rng.sample(idxs, 2)
                yield a, b, concept_id
                n += 1
                if n >= self.pairs_per_epoch:
                    return


def collate_pair_indices(
    batch: List[Tuple[int, int, str]],
    term_tokens: Dict[str, List[List[int]]],
    tokenizer,
):
    input_ids_pool = term_tokens["input_ids"]
    attn_pool = term_tokens["attention_mask"]
    features: List[Dict[str, List[int]]] = []
    labels: List[int] = []
    label_to_idx: Dict[str, int] = {}
    for a, b, cid in batch:
        if cid not in label_to_idx:
            label_to_idx[cid] = len(label_to_idx)
        idx = label_to_idx[cid]
        features.append({"input_ids": input_ids_pool[a], "attention_mask": attn_pool[a]})
        features.append({"input_ids": input_ids_pool[b], "attention_mask": attn_pool[b]})
        labels.extend([idx, idx])
    encoded = tokenizer.pad(features, padding=True, return_tensors="pt")
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
    model, loader, optimizer, scheduler, scaler, device, log_interval, logger,
    start_step=0, partial_save=None,
) -> Dict[str, float]:
    """Run one epoch.

    `start_step`: skip this many batches before doing real training (used to
        fast-forward a resumed mid-epoch session to its saved step).
    `partial_save`: optional dict with keys {dst, every_steps, tokenizer, epoch,
        train_state_extra} -- when set, writes a full checkpoint to `dst` every
        `every_steps` real training steps. `epoch` is the 0-indexed in-progress
        epoch number; on resume it is fed back in as start_epoch.
    """
    model.train()
    losses: List[float] = []
    t0 = time.time()
    real_step = start_step  # 0-indexed count of real training steps done this epoch
    for raw_step, (encoded, labels) in enumerate(loader):
        if raw_step < start_step:
            continue  # cheap skip; tokens are pre-built so collate is just tensor stacking
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
        real_step += 1
        if real_step % log_interval == 0:
            recent = losses[-log_interval:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  step {real_step:>5d}: loss={avg:.4f}  lr={lr:.2e}  "
                f"elapsed={elapsed:.0f}s"
            )
        if partial_save and partial_save["every_steps"] > 0 \
                and real_step % partial_save["every_steps"] == 0:
            state = dict(partial_save["train_state_extra"])
            state.update({
                "epoch": partial_save["epoch"],     # 0-indexed in-progress epoch
                "step_in_epoch": real_step,
                "partial": True,
            })
            save_checkpoint(
                model, partial_save["tokenizer"], partial_save["dst"], state,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            )
            logger.info(f"  partial ckpt @ step {real_step} -> {partial_save['dst'].name}/")
    epoch_avg = sum(losses) / max(len(losses), 1)
    return {"avg_loss": epoch_avg, "steps": len(losses), "elapsed_s": time.time() - t0}


def save_checkpoint(
    model,
    tokenizer,
    dst: Path,
    train_state: Dict[str, object],
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # save_pretrained handles model bin + config; tokenizer files written separately
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)
    if train_state:
        (dst / "train_state.json").write_text(
            json.dumps(train_state, indent=2), encoding="utf-8"
        )
    if optimizer is not None:
        torch.save(optimizer.state_dict(), dst / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), dst / "scheduler.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), dst / "scaler.pt")


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
    p.add_argument("--save-every-steps",  type=int,   default=500,
                   help="Save a partial checkpoint to checkpoints/partial/ every N steps "
                        "so SLURM walltime kills don't waste mid-epoch progress. "
                        "Set to 0 to disable.")
    p.add_argument("--smoke-test",        action="store_true",
                   help="Quick local sanity run: 1 epoch, ~1 K pairs, batch 8.")
    p.add_argument("--resume-from",       default=None,
                   help="Resume training from a checkpoint dir (e.g. .../checkpoints/epoch_03 "
                        "or .../checkpoints/partial). Loads model + optimizer + scheduler + scaler "
                        "state and continues with the same total_steps / warmup_steps schedule. "
                        "Requires optimizer.pt and scheduler.pt; original --epochs / --batch-size "
                        "/ --pairs-per-concept must be preserved.")
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

    model_src = args.resume_from or args.biolinkbert_dir
    logger.info(f"loading model + tokenizer from {model_src} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_src)
    model = AutoModel.from_pretrained(model_src)
    model.to(device)

    term_tokens, concept_to_idxs = precompute_term_tokens(
        concept_to_terms, tokenizer, args.max_length, logger
    )

    steps_per_epoch = max(pairs_per_epoch // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(total_steps * DEFAULT_WARMUP_RATIO), 1)
    logger.info(f"  steps_per_epoch       = {steps_per_epoch:,}")
    logger.info(f"  total_steps           = {total_steps:,}  (warmup={warmup_steps})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    start_epoch = 0
    start_step = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        state_path = resume_path / "train_state.json"
        if not state_path.exists():
            raise SystemExit(f"--resume-from: missing train_state.json at {state_path}")
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        saved_total = saved.get("total_steps")
        saved_warmup = saved.get("warmup_steps")
        if saved_total is not None and saved_total != total_steps:
            raise SystemExit(
                f"--resume-from: total_steps mismatch (saved={saved_total}, current={total_steps}). "
                f"Use the same --epochs / --batch-size / --pairs-per-concept as the original run."
            )
        if saved_warmup is not None and saved_warmup != warmup_steps:
            raise SystemExit(
                f"--resume-from: warmup_steps mismatch (saved={saved_warmup}, current={warmup_steps})."
            )
        opt_path = resume_path / "optimizer.pt"
        sched_path = resume_path / "scheduler.pt"
        if not opt_path.exists() or not sched_path.exists():
            raise SystemExit(
                f"--resume-from: {resume_path} is model-only (no optimizer.pt / scheduler.pt). "
                f"This checkpoint predates resume support; either start fresh or wait for "
                f"a future epoch checkpoint that includes full state."
            )
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        scheduler.load_state_dict(torch.load(sched_path, map_location=device))
        if scaler is not None:
            scaler_path = resume_path / "scaler.pt"
            if scaler_path.exists():
                scaler.load_state_dict(torch.load(scaler_path, map_location=device))
        if saved.get("partial"):
            # Mid-epoch partial: saved.epoch is 0-indexed in-progress epoch.
            start_epoch = int(saved["epoch"])
            start_step = int(saved.get("step_in_epoch", 0))
            logger.info(
                f"resumed from partial {resume_path} -- epoch {start_epoch + 1}/{args.epochs} "
                f"@ step {start_step} (scheduler last_step={scheduler.last_epoch})"
            )
        else:
            start_epoch = int(saved["epoch"])
            logger.info(
                f"resumed from {resume_path} -- starting at epoch {start_epoch + 1}/{args.epochs} "
                f"(scheduler last_step={scheduler.last_epoch})"
            )

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    partial_dir = Path(args.checkpoints_dir) / "partial"
    train_state_extra = {
        "biolinkbert_dir": args.biolinkbert_dir,
        "smoke_test": args.smoke_test,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    epoch_logs: List[Dict[str, object]] = []

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"epoch {epoch + 1}/{args.epochs} starting")
        dataset = SnomedPairIndexDataset(
            concept_to_idxs,
            pairs_per_epoch=pairs_per_epoch,
            max_pairs_per_concept=args.pairs_per_concept,
            seed=args.seed + epoch,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_pair_indices(b, term_tokens, tokenizer),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        partial_save = {
            "dst": partial_dir,
            "every_steps": args.save_every_steps,
            "tokenizer": tokenizer,
            "epoch": epoch,
            "train_state_extra": train_state_extra,
        } if args.save_every_steps > 0 else None
        epoch_stats = train_one_epoch(
            model, loader, optimizer, scheduler, scaler, device,
            args.log_interval, logger,
            start_step=(start_step if epoch == start_epoch else 0),
            partial_save=partial_save,
        )
        logger.info(
            f"epoch {epoch + 1}: avg_loss={epoch_stats['avg_loss']:.4f}  "
            f"steps={epoch_stats['steps']}  elapsed={epoch_stats['elapsed_s']:.0f}s"
        )
        ckpt_dir = Path(args.checkpoints_dir) / f"epoch_{epoch + 1:02d}"
        save_checkpoint(
            model, tokenizer, ckpt_dir,
            {
                "epoch": epoch + 1,
                "avg_loss": epoch_stats["avg_loss"],
                "steps": epoch_stats["steps"],
                "elapsed_s": epoch_stats["elapsed_s"],
                "biolinkbert_dir": args.biolinkbert_dir,
                "smoke_test": args.smoke_test,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            },
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        logger.info(f"  saved checkpoint -> {relative_to_repo(ckpt_dir)}")
        epoch_logs.append({
            "epoch": epoch + 1,
            "avg_loss": epoch_stats["avg_loss"],
            "elapsed_s": epoch_stats["elapsed_s"],
        })
        # The partial/ dir, if it exists, holds in-progress state for the epoch
        # we just finished. It's now stale; remove so auto-resume picks the
        # epoch_NN/ checkpoint instead of a partial that's older than this one.
        if partial_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(partial_dir, ignore_errors=True)

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
