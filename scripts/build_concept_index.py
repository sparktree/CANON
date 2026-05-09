"""Precompute SapBERT embeddings for the SNOMED candidate set (CANON Phase 3.2).

Reads the active SNOMED CT description file, picks one preferred term per
concept (FSN with semantic-tag stripped, falling back to the first synonym),
encodes the terms with the SapBERT-pretrained encoder, L2-normalizes, writes:

  outputs/phase3/concept_index/concept_ids.json        list[str], length N
  outputs/phase3/concept_index/concept_emb.safetensors {"embeddings": (N, H) fp16}
  outputs/phase3/concept_index/build_summary.json      hyperparams + counts

The full SNOMED snapshot has ~370K active concepts. Smoke mode caps to 5,000.

CLI
---
Production:
    python scripts/build_concept_index.py
Smoke:
    python scripts/build_concept_index.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

try:
    import config
    from utils import choose_torch_device
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from utils import choose_torch_device


FSN_TYPE_ID = "900000000000003001"
SYN_TYPE_ID = "900000000000013009"
_FSN_SUFFIX = re.compile(r"\s*\([^)]+\)\s*$")

DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_LENGTH = 64
SMOKE_CAP = 5000


def _strip_fsn_tag(term: str) -> str:
    return _FSN_SUFFIX.sub("", term).strip()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_concept_index")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [build_concept_index] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_preferred_terms(desc_path: Path, logger: logging.Logger) -> Dict[str, str]:
    """Return {concept_id: preferred_term}.

    Preference order: FSN with semantic-tag stripped > first active synonym.
    """
    fsn: Dict[str, str] = {}
    syn: Dict[str, str] = {}
    rows = 0
    with desc_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {name: idx for idx, name in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[col["active"]] != "1":
                continue
            cid = parts[col["conceptId"]]
            tid = parts[col["typeId"]]
            term = parts[col["term"]]
            if tid == FSN_TYPE_ID:
                if cid not in fsn:
                    fsn[cid] = _strip_fsn_tag(term)
            elif tid == SYN_TYPE_ID:
                if cid not in syn:
                    syn[cid] = term.strip()
            rows += 1
    out: Dict[str, str] = {}
    for cid, term in fsn.items():
        out[cid] = term
    for cid, term in syn.items():
        if cid not in out and term:
            out[cid] = term
    logger.info(f"scanned {rows:,} desc rows, kept {len(out):,} unique active concepts")
    return out


def load_soft_lookup_terms(lookup_path: Path, logger: logging.Logger) -> Dict[str, str]:
    """Return one candidate term per SNOMED concept from Phase 2 soft mappings."""
    with lookup_path.open("r", encoding="utf-8") as fh:
        lookup = json.load(fh)

    best: Dict[str, Tuple[float, str]] = {}
    candidate_rows = 0
    for candidates in lookup.values():
        for cand in candidates:
            cid = str(cand.get("snomed_id") or "").strip()
            term = str(cand.get("term") or "").strip()
            if not cid or not term:
                continue
            prob = float(cand.get("prob") or 0.0)
            prev = best.get(cid)
            if prev is None or prob > prev[0]:
                best[cid] = (prob, term)
            candidate_rows += 1

    out = {cid: term for cid, (_prob, term) in best.items()}
    logger.info(
        f"scanned {candidate_rows:,} soft-mapping candidate rows, "
        f"kept {len(out):,} unique candidate concepts"
    )
    return out


@torch.inference_mode()
def encode_concepts(
    concept_terms: List[Tuple[str, str]],
    encoder_dir: Path,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[List[str], torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    model = AutoModel.from_pretrained(str(encoder_dir)).to(device).eval()

    ids: List[str] = [c for c, _ in concept_terms]
    terms: List[str] = [t for _, t in concept_terms]

    chunks: List[torch.Tensor] = []
    n = len(terms)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = terms[start:end]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        cls = out.last_hidden_state[:, 0]
        cls = F.normalize(cls, dim=-1)
        chunks.append(cls.detach().to(torch.float16).cpu())
        if (start // batch_size) % 20 == 0:
            logger.info(f"  encoded {end:,}/{n:,}")
    emb = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, model.config.hidden_size, dtype=torch.float16)
    return ids, emb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--encoder-dir", default=str(config.SAPBERT_ENCODER_DIR))
    parser.add_argument("--descriptions", default=str(config.SNOMED_FILES["descriptions"]))
    parser.add_argument("--output-dir", default=str(config.CONCEPT_INDEX_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--from-soft-lookup", action="store_true",
                        help="Index only concepts appearing in Phase 2 soft mappings.")
    parser.add_argument("--soft-lookup", default=str(config.SOFT_MAPPING_LOOKUP))
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--cap", type=int, default=None,
                        help="Cap number of concepts (smoke override).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "log.txt")

    cap = args.cap
    if args.smoke_test and cap is None:
        cap = SMOKE_CAP

    encoder_dir = Path(args.encoder_dir)
    if not encoder_dir.is_dir():
        # Fall back to the BioLinkBERT base if SapBERT pretraining hasn't run.
        fallback = Path(os.environ.get("CANON_BIOLINKBERT", str(config.BIOLINKBERT_DIR)))
        logger.info(f"encoder_dir {encoder_dir} missing; falling back to {fallback}")
        encoder_dir = fallback

    desc_path = Path(args.descriptions)
    logger.info(f"descriptions = {desc_path}")
    logger.info(f"soft_lookup  = {args.soft_lookup}")
    logger.info(f"encoder      = {encoder_dir}")
    logger.info(f"output_dir   = {output_dir}")
    logger.info(f"smoke_test   = {args.smoke_test}  cap = {cap}  from_soft_lookup = {args.from_soft_lookup}")

    t0 = time.time()
    if args.from_soft_lookup:
        preferred = load_soft_lookup_terms(Path(args.soft_lookup), logger)
    else:
        preferred = load_preferred_terms(desc_path, logger)

    items = sorted(preferred.items(), key=lambda kv: kv[0])
    if cap is not None and len(items) > cap:
        items = items[:cap]
    logger.info(f"encoding {len(items):,} concepts (batch={args.batch_size}, max_len={args.max_length})")

    device = choose_torch_device(args.device)
    ids, emb = encode_concepts(
        items,
        encoder_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        logger=logger,
    )

    ids_path = output_dir / "concept_ids.json"
    emb_path = output_dir / "concept_emb.safetensors"
    summary_path = output_dir / "build_summary.json"

    with ids_path.open("w", encoding="utf-8") as fh:
        json.dump(ids, fh)
    save_file({"embeddings": emb}, str(emb_path))

    summary = {
        "mode": "smoke" if args.smoke_test else "full",
        "encoder_dir": str(encoder_dir),
        "descriptions": str(desc_path),
        "num_concepts": len(ids),
        "embedding_dim": int(emb.size(-1)) if emb.numel() else 0,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "cap": cap,
        "device": str(device),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"wrote {emb_path} ({emb.shape}) + {ids_path} + {summary_path}")
    logger.info(f"elapsed {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
