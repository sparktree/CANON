#!/bin/bash
# CANON Phase 3.1 — SapBERT-style pre-training of BioLinkBERT on SNOMED synonyms.
#
# Submit from BigRed200 with:
#     sbatch slurm/sapbert.slurm
#
# Adjust account, partition, and module load lines for your IU group.
# The script assumes your CANON checkout, BioLinkBERT weights, and Data/
# directory all live on Slate to keep ~/ off the small home filesystem.

#SBATCH --job-name=canon-sapbert
#SBATCH --partition=gpu
#SBATCH --account=r00877
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1                  # one A100 is plenty for SapBERT-base
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/N/slate/alikh/canon/logs/sapbert.%j.out
#SBATCH --error=/N/slate/alikh/canon/logs/sapbert.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khurramali1997@gmail.com

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load python/3.11
source /N/slate/alikh/canon/.venv/bin/activate

# ---------------------------------------------------------------------------
# Paths -- code in $HOME (small), data + model + outputs on Slate
# ---------------------------------------------------------------------------
export CANON_ROOT="/N/slate/alikh/canon"
export CANON_REPO="/N/u/alikh/BigRed200/CANON"
export DATA_ROOT="/N/slate/alikh/lingl675/data/Data"
export CANON_DATA_ROOT="$DATA_ROOT"
export CANON_BIOLINKBERT="$CANON_ROOT/BioLinkBERT"
export CANON_OUTPUTS_ROOT="$CANON_ROOT/outputs"

# HuggingFace cache on Slate so models/tokenizers are reused across jobs
export HF_HOME="$CANON_ROOT/.hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"

cd "$CANON_REPO"
mkdir -p "$CANON_ROOT/logs" "$CANON_OUTPUTS_ROOT"

echo "[slurm] node    = $(hostname)"
echo "[slurm] gpus    = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "[slurm] python  = $(python --version)"
echo "[slurm] CANON   = $CANON_ROOT"
echo "[slurm] repo    = $CANON_REPO"
echo "[slurm] data    = $CANON_DATA_ROOT"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# If a safetensors model is present, prefer it and disable any pytorch_model.bin
# to avoid transformers attempting to use torch.load (security guard requires
# newer torch). This is reversible: the script merely renames the file.
if [ -f "$CANON_ROOT/BioLinkBERT/model.safetensors" ] && [ -f "$CANON_ROOT/BioLinkBERT/pytorch_model.bin" ]; then
    echo "[slurm] safetensors present; renaming pytorch_model.bin -> pytorch_model.bin.disabled"
    mv -n "$CANON_ROOT/BioLinkBERT/pytorch_model.bin" "$CANON_ROOT/BioLinkBERT/pytorch_model.bin.disabled" || true
fi

# Auto-detect the most recent epoch checkpoint with full optimizer state and
# resume from it. The 4h wall fits ~3 epochs, so chained sbatch invocations
# will roll the model from epoch_03 -> epoch_06 -> epoch_08.
RESUME_ARGS=()
LATEST_CKPT=""
if [ -d "$CANON_OUTPUTS_ROOT/phase3/sapbert/checkpoints" ]; then
    LATEST_CKPT=$(ls -d "$CANON_OUTPUTS_ROOT"/phase3/sapbert/checkpoints/epoch_* 2>/dev/null | sort -V | tail -1)
fi
if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/optimizer.pt" ]; then
    echo "[slurm] resuming from $LATEST_CKPT"
    RESUME_ARGS=(--resume-from "$LATEST_CKPT")
elif [ -n "$LATEST_CKPT" ]; then
    echo "[slurm] found $LATEST_CKPT but no optimizer.pt -- starting fresh run"
fi

# Production training: 8 epochs, batch 128 (input batch 256 after pairing),
# lr 2e-5, full pair coverage (no --max-pairs cap).
# num-workers=0 is intentional: the IterableDataset is not worker-sharded, and
# tokenization is now pre-computed once in main() so collate is GPU-feeding
# tensor work, not Python tokenizer calls.
python scripts/sapbert_pretrain.py \
    "${RESUME_ARGS[@]}" \
    --epochs 8 \
    --batch-size 128 \
    --lr 2e-5 \
    --pairs-per-concept 10 \
    --mixed-precision yes \
    --num-workers 0

echo "[slurm] phase 3.1 complete; encoder at $CANON_OUTPUTS_ROOT/phase3/sapbert/encoder/"
