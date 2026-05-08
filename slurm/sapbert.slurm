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
#SBATCH --account=YOUR_RT_ACCOUNT          # replace with your IU research account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1                  # one H100 is plenty for SapBERT-base
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sapbert.%j.out
#SBATCH --error=logs/sapbert.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_USERNAME@iu.edu   # replace

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load python/gpu/3.11                # IU's CUDA-aware Python
# If you maintain a venv on Slate, activate it here:
#   source $SLATE/canon-venv/bin/activate
# Otherwise, use a conda env:
#   module load anaconda
#   conda activate canon

# ---------------------------------------------------------------------------
# Paths -- everything on Slate to avoid home-quota pressure
# ---------------------------------------------------------------------------
export CANON_ROOT="$SLATE/canon"
export CANON_DATA_ROOT="$CANON_ROOT/Data"
export CANON_BIOLINKBERT="$CANON_ROOT/BioLinkBERT"
export CANON_OUTPUTS_ROOT="$CANON_ROOT/CANON/outputs"

# Optional: HuggingFace cache on Slate so models/tokenizers are reused across jobs
export HF_HOME="$CANON_ROOT/.hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"

cd "$CANON_ROOT/CANON"
mkdir -p logs

echo "[slurm] node    = $(hostname)"
echo "[slurm] gpus    = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "[slurm] python  = $(python --version)"
echo "[slurm] CANON   = $CANON_ROOT"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# Production training: 8 epochs, batch 128 (input batch 256 after pairing),
# lr 2e-5, full pair coverage (no --max-pairs cap).
python scripts/sapbert_pretrain.py \
    --epochs 8 \
    --batch-size 128 \
    --lr 2e-5 \
    --pairs-per-concept 10 \
    --mixed-precision yes \
    --num-workers 4

echo "[slurm] phase 3.1 complete; encoder at $CANON_OUTPUTS_ROOT/phase3/sapbert/encoder/"
