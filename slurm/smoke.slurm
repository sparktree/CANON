#!/bin/bash
# CANON pre-flight smoke run — exercises every implemented phase end-to-end
# on a GPU node before submitting the production Phase 3.1 sbatch.
#
# Submit with:
#     sbatch slurm/smoke.slurm

#SBATCH --job-name=canon-smoke
#SBATCH --partition=gpu
#SBATCH --account=r00877
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/N/slate/alikh/canon/logs/smoke.%j.out
#SBATCH --error=/N/slate/alikh/canon/logs/smoke.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khurramali1997@gmail.com

set -euo pipefail

module purge
module load python/3.11
source /N/slate/alikh/canon/.venv/bin/activate

export CANON_ROOT="/N/slate/alikh/canon"
export CANON_REPO="/N/u/alikh/BigRed200/CANON"
export DATA_ROOT="/N/slate/alikh/lingl675/data/Data"
export CANON_DATA_ROOT="$DATA_ROOT"
export CANON_BIOLINKBERT="$CANON_ROOT/BioLinkBERT"
export CANON_OUTPUTS_ROOT="$CANON_ROOT/outputs"
export HF_HOME="$CANON_ROOT/.hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"

cd "$CANON_REPO"
mkdir -p "$CANON_ROOT/logs" "$CANON_OUTPUTS_ROOT"

echo "[smoke] node    = $(hostname)"
echo "[smoke] gpus    = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"
echo "[smoke] python  = $(python --version)"
echo "[smoke] CANON   = $CANON_ROOT"
echo "[smoke] repo    = $CANON_REPO"
echo "[smoke] data    = $CANON_DATA_ROOT"

# Run every implemented phase. Phase 3.1 runs in --smoke-test mode automatically
# (main.py wraps it). Phase 2.6 stays gated by CANON_DOWNLOAD_SILVER (unset),
# so it reuses the cached PubTator3 silver outputs already in the repo.
python main.py

echo "[smoke] all phases complete"
