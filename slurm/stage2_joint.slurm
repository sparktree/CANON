#!/bin/bash
# CANON Phase 3.4 -- Stage 2 joint multi-task training.

#SBATCH --job-name=canon-s2joint
#SBATCH --partition=gpu
#SBATCH --account=r00877
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/N/slate/alikh/canon/logs/s2joint.%j.out
#SBATCH --error=/N/slate/alikh/canon/logs/s2joint.%j.err
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
mkdir -p "$CANON_ROOT/logs"
echo "[s2joint] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader || true)"

python scripts/train_stage2.py --epochs 15 --batch-size 8

echo "[s2joint] done -> $CANON_OUTPUTS_ROOT/phase3/stage2/"
