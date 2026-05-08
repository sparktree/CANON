#!/bin/bash
# CANON Phase 3.3 -- Stage 1 relation extraction head training.

#SBATCH --job-name=canon-s1rel
#SBATCH --partition=gpu
#SBATCH --account=r00877
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/N/slate/alikh/canon/logs/s1rel.%j.out
#SBATCH --error=/N/slate/alikh/canon/logs/s1rel.%j.err
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
echo "[s1rel] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader || true)"

python scripts/train_stage1.py --head rel --epochs 10 --batch-size 8

echo "[s1rel] done -> $CANON_OUTPUTS_ROOT/phase3/stage1/rel/"
