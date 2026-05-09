#!/bin/bash
# CANON Phase 3.2 — encode every active SNOMED concept's preferred term with
# the SapBERT-pretrained encoder and cache the (~370K, 768) fp16 matrix used
# by the concept-normalization head.
#
# One-shot prerequisite for Stage 1/2/3 training. Cheap (single A100, ~30 min)
# but rerun whenever the SapBERT encoder is updated.
#
#     sbatch slurm/build_concept_index.slurm
#     sbatch -A r00998 slurm/build_concept_index.slurm   # alternate account
#     sbatch -A c02114 slurm/build_concept_index.slurm

#SBATCH --job-name=canon-cidx
#SBATCH --partition=gpu
#SBATCH --account=r00877
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/N/slate/alikh/canon/logs/cidx.%j.out
#SBATCH --error=/N/slate/alikh/canon/logs/cidx.%j.err
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

echo "[cidx] node    = $(hostname)"
echo "[cidx] gpus    = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"

python scripts/build_concept_index.py \
    --batch-size 256 \
    --max-length 64

echo "[cidx] done -> $CANON_OUTPUTS_ROOT/phase3/concept_index/"

# Race pattern: cidx is submitted on r00877 / r00998 / c02114 simultaneously
# to dodge fairshare backlogs. The first one to finish cancels the others.
OTHER_JIDS=$(squeue -u "$USER" -h -o '%i %j' \
    | awk -v me="$SLURM_JOB_ID" '$2=="canon-cidx" && $1!=me {print $1}' \
    | tr '\n' ' ')
if [ -n "$OTHER_JIDS" ]; then
    echo "[cidx] cancelling sibling cidx jobs: $OTHER_JIDS"
    scancel $OTHER_JIDS || true
fi
