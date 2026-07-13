#!/bin/bash
#SBATCH --job-name=canon-phase4
#SBATCH --partition=general
#SBATCH --account=r00877
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/N/slate/alikh/canon/logs/phase4.%j.out
#SBATCH --error=/N/slate/alikh/canon/logs/phase4.%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge
module load python/3.11
source /N/slate/alikh/canon/.venv/bin/activate
export CANON_ROOT="/N/slate/alikh/canon"
export CANON_REPO="/N/u/alikh/BigRed200/CANON"
export DATA_ROOT="/N/slate/alikh/lingl675/data/Data"
export CANON_OUTPUTS_ROOT="$CANON_ROOT/outputs"
cd "$CANON_REPO"

python scripts/phase4_evaluate.py \
  --predictions "$CANON_OUTPUTS_ROOT/phase3/csp_predictions/test.jsonl" \
  --gold "$CANON_OUTPUTS_ROOT/phase2/splits/test.jsonl" \
  --assignment csp \
  --output "$CANON_OUTPUTS_ROOT/phase4/full_csp.json"
