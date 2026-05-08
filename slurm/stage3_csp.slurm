#!/bin/bash
# CANON Phase 3.5 -- CSP solver inference pass over dev/test splits (CPU).

#SBATCH --job-name=canon-csp
#SBATCH --partition=general
#SBATCH --account=r00877
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/N/slate/alikh/canon/logs/csp.%j.out
#SBATCH --error=/N/slate/alikh/canon/logs/csp.%j.err
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
echo "[csp] node=$(hostname) cpus=$(nproc)"

python scripts/csp_solver.py --split dev --top-k-types 2 --top-k-concepts 10 --top-k-relations 3 --timeout-ms 5000
python scripts/csp_solver.py --split test --top-k-types 2 --top-k-concepts 10 --top-k-relations 3 --timeout-ms 5000

echo "[csp] done -> $CANON_OUTPUTS_ROOT/phase3/csp_predictions/"
