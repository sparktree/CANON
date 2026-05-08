#!/usr/bin/env bash
# CANON Phase 3 dependency chain submitter.
#
# Submits sapbert -> {cidx, stage1 ner/norm/rel} -> {stage2, csp} -> stage3
# with afterok dependencies. Accounts rotate r00998 -> r00877 -> c02114 so
# the three independent stage-1 heads land on three distinct fairshare slots.
#
# Usage:
#   bash scripts/queue_phase3.sh                 # straight production chain
#   bash scripts/queue_phase3.sh --smoke         # smoke job first; chain blocks on its completion (afterany)
#
# Run from the repo root.

set -euo pipefail

SLURM_DIR="slurm"
SMOKE_FIRST=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FIRST=1
fi

if [[ ! -d "$SLURM_DIR" ]]; then
    echo "error: run this from the CANON repo root (cwd=$PWD)" >&2
    exit 1
fi

declare -a CHAIN_LOG=()

submit() {
    # submit <account> <dep-spec or empty> <slurm-script>
    local account="$1" dep="$2" script="$3"
    local args=(--parsable -A "$account")
    if [[ -n "$dep" ]]; then
        args+=(--dependency="$dep")
    fi
    args+=("$SLURM_DIR/$script")
    local jid
    jid=$(sbatch "${args[@]}")
    echo "$jid"
}

PRECHAIN_DEP=""
if [[ "$SMOKE_FIRST" == "1" ]]; then
    SMOKE_JID=$(submit r00877 "" smoke.slurm)
    CHAIN_LOG+=("smoke    = $SMOKE_JID  (r00877)")
    PRECHAIN_DEP="afterany:$SMOKE_JID"
fi

# Order: sapbert, cidx, s1ner, s1norm, s1rel, s2joint, csp, s3ft
# Accounts rotate r00998 -> r00877 -> c02114.

SAPBERT_JID=$(submit r00998 "$PRECHAIN_DEP" sapbert.slurm)
CHAIN_LOG+=("sapbert  = $SAPBERT_JID  (r00998)")

CIDX_JID=$(submit r00877 "afterok:$SAPBERT_JID" build_concept_index.slurm)
CHAIN_LOG+=("cidx     = $CIDX_JID  (r00877)  [afterok:$SAPBERT_JID]")

S1N_JID=$(submit c02114 "afterok:$SAPBERT_JID" stage1_ner.slurm)
CHAIN_LOG+=("s1ner    = $S1N_JID  (c02114)  [afterok:$SAPBERT_JID]")

S1M_JID=$(submit r00998 "afterok:$SAPBERT_JID" stage1_norm.slurm)
CHAIN_LOG+=("s1norm   = $S1M_JID  (r00998)  [afterok:$SAPBERT_JID]")

S1R_JID=$(submit r00877 "afterok:$SAPBERT_JID" stage1_rel.slurm)
CHAIN_LOG+=("s1rel    = $S1R_JID  (r00877)  [afterok:$SAPBERT_JID]")

S2_JID=$(submit c02114 "afterok:$S1N_JID:$S1M_JID:$S1R_JID" stage2_joint.slurm)
CHAIN_LOG+=("s2joint  = $S2_JID  (c02114)  [afterok:$S1N_JID:$S1M_JID:$S1R_JID]")

CSP_JID=$(submit r00998 "afterok:$CIDX_JID" stage3_csp.slurm)
CHAIN_LOG+=("csp      = $CSP_JID  (r00998)  [afterok:$CIDX_JID]")

S3_JID=$(submit r00877 "afterok:$S2_JID" stage3_finetune.slurm)
CHAIN_LOG+=("s3ft     = $S3_JID  (r00877)  [afterok:$S2_JID]")

echo
echo "Phase 3 chain submitted:"
printf '  %s\n' "${CHAIN_LOG[@]}"
echo
echo "Watch with: squeue -u \$USER -o '%.10i %.10a %.13j %.2t %.10M %.20E'"
