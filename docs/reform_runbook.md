# CANON Reform Runbook

The canonical Phase 2 source is one JSON-LD file per document. JSONL split
files are derived training caches; do not edit them as source data. The Phase 3
production dependency chain is submitted by `scripts/queue_phase3.sh`.

## Required comparisons

Use separate output and concept-index directories for every run. A no-SapBERT
run must first build its concept matrix with the same base encoder used for
training:

```bash
python scripts/build_concept_index.py \
  --encoder-dir "$CANON_BIOLINKBERT" \
  --output-dir outputs/phase3/ablations/no_sapbert/concept_index
python scripts/train_stage1.py --head norm \
  --encoder-dir "$CANON_BIOLINKBERT" \
  --concept-index-dir outputs/phase3/ablations/no_sapbert/concept_index \
  --output-dir outputs/phase3/ablations/no_sapbert/stage1
```

Run all three Stage 1 heads with the same ablation flags before Stage 2. The
minimum component removals map to these switches:

| Ablation | Training switch |
|---|---|
| no SapBERT | `--encoder-dir` plus matching `--concept-index-dir` |
| no SNOMED synthetic | `--exclude-corpus synthetic` |
| no PubTator silver | `--exclude-corpus silver` |
| hard labels | `--hard-labels` |
| no confidence weights | `--no-confidence-weighting` |
| independent heads | `csp_solver.py --independent-stage1-dir DIR` |

The joint soft-consistency configuration adds `--soft-consistency` to Stage 2.
Every training summary records encoder, index, exclusions, label mode, and
weight mode so mislabeled ablation artifacts can be detected.

Compose all three separately fine-tuned encoders and heads into one prediction
artifact with:

```bash
python scripts/csp_solver.py --split test \
  --independent-stage1-dir outputs/phase3/stage1 \
  --output-dir outputs/phase3/independent_predictions
```

The resulting file contains both `neural` and `csp` assignments. Evaluate its
`neural` assignment for the independent no-CSP baseline and its `csp`
assignment for ablation (f).

Phase 4 consumes prediction JSONL files under one evaluation contract:

```bash
python scripts/run_phase4.py \
  --config independent=PATH:neural \
  --config joint=PATH:neural \
  --config joint_soft=PATH:neural \
  --config csp=PATH:csp
```

External PubTator/AIONER, TaggerOne, and BioREx baselines must be exported to
the same prediction assignment schema. Their model downloads and executions
remain external production prerequisites; CANON does not silently substitute
random or gold inputs when they are absent.
