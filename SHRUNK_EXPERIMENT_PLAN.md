# CANON Shrunk Experiment Plan

This plan is for a constrained local machine: Apple M4, 16 GB RAM, no usable
MPS backend in the current Python environment, and a roughly 5-hour wall-clock
budget. The goal is to produce defensible development-set results using the
existing data and pipeline artifacts without attempting the full GPU-scale
Phase 3 schedule.

## Constraints

- Do not regenerate Phase 1 or Phase 2 artifacts.
- Do not build the full SNOMED concept index locally.
- Treat local runs as pilot experiments, not final production training.
- Prefer completed metrics over larger runs that may not finish.
- Use full dev evaluation whenever feasible.

## Available Inputs

- Full Phase 2 train/dev/test split:
  - `outputs/phase2/splits/train.jsonl`
  - `outputs/phase2/splits/dev.jsonl`
  - `outputs/phase2/splits/test.jsonl`
- Phase 2 soft mapping lookup:
  - `outputs/phase2/soft_mapping_lookup.json`
- Latest SapBERT encoder:
  - `outputs/phase3/sapbert`
  - `train_state.json` reports `"epoch": 7`, treated here as the latest
    epoch-8-style checkpoint from the upstream pretraining run.
- MRCM and hierarchy artifacts:
  - `outputs/phase1/mrcm_constraints.json`
  - `outputs/phase1/snomed_ancestors.pkl`

## Reduced Data Policy

Create two deterministic train subsets:

- `train_gold.jsonl`: all available gold training documents.
  - BioRED: 400 docs
  - BC5CDR: 500 docs
  - Total: 900 docs
- `train_fast.jsonl`: gold plus bounded augmentation.
  - BioRED: 400 docs
  - BC5CDR: 500 docs
  - PubTator3 silver: 1,500 docs
  - SNOMED synthetic: 3,000 docs
  - Total: 5,400 docs

Command:

```bash
python scripts/make_fast_splits.py --silver 1500 --synthetic 3000
```

Expected outputs:

- `outputs/phase2/splits/train_gold.jsonl`
- `outputs/phase2/splits/train_fast.jsonl`
- `outputs/phase2/splits/fast_split_summary.json`

## Reduced Concept Index

Build a candidate-only concept index from Phase 2 soft mappings instead of
indexing all active SNOMED concepts. This keeps normalization feasible locally
while preserving the candidates the dataset actually trains against.

Command:

```bash
python scripts/build_concept_index.py \
  --from-soft-lookup \
  --encoder-dir outputs/phase3/sapbert \
  --output-dir outputs/phase3/concept_index_sapbert_epoch8 \
  --batch-size 64 \
  --device cpu
```

Expected outputs:

- `outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json`
- `outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors`
- `outputs/phase3/concept_index_sapbert_epoch8/build_summary.json`

Current epoch-8 encoder run produced:

- Concepts: 30,258
- Embedding dim: 768
- Build time: 76.92 seconds

## Stage 1 Experiments

### NER Head

Use `train_fast` because silver and synthetic documents still provide useful
entity span supervision.

```bash
python scripts/train_stage1.py \
  --head ner \
  --epochs 3 \
  --batch-size 2 \
  --max-length 384 \
  --max-docs 2000 \
  --train-path outputs/phase2/splits/train_fast.jsonl \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --device cpu \
  --encoder-dir outputs/phase3/sapbert \
  --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \
  --output-dir outputs/phase3/stage1_epoch8
```

Current epoch-8 encoder result:

- Best dev NER F1: 0.7820

### Relation Head

Use `train_gold` because relation labels are most reliable in gold data.

```bash
python scripts/train_stage1.py \
  --head rel \
  --epochs 3 \
  --batch-size 2 \
  --max-length 384 \
  --max-docs 900 \
  --train-path outputs/phase2/splits/train_gold.jsonl \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --device cpu \
  --encoder-dir outputs/phase3/sapbert \
  --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \
  --output-dir outputs/phase3/stage1_epoch8
```

Current epoch-8 encoder result:

- Best dev relation macro F1: 0.1253

### Normalization Head

Use `train_fast` because synthetic and silver examples improve concept
coverage. The candidate-only concept index is required before this run.

```bash
python scripts/train_stage1.py \
  --head norm \
  --epochs 2 \
  --batch-size 2 \
  --max-length 384 \
  --max-docs 1000 \
  --train-path outputs/phase2/splits/train_fast.jsonl \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --device cpu \
  --encoder-dir outputs/phase3/sapbert \
  --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \
  --output-dir outputs/phase3/stage1_epoch8
```

Current epoch-8 encoder result:

- Best dev norm top-1: 0.0774
- Final epoch norm ancestor: 0.1667

## Stage 2 Joint Experiment

Warm-start from the three epoch-8 Stage 1 heads and run a short joint pass.

```bash
python scripts/train_stage2.py \
  --epochs 2 \
  --batch-size 2 \
  --max-length 384 \
  --max-docs 1000 \
  --train-path outputs/phase2/splits/train_fast.jsonl \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --device cpu \
  --encoder-dir outputs/phase3/sapbert \
  --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \
  --stage1-dir outputs/phase3/stage1_epoch8 \
  --output-dir outputs/phase3/stage2_epoch8
```

Current epoch-8 encoder result:

- Aggregate: 0.3966
- NER F1: 0.7469
- Norm top-1: 0.1969
- Norm ancestor: 0.2704
- Relation macro F1: 0.1294
- Epoch times: 370.01s, 360.11s

Compared with the earlier local CPU run, the epoch-8 encoder improves
normalization substantially after joint training but lowers NER enough that
the aggregate score is lower:

| Run | Aggregate | NER F1 | Norm Top-1 | Norm Ancestor | Rel Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier local CPU | 0.4027 | 0.7876 | 0.1605 | 0.1900 | 0.1316 |
| Epoch-8 CPU | 0.3966 | 0.7469 | 0.1969 | 0.2704 | 0.1294 |

## Apple Metal / MPS Experiment

The MPS numbers below were run before the epoch-8 encoder rerun, using
`outputs/phase3/stage1_local` as the warm start. They are retained as hardware
evidence, not as the current best model recommendation.

The default sandboxed Python process cannot see MPS on this machine:

- `torch.backends.mps.is_built()`: true
- `torch.backends.mps.is_available()`: false
- `torch.mps.device_count()`: 0

Running outside the sandbox with the separate Python 3.11 environment at
`/private/tmp/canon-mps-venv` exposes the Metal device:

- `torch.backends.mps.is_available()`: true
- `torch.mps.device_count()`: 1
- device: `mps:0`

Use CPU fallback because some PyTorch operations in the model stack may not
have MPS kernels:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 /private/tmp/canon-mps-venv/bin/python scripts/train_stage2.py \
  --epochs 2 \
  --batch-size 2 \
  --max-length 384 \
  --max-docs 1000 \
  --train-path outputs/phase2/splits/train_fast.jsonl \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --device mps \
  --stage1-dir outputs/phase3/stage1_local \
  --output-dir outputs/phase3/stage2_mps
```

Current MPS result:

- Aggregate: 0.4024
- NER F1: 0.7907
- Norm top-1: 0.1556
- Norm ancestor: 0.1845
- Relation macro F1: 0.1316
- Epoch times: 554.33s, 541.36s

For this workload, MPS was slower than CPU because fallback and transfer
overhead dominate. The CPU Stage 2 epoch times were 392.76s and 375.72s.

## Optional Stage 3

Skip Stage 3 locally unless Stage 2 finishes with substantial time remaining.
The CSP pass adds complexity and is less valuable than obtaining stable Stage 1
and Stage 2 metrics under this hardware budget.

If running Stage 3 anyway, keep it small:

```bash
python scripts/train_stage3.py \
  --epochs 1 \
  --batch-size 2 \
  --max-length 384 \
  --max-docs 300 \
  --train-path outputs/phase2/splits/train_fast.jsonl \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --device cpu \
  --stage2-dir outputs/phase3/stage2_epoch8/best \
  --output-dir outputs/phase3/stage3_epoch8
```

## Reporting

Report these results as constrained local experiments:

- The full data pipeline and production Slurm scripts exist.
- The sandboxed local environment could not see MPS. Running outside the
  sandbox exposed MPS, but it was slower than CPU for this workload.
- The full SNOMED index was replaced with a candidate-only index derived from
  Phase 2 soft mappings.
- Dev metrics were measured on the full dev split unless otherwise stated.
- Production-scale conclusions should come from the GPU Slurm path.

## Decision Rules

- If the goal is best aggregate from the shrunk local experiments, use
  `outputs/phase3/stage2_local/best`.
- If the goal is stronger normalization behavior, use
  `outputs/phase3/stage2_epoch8/best`.
- If the goal is per-task ablation on the latest encoder, report
  `outputs/phase3/stage1_epoch8`.
- If the goal is production performance, submit the existing Slurm jobs rather
  than extending CPU-local training.
- If time is under 2 hours, skip Stage 2 and report Stage 1 only.
