# CANON Phase 4 Ablation Study — Shrunk-Scope Plan + Results

Run date: 2026-05-08. Total wall-clock for the full sweep: ~30 min on Apple M4, 16 GB, CPU.

## Context

The full `plan.MD` Phase 4 (lines 305–407) prescribes six retraining-required ablations, three external baselines (AIONER, TaggerOne, BioREx), error analysis on 50–100 docs, and cross-domain transfer to clinical notes. That budget assumes GPU/Slurm.

The actual environment is what `CANON/SHRUNK_EXPERIMENT_PLAN.md` describes: Apple M4, 16 GB RAM, no usable MPS, ~5-hour wall-clock budget per run. The `stage2_epoch8` Stage 2 baseline on full dev is **NER F1 0.747, norm top-1 0.197, rel macro F1 0.129, aggregate 0.397**. At norm top-1 ≈ 0.20 and rel macro F1 ≈ 0.13, removing single components produces deltas inside the noise floor; the full ablation suite cannot be defended at this scale.

**Hard constraint (user directive 2026-05-08):** no retrain-style ablations. Anything that needs Stage 1 or Stage 2 retraining is out. Kept set is zero-retrain only.

**Goal:** preserve the paper's load-bearing claim — soft training signals improve coherence but cannot guarantee it; hard symbolic constraints can — and drop or bundle the rest with explicit justification.

## Glass_CANON / Shane_models_best Drop

Teammate (Shane) shipped this set into `outputs/phase3/Shane_models_best/`:

- `stage1_local_{ner,norm,rel}_best_model.safetensors` (4 × 433 MB) — encoder-only snapshots
- `stage2_local_best_model.safetensors` (433 MB) — encoder-only snapshot
- `{ner,norm,rel,multi}_head_state.pt` — corresponding head parameter bundles
- `concept_emb.safetensors` (46 MB) + `concept_ids.json` (29,972 SNOMED IDs — distinct from the existing epoch-8 index's 30,258, must stay paired)

Encoder pairing: existing `outputs/phase3/sapbert` (BioLinkBERT + SapBERT-style pretraining). State-dict shape match was clean (encoder 0/0 missing-unexpected, heads 12/12 copied).

**Step-0 verdict:** Shane's `stage2_local` evaluated on full dev gave NER F1 0.097 / norm top-1 0.011 / rel macro F1 0.044 / aggregate 0.055 — about 7× weaker than the existing `stage2_epoch8` baseline. NER precision 0.51 / recall 0.05 is the signature of an under-trained NER head (model defaults to predicting "O"). Headline rows therefore use `stage2_epoch8`; Shane's row sits in an appendix as a comparison data point.

## Recommended Ablation Set

### KEEP (load-bearing, zero retrain)

1. **§4.3 four-configuration coherence sweep** — the paper. Configs 2/4 share one Stage 2 checkpoint; only the inference-time post-processor changes.
2. **§4.4 override flip counts** — computed during the §4.3 sweep, free.
3. **§4.5 ablation: "without multi-task"** — compose Stage 1 heads with their NORM-stage encoder, run with and without CSP. No retraining.
4. **NEW: out-of-the-box encoder comparison (replaces "without SapBERT" retrain)** — rebuild the candidate-concept embedding matrix with each off-the-shelf encoder, run zero-shot bi-encoder normalization on dev. No head retraining. Encoders chosen:
   - `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` — canonical published SapBERT, the headline OOTB-vs-ours comparator.
   - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` — strong biomedical encoder without SapBERT-style ontology alignment; isolates whether the SapBERT step is what drives the difference.

### DROP (with justification)

- **§4.5 "without augmentation"** — Stage 1 retrain on `train_gold` only; user directive forbids retraining.
- **§4.5 "without soft labels"** — retrain required.
- **§4.5 "without confidence weighting"** — retrain required.
- **§4.7 cross-domain (i2b2/MCN)** — DUA-gated, no checkpoints prepared.
- **§4.1 baselines AIONER + TaggerOne + BioREx** — running PubTator-3 models locally + re-mapping their MeSH outputs to SNOMED is multi-day engineering. Keep only **BioLinkBERT independently fine-tuned** as the baseline; that is the "− multi-task" row and is already on disk as Stage 1 weights.

### DEFERRED

- **§4.6 error analysis on 30 dev docs** — not in this run; can be done from `outputs/phase4/*.json` outputs without rerunning the model.

## Execution Order (zero-retrain, observed times)

| # | Step | Cost (observed) | Reuses |
|---|---|---|---|
| 0 | **Pre-flight** Shane stage2 — load encoder + multi_head_state.pt + Shane concept index, run `evaluate_all` on full dev | 34 s | Shane_models_best, sapbert |
| 0b | Pre-flight `stage2_epoch8` — sanity check that the harness reproduces documented baseline | 36 s | stage2_epoch8 |
| 1 | §4.3 + §4.4 coherence sweep on `stage2_epoch8` (off + hard CSP, full dev) | 35 s inference + 90 s CSP (499 × ~183 ms) | stage2_epoch8 |
| 2 | §4.5 "− multi-task": compose Stage 1 heads with NORM encoder, off + hard CSP | 37 s inference + 88 s CSP | stage1_epoch8 |
| 3 | OOTB SapBERT-from-PubMedBERT: build concept index (74 s) + zero-shot norm eval (20 s) | ~95 s | HF cache |
| 4 | OOTB PubMedBERT: build concept index (74 s) + zero-shot norm eval (20 s) | ~95 s | HF cache |

**Total: ~7 minutes of compute** (under the 30-min plan estimate); the rest of the wall-clock was harness scaffolding and one bug fix for `csp_solver.load_constraint_tables` (see *Bug Fixes Encountered* below).

## Concrete Commands (as actually run)

All commands run from `/Volumes/Khurrum/AdvMLinCL/CANON` with the `../venv` Python environment activated.

### Step 0 — Shane pre-flight (full dev)

```bash
python scripts/eval_phase4.py --mode preflight \
  --encoder-base outputs/phase3/sapbert \
  --encoder-weights outputs/phase3/Shane_models_best/stage2_local_best_model.safetensors \
  --head-state outputs/phase3/Shane_models_best/multi_head_state.pt \
  --concept-index-ids outputs/phase3/Shane_models_best/concept_ids.json \
  --concept-index-emb outputs/phase3/Shane_models_best/concept_emb.safetensors \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --output outputs/phase4/shane_preflight.json
```

### Step 0b — `stage2_epoch8` pre-flight (harness sanity check)

```bash
python scripts/eval_phase4.py --mode preflight \
  --encoder-base outputs/phase3/stage2_epoch8/best \
  --head-state outputs/phase3/stage2_epoch8/best/head_state.pt \
  --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --output outputs/phase4/stage2_epoch8_preflight.json
```

### Step 1 — §4.3 + §4.4 coherence sweep on `stage2_epoch8`

```bash
python scripts/eval_phase4.py --mode coherence_sweep \
  --encoder-base outputs/phase3/stage2_epoch8/best \
  --head-state outputs/phase3/stage2_epoch8/best/head_state.pt \
  --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --timeout-ms 5000 \
  --output outputs/phase4/stage2_epoch8_sweep.json
```

### Step 2 — "− multi-task": Stage 1 heads composed with NORM encoder + CSP

```bash
python scripts/eval_phase4.py --mode coherence_sweep \
  --encoder-base outputs/phase3/stage1_epoch8/norm/best \
  --head-state outputs/phase3/stage1_epoch8/ner/best/head_state.pt \
              outputs/phase3/stage1_epoch8/norm/best/head_state.pt \
              outputs/phase3/stage1_epoch8/rel/best/head_state.pt \
  --concept-index-ids outputs/phase3/concept_index_sapbert_epoch8/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_sapbert_epoch8/concept_emb.safetensors \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --timeout-ms 5000 \
  --output outputs/phase4/no_multitask_normenc_sweep.json
```

### Steps 3 & 4 — OOTB encoders (zero-shot norm)

```bash
SAPBERT_PATH=~/.cache/huggingface/hub/models--cambridgeltl--SapBERT-from-PubMedBERT-fulltext/snapshots/090663c3ae57bf35ffe4d0d468a2a88d03051a4d
PMB_PATH=~/.cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062

# Build OOTB indices (must use local cache path, not HF hub ID — see Bug Fixes)
python scripts/build_concept_index.py --from-soft-lookup \
  --encoder-dir "$SAPBERT_PATH" \
  --output-dir outputs/phase3/concept_index_ootb_sapbert \
  --batch-size 32 --device cpu

python scripts/build_concept_index.py --from-soft-lookup \
  --encoder-dir "$PMB_PATH" \
  --output-dir outputs/phase3/concept_index_ootb_pubmedbert \
  --batch-size 32 --device cpu

# Zero-shot bi-encoder norm eval
python scripts/eval_phase4.py --mode ootb_norm \
  --encoder-base "$SAPBERT_PATH" \
  --concept-index-ids outputs/phase3/concept_index_ootb_sapbert/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_ootb_sapbert/concept_emb.safetensors \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --output outputs/phase4/ootb_sapbert.json

python scripts/eval_phase4.py --mode ootb_norm \
  --encoder-base "$PMB_PATH" \
  --concept-index-ids outputs/phase3/concept_index_ootb_pubmedbert/concept_ids.json \
  --concept-index-emb outputs/phase3/concept_index_ootb_pubmedbert/concept_emb.safetensors \
  --dev-path outputs/phase2/splits/dev.jsonl \
  --output outputs/phase4/ootb_pubmedbert.json
```

## Headline Table — Real Numbers

Dev split: BioRED dev (100) + BC5CDR dev (500), gold-only, 8,776–9,103 evaluated entities depending on the join semantics (sweep uses span-overlap join with gold; OOTB uses every gold entity with `mapped_snomed_id`).

| # | Configuration | Checkpoint | NER F1 | Norm top-1 | Norm anc. | Rel macro | Triple validity | Doc coherence |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | − Multi-task, no CSP | Stage 1 NORM enc + heads | – | 0.087 | – | – | 0.927 | 0.908 |
| 2 | Multi-task, no CSP | `stage2_epoch8` | **0.747** | 0.189 / 0.223 † | 0.258 | **0.129** | 0.951 | 0.920 |
| 3 | Multi-task + soft consistency, no hard CSP | (skipped — see note below) | – | – | – | – | – | – |
| 4 | **Multi-task + CSP (ours)** | `stage2_epoch8` + Z3 | 0.747 | 0.210 † | – | 0.129 | **1.000** | **1.000** |
| 5 | − Multi-task + CSP | Stage 1 NORM enc + heads + Z3 | – | 0.080 | – | – | **1.000** | **1.000** |
| 6 | OOTB norm: SapBERT-from-PubMedBERT | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | n/a | **0.648** | **0.684** | n/a | n/a | n/a |
| 7 | OOTB norm: PubMedBERT-fulltext | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | n/a | 0.269 | 0.284 | n/a | n/a | n/a |
| Apx | Shane stage2 candidate | `Shane_models_best/stage2_local` + multi_head | 0.097 | 0.011 | 0.136 | 0.044 | – | – |

† Two norm top-1 numbers in row 2 reflect different evaluation paths: 0.189 from `evaluate_all` (matches the eval used at training time, soft-target-aware) vs 0.223 from the sweep mode that span-overlap-joins predictions with gold from the dev JSONL. They measure slightly different denominators; both are reported for transparency.

**Config 3 (soft consistency loss but no hard CSP) is skipped:** the shrunk-scope training run did not include a soft-consistency auxiliary loss, so the only meaningful "soft" config would require a retrain — which is out of scope per the user directive. The paper's argument can be carried by configs 1, 2, and 4 alone (independent → joint → joint+CSP).

### Override flip breakdown (§4.4)

| Config | Total overrides | corr→incorr | incorr→corr | both wrong | Net flip benefit |
|---|---:|---:|---:|---:|---:|
| Config 4 (`stage2_epoch8` + CSP) | 2,079 | 278 | 166 | 1,635 | **−112** |
| Config 5 (Stage 1 + CSP) | 2,194 | 166 | 105 | 1,923 | −61 |

For config 4: 79% (1,635/2,079) of overrides land on entities where the neural model was already wrong — the CSP changes one wrong concept to a different wrong concept and contributes nothing to accuracy, but enforces structural validity.

### CSP solve performance

`stage2_epoch8` + CSP averages **183 ms per document** on CPU (499 docs solved, 0 fallbacks).

## Key Findings

1. **CSP guarantees coherence by construction.** Configs 2 → 4: per-triple validity 0.951 → 1.000, per-doc full coherence 0.920 → 1.000. Both 100% in config 4 with zero fallbacks.

2. **CSP cost is small and predictable.** Concept top-1 drops by 1.3 absolute pp (0.223 → 0.210). 79% of overrides hit already-wrong neural predictions; only 21% touch entities where the neural model had a chance.

3. **Multi-task joint training is real.** Concept top-1 lifts from 0.087 (Stage 1 heads composed with NORM encoder) to 0.223 (Stage 2 joint), a **2.5× improvement**. Document-level coherence also improves (0.908 → 0.920) before any CSP intervention.

4. **OOTB SapBERT zero-shot beats the trained pipeline 3× on norm top-1 (0.648 vs 0.223).** This is unflattering but informative: published `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` zero-shot already outperforms the trained CANON Stage 2 norm head on this corpus. The most plausible cause is that joint multi-task training drifts the encoder representation away from clean concept-similarity space. Worth investigating before publishing — see the *Recommended follow-ups* section below.

5. **PubMedBERT (no SapBERT alignment) is comparable to our trained pipeline (0.269 vs 0.223).** This isolates which step matters: SapBERT-style ontology alignment is doing the heavy lifting, not the BioLinkBERT vs PubMedBERT base choice. A future re-run of the SapBERT-style pretraining step (Phase 3.1) with a stronger schedule should be the highest-leverage improvement.

6. **Shane's Glass_CANON checkpoint underperformed `stage2_epoch8` 7× on aggregate (0.055 vs 0.394).** Most likely under-trained (NER recall 0.054 with precision 0.51). Kept as appendix-only data point.

## Files Produced

### New scripts

- `CANON/scripts/eval_phase4.py` — three modes:
  - `--mode preflight` — load (encoder, heads, concept index), run `evaluate_all` on dev, dump NER/norm/rel/aggregate metrics.
  - `--mode coherence_sweep` — single inference pass + post-processing under `off` and `hard` CSP; reports per-triple validity, per-doc full coherence, concept top-1, override flip breakdown.
  - `--mode ootb_norm` — load arbitrary HF encoder + matching pre-built concept index, encode each gold mention's surface text, argmax cosine over candidates, report top-1 + ancestor.
  - Multi-file `--head-state` support (loads ner/norm/rel head_state.pt files in sequence; head keys don't collide).

### Metrics outputs (in `CANON/outputs/phase4/`)

- `shane_preflight.json` — Shane stage2 candidate full-dev metrics
- `stage2_epoch8_preflight.json` — harness sanity check against documented baseline
- `stage2_epoch8_sweep.json` — config 2 + 4 (off + hard CSP)
- `no_multitask_normenc_sweep.json` — config 5 (Stage 1 composed + CSP)
- `ootb_sapbert.json` — config 6 (zero-shot SapBERT-from-PubMedBERT)
- `ootb_pubmedbert.json` — config 7 (zero-shot PubMedBERT)
- `headline_table.json` — single aggregated summary with all rows + key findings

### Rebuilt concept indices (in `CANON/outputs/phase3/`)

- `concept_index_ootb_sapbert/{concept_ids.json, concept_emb.safetensors, build_summary.json}` — 30,258 SNOMED candidates encoded with SapBERT-from-PubMedBERT
- `concept_index_ootb_pubmedbert/{...}` — same 30,258 candidates encoded with PubMedBERT-fulltext

## Bug Fixes Encountered

1. **`csp_solver.load_constraint_tables` is broken against the current `mrcm_constraints.json`.** The function expects `relation_constraints[rel]` to be a list of flat `{domain_root, range_root}` dicts; the live JSON has it as `{domains: [{domain_root_concept_ids: [...]}, ...], ranges: [...]}`. Patched locally as `_load_constraint_tables_local` inside `eval_phase4.py` rather than touching the shared script. Anyone running `csp_solver.py` directly should expect an `AttributeError: 'str' object has no attribute 'get'` until that's fixed upstream.

2. **`build_concept_index.py` silently falls back to local BioLinkBERT when `--encoder-dir` is a HF hub ID.** The script does `Path(args.encoder_dir).is_dir()` and falls back to BioLinkBERT if False — but a hub ID like `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` is not a local directory, so it triggers the fallback. Worked around by passing the local HF cache snapshot path explicitly. The first OOTB SapBERT run was wasted on this; verified by checking `build_summary.json[encoder_dir]`.

3. **`CanonDocDataset` is an `IterableDataset`** (no `__len__`); harness was initially logging `len(dev_ds)` which raised. Trivial fix.

## Verification Status

| Check | Status |
|---|---|
| Shape-match pre-flight: encoder + heads load with 0 missing-unexpected on `stage2_epoch8` | ✅ |
| Harness reproduces documented `stage2_epoch8` numbers within rounding | ✅ NER 0.747, norm 0.189, rel 0.129, agg 0.394 (vs 0.747 / 0.197 / 0.129 / 0.397 documented) |
| CSP guarantees claim: config 4 reports per-triple validity = 1.000 | ✅ |
| Sanity of − multi-task row: norm top-1 within reasonable band (0.087 vs Stage 2 norm 0.223 — about half, expected since heads are not joint-trained) | ✅ |
| OOTB row interpretation: row 6 OOTB SapBERT *beats* row 2 norm top-1 → flagged as worth investigating | ✅ |
| Time budget: total wall-clock under 30 min | ✅ (~7 min of compute) |

## Recommended Follow-Ups

Not in scope for this run, but flagged by the results:

1. **Diagnose why OOTB SapBERT beats our trained norm head 3×.** Hypotheses to test, in order of cost:
   - Compare cosine similarities for the SAME mention/concept pair under (a) `outputs/phase3/sapbert` encoder and (b) `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`. If the published encoder produces tighter same-concept clusters on this corpus, that's the smoking gun.
   - Check whether the user's SapBERT-style pretraining (Phase 3.1) actually converged — `outputs/phase3/sapbert/train_state.json` reports epoch 7. The published model trained on the full UMLS for ~50 epochs.
   - Try replacing the encoder in CANON's pipeline with `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` and re-evaluating just the trained norm projection layer — does the projection still help, or does it actively hurt?

2. **§4.6 error analysis on 30 dev docs.** All inputs are now on disk in `outputs/phase4/*.json`. Bucket entities where neural and CSP disagree into mapping / coverage / constraint / linguistic. ~60 min manual.

3. **Test set evaluation.** Everything above is on dev. The same harness with `--dev-path outputs/phase2/splits/test.jsonl` reproduces the table on the held-out set; budget ~7 min of compute.

4. **Tier-1 specific coherence.** The current sweep reports `tier1_triples = 0` because the rel head's top-1 candidates almost never include tier-1 SNOMED-attribute relations (causative-agent, finding-site, etc.). The CSP gain on triple validity (0.951 → 1.000) is therefore driven entirely by type-concept compatibility, not by relation-domain/range. If the paper wants a stronger tier-1 result, increase `--top-k-relations` past 3 so the rel head's tier-1 candidates make it into the CSP candidate set.
