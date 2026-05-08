# Outputs Manifest

This file lists artifacts produced by the CANON pipeline that are **not**
checked into git, why, and how to reproduce them locally. Smaller artifacts
(reference tables, summary JSONs, the gold relation_mapped/mapped/unified
splits) remain tracked and ship with the repo.

The exclusion rule today is strictly minimal: only files that exceed
GitHub's 100 MB per-blob limit are gitignored. Everything else is tracked.
See `.gitignore` for the live list.

## Untracked artifacts

| Path | Size | Phase | Producer | Regenerate with |
|---|---|---|---|---|
| `outputs/phase2/silver/train.jsonl` | ~558 MB | 2.6 (legacy variant) | `scripts/pubtator_silver.py` — synthetic-text "{subj} {rel} {obj}" silver from the bulk PubTator3 TSVs | `CANON_DOWNLOAD_SILVER=1 python -m scripts.pubtator_silver` (requires `Data/PubTator3-2/{disease,chemical,relation}2pubtator3.gz`) |
| `outputs/phase2/splits/train.jsonl` | ~145 MB | 2.7 | `scripts/assemble_splits.py` — concatenates BioRED + BC5CDR + synthetic + PubTator3 silver train shards | `python main.py --only 2.7` (requires Phase 2.3, 2.5, 2.6 outputs to already exist) |

## Files that are tracked but worth flagging

These are still under the 100 MB limit and remain in git, but co-devs should
know they bloat clones:

| Path | Size | Notes |
|---|---|---|
| `outputs/phase2/silver/PubTator3/train.jsonl` | ~94 MB | Canonical Phase 2.6 output (BioCXML-derived). Will exceed 100 MB if SAMPLE_SIZE in `silver_pubtator.py` is raised — gitignore proactively before that happens. |
| `outputs/phase2/synthetic/train.jsonl` | ~43 MB | Phase 2.5 SNOMED-derived synthetic triples |
| `outputs/phase1/snomed_hierarchy.pkl` | ~38 MB | Phase 1.6 NetworkX graph |
| `outputs/phase2/soft_mapping_lookup.json` | ~15 MB | Phase 2.4 soft-mapping table |
| `outputs/phase2/silver_raw/bench_20k.biocjson` | ~14 MB | Legacy Phase 2.6 benchmark cache |
| `outputs/phase2/silver/raw/biocxml/` | ~135 MB total (250 small XML files) | Phase 2.6 PubTator3 API response cache. Each batch is well under the limit, but the directory is large. Already lives under `silver/raw/` and is regenerable via the Phase 2.6 step. |

## Regeneration order

If you clone fresh and need every output, run phases in order. The pipeline
has a single CLI entry point:

```
python main.py --only 1.1 1.2 1.3 1.4 1.5 1.6 1.7
python main.py --only 2.1 2.2 2.3 2.4 2.5
CANON_DOWNLOAD_SILVER=1 python main.py --only 2.6   # network call
python main.py --only 2.7                            # depends on 2.3, 2.5, 2.6
```

Phase 2.6 is the only step with an external dependency (the PubTator 3.0
REST API) and is gated by `CANON_DOWNLOAD_SILVER=1` so a default
`python main.py` run stays offline. The 250-batch BioC-XML cache under
`outputs/phase2/silver/raw/biocxml/` makes re-runs of 2.6 free.

## Adding new gitignored outputs

When a new output crosses 100 MB (or 50 MB, GitHub's warning threshold),
add it to `.gitignore` and document it here in the same row format. Don't
silently exclude — every excluded file should appear in this manifest with
a regeneration command.
