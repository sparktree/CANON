# Phase 1 Test Plan

## Goals

Phase 1 is mostly data-pipeline code with large local artifacts. The test plan
therefore separates fast invariant checks from expensive rebuild checks.

## Fast Checks

Run on every branch before committing:

```bash
python3 -m unittest discover -s tests
```

These checks should not rewrite outputs or parse the full UMLS/SNOMED releases.
They validate:

- `main.py` registers every implemented Phase 1 step.
- relation schema CSV matches the in-memory mapping table.
- relation schema probability groups sum to `1.0`.
- MRCM JSON contains every Tier-1 relation with non-empty domain/range blocks.
- MRCM JSON provenance matches the configured RF2 MRCM files.
- SNOMED hierarchy stats and cache artifacts exist and pass basic consistency
  checks.

## Rebuild Checks

Run when changing parser/build logic or updating source data:

```bash
python3 main.py --only 1.1
python3 main.py --only 1.2
python3 main.py --only 1.3
python3 main.py --only 1.4
python3 main.py --only 1.5
python3 main.py --only 1.6
```

For 1.1 and 1.6, use `--force-reparse` only when validating full rebuilds from
raw files. Otherwise use cache-backed runs to avoid unnecessary long work.

## Current Known Risks

- Generated artifacts can become stale when `config.py` discovers a newer local
  SNOMED snapshot.
- MRCM ECL operators are preserved in the raw ECL strings, but only root SCTIDs
  are currently surfaced in the simplified root-id lists.
- Phase 1.4 exact relation lookup is ordered; Phase 2 should either canonicalize
  entity-pair orientation or the mapping table should include observed reverse
  orientations.
