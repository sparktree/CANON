# Phase 4.6 Error Review

Review the document text, gold annotation, unconstrained neural assignment, and
CSP assignment in each `error_review.jsonl` record. Set `review_category` to
exactly one of:

- `mapping-failure`: the source-to-SNOMED mapping or gold normalization is wrong.
- `coverage-gap`: the correct concept was absent from the normalization candidates.
- `constraint-over-restriction`: a valid unusual assignment was rejected by MRCM/SN.
- `linguistic-failure`: spans, types, concepts, or relations were misunderstood
  despite adequate candidates and valid constraints.

Use `review_notes` for the evidence. Do not label an absent CTD assertion as a
mapping failure: CTD is incomplete and its attestation rate is a lower bound.
Summarize a completed artifact with:

```bash
python scripts/error_analysis.py summarize --input outputs/phase4/error_review.jsonl
```
