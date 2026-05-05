# Phase 1.3 — Non-MeSH Vocabulary Scoping Decision

## Decision

Genes (NCBI Gene), variants (dbSNP), species (NCBI Taxonomy), and cell lines
(Cellosaurus) are **excluded from SNOMED CT concept normalization**. They
remain first-class participants in NER and relation extraction, but they
do **not** receive SNOMED-mapped target labels and they are **not** subject
to the CSP solver's MRCM constraints.

The plan's stated rationale is taken in full: SNOMED CT has limited
genomic content, so genes and variants do not map cleanly. Holding
species and cell lines out of SNOMED normalization too keeps the CSP
solver's domain coherent — every entity it constrains has a SNOMED code,
every entity without a SNOMED code is invisible to it.

## Concrete consequences per pipeline component

| Component | In-scope (chemical, disease) | Out-of-scope (gene, variant, species, cell_line) |
| --- | --- | --- |
| NER head (Phase 3.2) | BIO tags emitted | BIO tags emitted (separate non-SNOMED tag classes) |
| Concept normalization head (Phase 3.2) | Bi-encoder over SNOMED candidates | **Skipped** |
| Soft-label preprocessing (Phase 2.4) | SNOMED neighborhood expansion | **Skipped** |
| Relation extraction head (Phase 3.2) | Pair embedding + MLP | Pair embedding + MLP (same as in-scope) |
| CSP solver (Phase 3.5) | Domain/range MRCM constraints applied | **Skipped** — entity treated as unconstrained |
| Coherence metric (Phase 4.3) | Counted in per-triple validity | Triples involving these entities are excluded from the validity rate |

Source codes for out-of-scope entities are still **retained** on every
mention (NCBI Gene IDs, dbSNP rsIDs, NCBI Taxonomy IDs, Cellosaurus IDs),
flagged `non_snomed=true` in the unified annotation format (Phase 2.1).
This keeps downstream linking to external KGs available without
contaminating the SNOMED-aware paths.

## Per-corpus entity types (registry source of truth)

The authoritative registry is [`scripts/entity_scope.py`](../scripts/entity_scope.py).
The table below mirrors that file at the time of writing.

| Corpus | Entity type | Vocabulary | Semantic class | SNOMED-normalized? |
| --- | --- | --- | --- | --- |
| BioRED | ChemicalEntity | MeSH | chemical | yes |
| BioRED | DiseaseOrPhenotypicFeature | MeSH | disease | yes |
| BioRED | GeneOrGeneProduct | NCBI Gene | gene | **no** |
| BioRED | SequenceVariant | dbSNP | variant | **no** |
| BioRED | OrganismTaxon | NCBI Taxonomy | species | **no** |
| BioRED | CellLine | Cellosaurus | cell_line | **no** |
| BC5CDR | Chemical | MeSH | chemical | yes |
| BC5CDR | Disease | MeSH | disease | yes |
| NCBI Disease | DiseaseClass / SpecificDisease / Modifier / CompositeMention | MeSH | disease | yes |
| NLM-Chem | Chemical | MeSH | chemical | yes |

## What this means for relation extraction

Relations are still extracted across every entity-type pair, including
out-of-scope ones (e.g., a `GeneOrGeneProduct`–`DiseaseOrPhenotypicFeature`
relation in BioRED). At inference time the CSP solver enforces MRCM
constraints **only** on relations where both endpoints carry SNOMED codes
— gene–disease relations therefore fall in Tier 2 (empirical) per the
Phase 1.4 schema and are not blocked by ontological constraints they
cannot satisfy.

## Why species and cell lines are excluded too (vs. mapping them to SNOMED's organism axis)

SNOMED's organism hierarchy could in principle accept species mappings,
and there is a coarse "Cell structure" subtree relevant to cell lines.
The plan calls this out as a judgment call. We are erring on the side of
exclusion because:

1. The MRCM domain/range constraints we extract in Phase 1.5 are anchored
   on Clinical-finding / Procedure / Body-structure / Substance — adding
   organisms or cell structures to the CSP solver expands its surface
   area without a corresponding evaluation signal in BioRED/BC5CDR.
2. NCBI Taxonomy and Cellosaurus IDs are already useful as-is for any
   downstream cross-KG linking; we lose nothing by retaining them and
   skipping the SNOMED indirection.
3. It makes the scoping rule trivially memorable: **MeSH-coded entities
   are SNOMED-normalized, everything else is not.**

If a future evaluation surfaces a need for species-level CSP constraints
(e.g., a cross-domain transfer benchmark), the registry in
`entity_scope.py` is the single switch that flips them in.

## Auditing

Run `python scripts/scope_audit.py` (or the Phase 1.3 step in `main.py`)
to regenerate `outputs/phase1/entity_scope_audit.csv` and
`entity_scope_summary.csv`. The audit is the empirical ground-truth for
how much of each corpus is in-scope vs. out-of-scope, and surfaces any
unregistered entity types as warnings.
