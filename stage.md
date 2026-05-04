# Data Staging Plan for Biomedical Knowledge Graph Construction

## Context

The project at `/Volumes/Khurrum/AdvMLinCL` contains four biomedical datasets (CDR, BioRED, UMLS 2025AB, SNOMED CT US Edition) totaling ~9 GB. The goal is to stage all four into PostgreSQL to support Knowledge Graph construction. There is currently no code in the project — only raw data files.

---

## PostgreSQL Schema Design

Five schemas, one per dataset plus an integration layer:

| Schema | Purpose | Key Tables |
|--------|---------|------------|
| `cdr` | Chemical-Disease Relation corpus | documents, entities, relations |
| `biored` | BioRED relation extraction corpus | documents, entities, relations |
| `umls` | UMLS Metathesaurus | mrconso, mrrel, mrsty, mrdef |
| `snomed` | SNOMED CT US Edition | concepts, descriptions, relationships, stated_relationships, extended_map, simple_map |
| `kg` | Unified Knowledge Graph | concepts, concept_xrefs, relations |

### Cross-Dataset Linkage (via UMLS as Rosetta Stone)

- **CDR/BioRED MeSH IDs** -> `umls.mrconso WHERE sab='MSH' AND sdui=mesh_id` -> CUI
- **SNOMED concept IDs** -> `umls.mrconso WHERE sab='SNOMEDCT_US' AND code=snomed_id` -> CUI
- **BioRED Gene IDs** -> UMLS HGNC source -> CUI

---

## File Structure to Create

```
/Volumes/Khurrum/AdvMLinCL/
  db/
    schema/
      01_create_schemas.sql
      02_cdr_tables.sql
      03_biored_tables.sql
      04_umls_tables.sql
      05_snomed_tables.sql
      06_kg_tables.sql
      07_indexes.sql
  scripts/
    config.py          # DB connection (from env vars), file paths, batch size
    utils.py           # get_connection(), bulk COPY helper, PubTator parser
    load_cdr.py        # Parse PubTator -> cdr schema
    load_biored.py     # Parse PubTator -> biored schema
    load_umls.py       # Parse RRF pipe-delimited -> umls schema
    load_snomed.py     # Parse tab-delimited Snapshot -> snomed schema
    build_kg.py        # Unify into kg schema using UMLS linkage
    validate.py        # Row counts, FK integrity checks
  requirements.txt     # psycopg2-binary, tqdm
```

---

## Implementation Phases

### Phase 0: SQL Schema Creation
- Write all `.sql` files under `db/schema/`
- Tables, primary keys, foreign keys, indexes (indexes created post-load for performance)

### Phase 1: Independent Dataset Loaders (can run in parallel: 1a, 1b, 1c)

**1a. `load_cdr.py`** (~1500 docs, ~30K entities, ~3K relations)
- Parse `CDR_TrainingSet.PubTator.txt`, `CDR_DevelopmentSet.PubTator.txt`, `CDR_TestSet.PubTator.txt`
- PubTator format: `|t|` = title, `|a|` = abstract, tab-delimited entity/relation lines
- Map MeSH ID `-1` to NULL; use `ON CONFLICT DO NOTHING` for idempotency

**1b. `load_biored.py`** (~1000 docs, ~13K entities, ~4K relations)
- Parse `Train.PubTator`, `Dev.PubTator`, `Test.PubTator`
- Handle 6 entity types, 8 relation types, novelty flag
- Handle composite identifiers (comma-separated MeSH IDs, variant notations)

**1c. `load_snomed.py`** (~3.6M relationships, ~1.7M descriptions, ~537K concepts)
- Load Snapshot files only (not Full)
- Tab-delimited with headers; use PostgreSQL `COPY` via `psycopg2.copy_expert()`
- Parse `YYYYMMDD` -> DATE, `0/1` -> BOOLEAN
- Loading order: concepts -> descriptions -> text_definitions -> relationships -> stated_relationships -> maps
- Drop indexes before load, recreate after

**1d. UMLS MetamorphoSys Extraction (manual prerequisite for Phase 2)**
- Extract `mmsys.zip`, run MetamorphoSys to produce RRF files
- Recommended subset: MSH, SNOMEDCT_US, NCI, GO, OMIM, HPO, RXNORM
- Output to `Data/2025AB-full/META/`

### Phase 2: UMLS Loading (depends on 1d)

**`load_umls.py`**
- Parse pipe-delimited RRF files (trailing `|` on each line)
- MRCONSO.RRF (~15M rows), MRREL.RRF (~80M rows), MRSTY.RRF (~4M rows), MRDEF.RRF (~500K rows)
- Use `COPY` for bulk performance; optionally filter to `LAT='ENG'` and relevant SABs
- Drop indexes before, recreate after; set `maintenance_work_mem = '2GB'`

### Phase 3: Knowledge Graph Integration (depends on all of Phase 1+2)

**`build_kg.py`**
1. Seed `kg.concepts` from UMLS preferred English names + semantic type classification
2. Link MeSH IDs from CDR/BioRED via `umls.mrconso(sab='MSH', sdui)`
3. Link SNOMED IDs via `umls.mrconso(sab='SNOMEDCT_US', code)`
4. Populate `kg.concept_xrefs` with all cross-references
5. Populate `kg.relations` from all four source datasets

### Phase 4: Validation

**`validate.py`**
- Row count checks per table
- FK integrity verification
- Sample spot-checks (e.g., known CID relations in CDR resolve to valid KG edges)

---

## Key Technical Decisions

1. **PubTator over BioC XML** for CDR/BioRED: simpler to parse, same data, one shared parser
2. **Separate staging schemas + kg integration schema**: keeps raw data faithful, allows independent reloads
3. **UMLS as linkage backbone**: already maps MeSH, SNOMED, OMIM, Gene IDs — no custom mapping needed
4. **PostgreSQL COPY for bulk loading**: required for SNOMED (millions of rows) and UMLS (tens of millions)
5. **Snapshot-only for SNOMED CT**: current state only, avoids multiplying table sizes with historical versions

---

## Critical Data Files

- `Data/CDR_Data/CDR.Corpus.v010516/CDR_*.PubTator.txt`
- `Data/BioRED/{Train,Dev,Test}.PubTator`
- `Data/SnomedCT_.../Snapshot/Terminology/sct2_*.txt`
- `Data/SnomedCT_.../Snapshot/Refset/Map/der2_*Map*.txt`
- `Data/2025AB-full/mmsys.zip` (extract first) -> `META/MRCONSO.RRF`, `MRREL.RRF`, `MRSTY.RRF`, `MRDEF.RRF`

---

## Verification

1. Run each SQL schema file against a local PostgreSQL instance
2. Run loaders in order: Phase 1a/1b/1c in parallel, then Phase 2, then Phase 3
3. Run `validate.py` to check row counts match expectations
4. Spot-check: query a known chemical-disease pair from CDR through the `kg` schema to verify end-to-end linkage
5. Spot-check: traverse a SNOMED "Is a" hierarchy through `kg.relations`

---

## Dependencies

`requirements.txt`: `psycopg2-binary`, `tqdm`
