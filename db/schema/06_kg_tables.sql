BEGIN;

CREATE TABLE IF NOT EXISTS kg.concepts (
    concept_id BIGSERIAL PRIMARY KEY,
    cui TEXT NOT NULL UNIQUE,
    preferred_name TEXT,
    semantic_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kg.concept_xrefs (
    xref_id BIGSERIAL PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    source_table TEXT NOT NULL,
    source_id TEXT NOT NULL,
    cui TEXT NOT NULL REFERENCES kg.concepts (cui),
    xref_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_dataset, source_table, source_id, cui, xref_type)
);

CREATE TABLE IF NOT EXISTS kg.relations (
    relation_id BIGSERIAL PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    source_doc_id TEXT,
    relation_type TEXT NOT NULL,
    subject_cui TEXT NOT NULL REFERENCES kg.concepts (cui),
    object_cui TEXT NOT NULL REFERENCES kg.concepts (cui),
    subject_source_id TEXT,
    object_source_id TEXT,
    novelty TEXT NOT NULL DEFAULT '',
    evidence TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (
        source_dataset,
        source_doc_id,
        relation_type,
        subject_cui,
        object_cui,
        subject_source_id,
        object_source_id,
        novelty
    )
);

COMMIT;
