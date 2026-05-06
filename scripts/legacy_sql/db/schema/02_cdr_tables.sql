BEGIN;

CREATE TABLE IF NOT EXISTS cdr.documents (
    pmid BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'dev', 'test')),
    source_file TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cdr.entities (
    entity_id BIGSERIAL PRIMARY KEY,
    pmid BIGINT NOT NULL REFERENCES cdr.documents (pmid) ON DELETE CASCADE,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    mention TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    mesh_id_raw TEXT,
    mesh_id TEXT,
    mesh_id_list TEXT[] NOT NULL DEFAULT '{}',
    composite_mentions TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cdr.relations (
    relation_id BIGSERIAL PRIMARY KEY,
    pmid BIGINT NOT NULL REFERENCES cdr.documents (pmid) ON DELETE CASCADE,
    relation_type TEXT NOT NULL DEFAULT 'CID',
    subject_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pmid, relation_type, subject_id, object_id)
);

COMMIT;
