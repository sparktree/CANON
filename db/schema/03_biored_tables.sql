BEGIN;

CREATE TABLE IF NOT EXISTS biored.documents (
    pmid BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'dev', 'test')),
    source_file TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS biored.entities (
    entity_id BIGSERIAL PRIMARY KEY,
    pmid BIGINT NOT NULL REFERENCES biored.documents (pmid) ON DELETE CASCADE,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    mention TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    identifier_raw TEXT NOT NULL,
    identifier_list TEXT[] NOT NULL DEFAULT '{}',
    normalized_identifier TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pmid, start_offset, end_offset, mention, entity_type, identifier_raw)
);

CREATE TABLE IF NOT EXISTS biored.relations (
    relation_id BIGSERIAL PRIMARY KEY,
    pmid BIGINT NOT NULL REFERENCES biored.documents (pmid) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    novelty TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pmid, relation_type, subject_id, object_id, novelty)
);

COMMIT;
