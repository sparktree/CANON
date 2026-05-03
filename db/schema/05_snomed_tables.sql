BEGIN;

CREATE TABLE IF NOT EXISTS snomed.concepts (
    id BIGINT PRIMARY KEY,
    effective_time DATE NOT NULL,
    active BOOLEAN NOT NULL,
    module_id BIGINT NOT NULL,
    definition_status_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS snomed.descriptions (
    id BIGINT PRIMARY KEY,
    effective_time DATE NOT NULL,
    active BOOLEAN NOT NULL,
    module_id BIGINT NOT NULL,
    concept_id BIGINT NOT NULL REFERENCES snomed.concepts (id),
    language_code TEXT NOT NULL,
    type_id BIGINT NOT NULL,
    term TEXT NOT NULL,
    case_significance_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS snomed.text_definitions (
    id BIGINT PRIMARY KEY,
    effective_time DATE NOT NULL,
    active BOOLEAN NOT NULL,
    module_id BIGINT NOT NULL,
    concept_id BIGINT NOT NULL REFERENCES snomed.concepts (id),
    language_code TEXT NOT NULL,
    type_id BIGINT NOT NULL,
    term TEXT NOT NULL,
    case_significance_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS snomed.relationships (
    id BIGINT PRIMARY KEY,
    effective_time DATE NOT NULL,
    active BOOLEAN NOT NULL,
    module_id BIGINT NOT NULL,
    source_id BIGINT NOT NULL REFERENCES snomed.concepts (id),
    destination_id BIGINT NOT NULL REFERENCES snomed.concepts (id),
    relationship_group INTEGER NOT NULL,
    type_id BIGINT NOT NULL,
    characteristic_type_id BIGINT NOT NULL,
    modifier_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS snomed.stated_relationships (
    id BIGINT PRIMARY KEY,
    effective_time DATE NOT NULL,
    active BOOLEAN NOT NULL,
    module_id BIGINT NOT NULL,
    source_id BIGINT NOT NULL REFERENCES snomed.concepts (id),
    destination_id BIGINT NOT NULL REFERENCES snomed.concepts (id),
    relationship_group INTEGER NOT NULL,
    type_id BIGINT NOT NULL,
    characteristic_type_id BIGINT NOT NULL,
    modifier_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS snomed.extended_map (
    id TEXT PRIMARY KEY,
    effective_time DATE NOT NULL,
    active BOOLEAN NOT NULL,
    module_id BIGINT NOT NULL,
    refset_id BIGINT NOT NULL,
    referenced_component_id BIGINT NOT NULL,
    map_group INTEGER NOT NULL,
    map_priority INTEGER NOT NULL,
    map_rule TEXT NOT NULL,
    map_advice TEXT NOT NULL,
    map_target TEXT,
    correlation_id BIGINT NOT NULL,
    map_category_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS snomed.simple_map (
    id TEXT PRIMARY KEY,
    effective_time DATE NOT NULL,
    active BOOLEAN NOT NULL,
    module_id BIGINT NOT NULL,
    refset_id BIGINT NOT NULL,
    referenced_component_id BIGINT NOT NULL,
    map_target TEXT
);

COMMIT;
