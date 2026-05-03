BEGIN;

CREATE UNIQUE INDEX IF NOT EXISTS cdr_entities_dedupe_idx
    ON cdr.entities (
        pmid,
        start_offset,
        end_offset,
        mention,
        entity_type,
        COALESCE(mesh_id_raw, ''),
        COALESCE(composite_mentions, '')
    );
CREATE INDEX IF NOT EXISTS cdr_entities_pmid_idx ON cdr.entities (pmid);
CREATE INDEX IF NOT EXISTS cdr_entities_mesh_single_idx ON cdr.entities (mesh_id);
CREATE INDEX IF NOT EXISTS cdr_entities_mesh_list_gin_idx ON cdr.entities USING GIN (mesh_id_list);
CREATE INDEX IF NOT EXISTS cdr_relations_pmid_idx ON cdr.relations (pmid);
CREATE INDEX IF NOT EXISTS cdr_relations_subject_idx ON cdr.relations (subject_id);
CREATE INDEX IF NOT EXISTS cdr_relations_object_idx ON cdr.relations (object_id);

CREATE INDEX IF NOT EXISTS biored_entities_pmid_idx ON biored.entities (pmid);
CREATE INDEX IF NOT EXISTS biored_entities_identifier_idx ON biored.entities (normalized_identifier);
CREATE INDEX IF NOT EXISTS biored_entities_identifier_list_gin_idx ON biored.entities USING GIN (identifier_list);
CREATE INDEX IF NOT EXISTS biored_relations_pmid_idx ON biored.relations (pmid);
CREATE INDEX IF NOT EXISTS biored_relations_subject_idx ON biored.relations (subject_id);
CREATE INDEX IF NOT EXISTS biored_relations_object_idx ON biored.relations (object_id);

CREATE INDEX IF NOT EXISTS umls_mrconso_cui_idx ON umls.mrconso (cui);
CREATE INDEX IF NOT EXISTS umls_mrconso_sab_sdui_idx ON umls.mrconso (sab, sdui);
CREATE INDEX IF NOT EXISTS umls_mrconso_sab_code_idx ON umls.mrconso (sab, code);
CREATE INDEX IF NOT EXISTS umls_mrconso_lat_ispref_idx ON umls.mrconso (lat, ispref);
CREATE INDEX IF NOT EXISTS umls_mrconso_str_gin_idx ON umls.mrconso USING GIN (to_tsvector('english', str));
CREATE INDEX IF NOT EXISTS umls_mrrel_cui1_idx ON umls.mrrel (cui1);
CREATE INDEX IF NOT EXISTS umls_mrrel_cui2_idx ON umls.mrrel (cui2);
CREATE INDEX IF NOT EXISTS umls_mrrel_rela_idx ON umls.mrrel (rela);
CREATE INDEX IF NOT EXISTS umls_mrsty_cui_idx ON umls.mrsty (cui);
CREATE INDEX IF NOT EXISTS umls_mrdef_cui_idx ON umls.mrdef (cui);

CREATE INDEX IF NOT EXISTS snomed_concepts_active_idx ON snomed.concepts (active);
CREATE INDEX IF NOT EXISTS snomed_descriptions_concept_idx ON snomed.descriptions (concept_id);
CREATE INDEX IF NOT EXISTS snomed_text_definitions_concept_idx ON snomed.text_definitions (concept_id);
CREATE INDEX IF NOT EXISTS snomed_relationships_source_idx ON snomed.relationships (source_id);
CREATE INDEX IF NOT EXISTS snomed_relationships_destination_idx ON snomed.relationships (destination_id);
CREATE INDEX IF NOT EXISTS snomed_stated_relationships_source_idx ON snomed.stated_relationships (source_id);
CREATE INDEX IF NOT EXISTS snomed_stated_relationships_destination_idx ON snomed.stated_relationships (destination_id);
CREATE INDEX IF NOT EXISTS snomed_extended_map_ref_component_idx ON snomed.extended_map (referenced_component_id);
CREATE INDEX IF NOT EXISTS snomed_simple_map_ref_component_idx ON snomed.simple_map (referenced_component_id);

CREATE INDEX IF NOT EXISTS kg_concepts_name_idx ON kg.concepts (preferred_name);
CREATE INDEX IF NOT EXISTS kg_xrefs_source_idx ON kg.concept_xrefs (source_dataset, source_id);
CREATE INDEX IF NOT EXISTS kg_xrefs_cui_idx ON kg.concept_xrefs (cui);
CREATE INDEX IF NOT EXISTS kg_relations_subject_idx ON kg.relations (subject_cui);
CREATE INDEX IF NOT EXISTS kg_relations_object_idx ON kg.relations (object_cui);
CREATE INDEX IF NOT EXISTS kg_relations_type_idx ON kg.relations (relation_type);
CREATE INDEX IF NOT EXISTS kg_relations_source_dataset_idx ON kg.relations (source_dataset);

COMMIT;
