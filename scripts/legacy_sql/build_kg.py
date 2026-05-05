from __future__ import annotations

import argparse

from utils import get_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified KG tables from staged schemas.")
    parser.add_argument("--reset", action="store_true", help="Truncate kg tables before rebuilding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with get_connection() as conn:
        with conn.cursor() as cursor:
            if args.reset:
                print("Resetting KG tables...")
                cursor.execute("TRUNCATE TABLE kg.relations, kg.concept_xrefs, kg.concepts RESTART IDENTITY CASCADE")
                conn.commit()

            print("Inserting into kg.concepts (preferred/semantic)...")
            cursor.execute(
                """
                WITH preferred AS (
                    SELECT DISTINCT ON (c.cui)
                        c.cui,
                        c.str
                    FROM umls.mrconso c
                    WHERE c.lat = 'ENG'
                    ORDER BY
                        c.cui,
                        CASE WHEN c.ispref = 'Y' THEN 0 ELSE 1 END,
                        CASE WHEN c.tty = 'PT' THEN 0 ELSE 1 END,
                        c.aui
                ),
                semantic AS (
                    SELECT
                        cui,
                        string_agg(DISTINCT sty, '|' ORDER BY sty) AS semantic_type
                    FROM umls.mrsty
                    GROUP BY cui
                )
                INSERT INTO kg.concepts (cui, preferred_name, semantic_type)
                SELECT p.cui, p.str, s.semantic_type
                FROM preferred p
                LEFT JOIN semantic s
                    ON s.cui = p.cui
                ON CONFLICT (cui) DO UPDATE
                SET preferred_name = EXCLUDED.preferred_name,
                    semantic_type = COALESCE(EXCLUDED.semantic_type, kg.concepts.semantic_type)
                """
            )
            conn.commit()

            print("Inserting into kg.concepts (fallback)...")
            cursor.execute(
                """
                INSERT INTO kg.concepts (cui, preferred_name, semantic_type)
                SELECT c.cui, MIN(c.str), NULL
                FROM umls.mrconso c
                GROUP BY c.cui
                ON CONFLICT (cui) DO NOTHING
                """
            )
            conn.commit()

            print("Inserting UMLS xrefs...")
            cursor.execute(
                """
                INSERT INTO kg.concept_xrefs (source_dataset, source_table, source_id, cui, xref_type)
                SELECT
                    'umls',
                    'mrconso',
                    COALESCE(NULLIF(c.sdui, ''), NULLIF(c.code, '')) AS source_id,
                    c.cui,
                    c.sab
                FROM umls.mrconso c
                WHERE c.sab IN ('MSH', 'SNOMEDCT_US', 'HGNC')
                  AND COALESCE(NULLIF(c.sdui, ''), NULLIF(c.code, '')) IS NOT NULL
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

            print("Inserting CDR xrefs...")
            cursor.execute(
                """
                INSERT INTO kg.concept_xrefs (source_dataset, source_table, source_id, cui, xref_type)
                SELECT
                    'cdr',
                    'entities',
                    m.mesh_id,
                    map.cui,
                    'MSH'
                FROM cdr.entities e
                CROSS JOIN LATERAL unnest(e.mesh_id_list) AS m(mesh_id)
                JOIN (
                    SELECT sdui, MIN(cui) AS cui
                    FROM umls.mrconso
                    WHERE sab = 'MSH' AND sdui IS NOT NULL
                    GROUP BY sdui
                ) map
                    ON map.sdui = m.mesh_id
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

            print("Inserting BioRED xrefs...")
            cursor.execute(
                """
                INSERT INTO kg.concept_xrefs (source_dataset, source_table, source_id, cui, xref_type)
                SELECT
                    'biored',
                    'entities',
                    token,
                    map.cui,
                    map.xref_type
                FROM biored.entities e
                CROSS JOIN LATERAL unnest(e.identifier_list) AS token
                JOIN (
                    SELECT sdui AS source_id, MIN(cui) AS cui, 'MSH'::TEXT AS xref_type
                    FROM umls.mrconso
                    WHERE sab = 'MSH' AND sdui IS NOT NULL
                    GROUP BY sdui
                    UNION
                    SELECT COALESCE(NULLIF(code, ''), sdui) AS source_id, MIN(cui) AS cui, 'HGNC'::TEXT AS xref_type
                    FROM umls.mrconso
                    WHERE sab = 'HGNC' AND COALESCE(NULLIF(code, ''), sdui) IS NOT NULL
                    GROUP BY COALESCE(NULLIF(code, ''), sdui)
                ) map
                    ON map.source_id = token
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

            print("Inserting SNOMED xrefs...")
            cursor.execute(
                """
                INSERT INTO kg.concept_xrefs (source_dataset, source_table, source_id, cui, xref_type)
                SELECT
                    'snomed',
                    'concepts',
                    s.id::TEXT,
                    map.cui,
                    'SNOMEDCT_US'
                FROM snomed.concepts s
                JOIN (
                    SELECT code, MIN(cui) AS cui
                    FROM umls.mrconso
                    WHERE sab = 'SNOMEDCT_US' AND code IS NOT NULL
                    GROUP BY code
                ) map
                    ON map.code = s.id::TEXT
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

            print("Inserting CDR relations...")
            cursor.execute(
                """
                WITH mesh_map AS (
                    SELECT sdui, MIN(cui) AS cui
                    FROM umls.mrconso
                    WHERE sab = 'MSH' AND sdui IS NOT NULL
                    GROUP BY sdui
                )
                INSERT INTO kg.relations (
                    source_dataset,
                    source_doc_id,
                    relation_type,
                    subject_cui,
                    object_cui,
                    subject_source_id,
                    object_source_id,
                    novelty,
                    evidence
                )
                SELECT
                    'cdr',
                    r.pmid::TEXT,
                    r.relation_type,
                    subj.cui,
                    obj.cui,
                    r.subject_id,
                    r.object_id,
                    '',
                    d.title || ' ' || d.abstract
                FROM cdr.relations r
                JOIN mesh_map subj
                    ON subj.sdui = r.subject_id
                JOIN mesh_map obj
                    ON obj.sdui = r.object_id
                JOIN cdr.documents d
                    ON d.pmid = r.pmid
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

            print("Inserting BioRED relations...")
            cursor.execute(
                """
                WITH biored_map AS (
                    SELECT source_id, MIN(cui) AS cui
                    FROM kg.concept_xrefs
                    WHERE source_dataset = 'biored'
                    GROUP BY source_id
                )
                INSERT INTO kg.relations (
                    source_dataset,
                    source_doc_id,
                    relation_type,
                    subject_cui,
                    object_cui,
                    subject_source_id,
                    object_source_id,
                    novelty,
                    evidence
                )
                SELECT
                    'biored',
                    r.pmid::TEXT,
                    r.relation_type,
                    subj.cui,
                    obj.cui,
                    r.subject_id,
                    r.object_id,
                    r.novelty,
                    d.title || ' ' || d.abstract
                FROM biored.relations r
                JOIN biored_map subj
                    ON subj.source_id = r.subject_id
                JOIN biored_map obj
                    ON obj.source_id = r.object_id
                JOIN biored.documents d
                    ON d.pmid = r.pmid
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

            print("Inserting SNOMED relations...")
            cursor.execute(
                """
                WITH snomed_map AS (
                    SELECT code, MIN(cui) AS cui
                    FROM umls.mrconso
                    WHERE sab = 'SNOMEDCT_US' AND code IS NOT NULL
                    GROUP BY code
                )
                INSERT INTO kg.relations (
                    source_dataset,
                    source_doc_id,
                    relation_type,
                    subject_cui,
                    object_cui,
                    subject_source_id,
                    object_source_id,
                    novelty,
                    evidence
                )
                SELECT
                    'snomed',
                    NULL,
                    rel.type_id::TEXT,
                    src.cui,
                    dst.cui,
                    rel.source_id::TEXT,
                    rel.destination_id::TEXT,
                    '',
                    NULL
                FROM snomed.relationships rel
                JOIN snomed_map src
                    ON src.code = rel.source_id::TEXT
                JOIN snomed_map dst
                    ON dst.code = rel.destination_id::TEXT
                WHERE rel.active = TRUE
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

            print("Inserting UMLS relations...")
            cursor.execute(
                """
                INSERT INTO kg.relations (
                    source_dataset,
                    source_doc_id,
                    relation_type,
                    subject_cui,
                    object_cui,
                    subject_source_id,
                    object_source_id,
                    novelty,
                    evidence
                )
                SELECT
                    'umls',
                    NULL,
                    COALESCE(NULLIF(mr.rela, ''), NULLIF(mr.rel, '')),
                    mr.cui1,
                    mr.cui2,
                    mr.aui1,
                    mr.aui2,
                    '',
                    NULL
                FROM umls.mrrel mr
                JOIN kg.concepts c1
                    ON c1.cui = mr.cui1
                JOIN kg.concepts c2
                    ON c2.cui = mr.cui2
                WHERE COALESCE(NULLIF(mr.rela, ''), NULLIF(mr.rel, '')) IS NOT NULL
                ON CONFLICT DO NOTHING
                """
            )
            conn.commit()

    print("KG build complete.")


if __name__ == "__main__":
    main()
