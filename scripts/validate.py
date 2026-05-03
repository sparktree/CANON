from __future__ import annotations

from utils import get_connection


TABLES = [
    "cdr.documents",
    "cdr.entities",
    "cdr.relations",
    "biored.documents",
    "biored.entities",
    "biored.relations",
    "umls.mrconso",
    "umls.mrsty",
    "umls.mrdef",
    "umls.mrrel",
    "snomed.concepts",
    "snomed.descriptions",
    "snomed.text_definitions",
    "snomed.relationships",
    "snomed.stated_relationships",
    "snomed.extended_map",
    "snomed.simple_map",
    "kg.concepts",
    "kg.concept_xrefs",
    "kg.relations",
]


def fetch_scalar(cursor, query: str) -> int:
    cursor.execute(query)
    value = cursor.fetchone()
    return int(value[0]) if value and value[0] is not None else 0


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            print("Row counts:")
            for table in TABLES:
                count = fetch_scalar(cursor, f"SELECT COUNT(*) FROM {table}")
                print(f"- {table}: {count:,}")

            print("\nIntegrity checks:")
            checks = {
                "cdr entities missing document": """
                    SELECT COUNT(*)
                    FROM cdr.entities e
                    LEFT JOIN cdr.documents d ON d.pmid = e.pmid
                    WHERE d.pmid IS NULL
                """,
                "cdr relations missing document": """
                    SELECT COUNT(*)
                    FROM cdr.relations r
                    LEFT JOIN cdr.documents d ON d.pmid = r.pmid
                    WHERE d.pmid IS NULL
                """,
                "biored entities missing document": """
                    SELECT COUNT(*)
                    FROM biored.entities e
                    LEFT JOIN biored.documents d ON d.pmid = e.pmid
                    WHERE d.pmid IS NULL
                """,
                "biored relations missing document": """
                    SELECT COUNT(*)
                    FROM biored.relations r
                    LEFT JOIN biored.documents d ON d.pmid = r.pmid
                    WHERE d.pmid IS NULL
                """,
                "kg cdr relations": "SELECT COUNT(*) FROM kg.relations WHERE source_dataset = 'cdr'",
                "kg biored relations": "SELECT COUNT(*) FROM kg.relations WHERE source_dataset = 'biored'",
                "kg snomed relations": "SELECT COUNT(*) FROM kg.relations WHERE source_dataset = 'snomed'",
                "kg umls relations": "SELECT COUNT(*) FROM kg.relations WHERE source_dataset = 'umls'",
            }
            for label, query in checks.items():
                print(f"- {label}: {fetch_scalar(cursor, query):,}")

            print("\nSpot checks:")
            cursor.execute(
                """
                SELECT
                    subject_cui,
                    relation_type,
                    object_cui,
                    source_dataset
                FROM kg.relations
                ORDER BY created_at DESC
                LIMIT 10
                """
            )
            rows = cursor.fetchall()
            for row in rows:
                print(f"- {row[3]}: {row[0]} --[{row[1]}]--> {row[2]}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
