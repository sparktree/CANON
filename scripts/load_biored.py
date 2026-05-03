from __future__ import annotations

import argparse
from pathlib import Path

from psycopg2.extras import execute_values
from tqdm import tqdm

from config import BATCH_SIZE, BIORED_FILES
from utils import ensure_required_files, get_connection, parse_pubtator, split_composite_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load BioRED PubTator data into biored schema.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--truncate", action="store_true", help="Truncate BioRED tables before loading.")
    return parser.parse_args()


def _identifier_fields(identifier_raw: str) -> tuple[str, list[str], str | None]:
    cleaned = identifier_raw.strip()
    identifier_list = split_composite_ids(cleaned, delimiter=",") if "," in cleaned else [cleaned]
    identifier_list = [token for token in identifier_list if token]
    normalized = identifier_list[0] if identifier_list else None
    return cleaned, identifier_list, normalized


def flush_documents(cursor, rows: list[tuple]) -> None:
    if not rows:
        return
    execute_values(
        cursor,
        """
        INSERT INTO biored.documents (pmid, title, abstract, split, source_file)
        VALUES %s
        ON CONFLICT (pmid) DO UPDATE
        SET title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            split = EXCLUDED.split,
            source_file = EXCLUDED.source_file
        """,
        rows,
    )
    rows.clear()


def flush_entities(cursor, rows: list[tuple]) -> None:
    if not rows:
        return
    execute_values(
        cursor,
        """
        INSERT INTO biored.entities (
            pmid,
            start_offset,
            end_offset,
            mention,
            entity_type,
            identifier_raw,
            identifier_list,
            normalized_identifier
        )
        VALUES %s
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    rows.clear()


def flush_relations(cursor, rows: list[tuple]) -> None:
    if not rows:
        return
    execute_values(
        cursor,
        """
        INSERT INTO biored.relations (
            pmid,
            relation_type,
            subject_id,
            object_id,
            novelty
        )
        VALUES %s
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    rows.clear()


def load_file(cursor, split_name: str, source_file: Path, batch_size: int) -> None:
    document_rows: list[tuple] = []
    entity_rows: list[tuple] = []
    relation_rows: list[tuple] = []

    for record in tqdm(parse_pubtator(source_file), desc=f"BioRED {split_name}", unit="doc"):
        document_rows.append(
            (
                record["pmid"],
                record["title"],
                record["abstract"],
                split_name,
                source_file.name,
            )
        )

        for entity in record["entities"]:
            identifier_raw, identifier_list, normalized = _identifier_fields(entity["identifier_raw"])
            entity_rows.append(
                (
                    entity["pmid"],
                    entity["start_offset"],
                    entity["end_offset"],
                    entity["mention"],
                    entity["entity_type"],
                    identifier_raw,
                    identifier_list,
                    normalized,
                )
            )

        for relation in record["relations"]:
            relation_rows.append(
                (
                    relation["pmid"],
                    relation["relation_type"],
                    relation["subject_id"],
                    relation["object_id"],
                    relation.get("novelty", "") or "",
                )
            )

        if len(document_rows) >= batch_size or len(entity_rows) >= batch_size or len(relation_rows) >= batch_size:
            flush_documents(cursor, document_rows)
            flush_entities(cursor, entity_rows)
            flush_relations(cursor, relation_rows)

    flush_documents(cursor, document_rows)
    flush_entities(cursor, entity_rows)
    flush_relations(cursor, relation_rows)


def main() -> None:
    args = parse_args()
    ensure_required_files(list(BIORED_FILES.values()))

    with get_connection() as conn:
        with conn.cursor() as cursor:
            if args.truncate:
                cursor.execute(
                    """
                    TRUNCATE TABLE biored.relations, biored.entities, biored.documents
                    RESTART IDENTITY CASCADE
                    """
                )
            for split_name, source_file in BIORED_FILES.items():
                load_file(cursor, split_name, source_file, args.batch_size)

        conn.commit()

    print("BioRED load complete.")


if __name__ == "__main__":
    main()
