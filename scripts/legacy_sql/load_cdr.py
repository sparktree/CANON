from __future__ import annotations

import argparse
from pathlib import Path

from psycopg2.extras import execute_values
from tqdm import tqdm

from config import BATCH_SIZE, CDR_FILES
from utils import ensure_required_files, get_connection, normalize_mesh_id, parse_pubtator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load CDR PubTator data into cdr schema.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--truncate", action="store_true", help="Truncate CDR tables before loading.")
    return parser.parse_args()


def _mesh_fields(identifier_raw: str | None, extra_info: str | None) -> tuple[str | None, str | None, list[str], str | None]:
    if identifier_raw is None:
        return None, None, [], extra_info

    raw_clean = identifier_raw.strip()
    if normalize_mesh_id(raw_clean) is None:
        return raw_clean, None, [], extra_info

    mesh_ids = [token.strip() for token in raw_clean.split("|") if token.strip()]
    single_mesh_id = mesh_ids[0] if len(mesh_ids) == 1 else None
    return raw_clean, single_mesh_id, mesh_ids, extra_info


def flush_documents(cursor, rows: list[tuple]) -> None:
    if not rows:
        return
    execute_values(
        cursor,
        """
        INSERT INTO cdr.documents (pmid, title, abstract, split, source_file)
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
        INSERT INTO cdr.entities (
            pmid,
            start_offset,
            end_offset,
            mention,
            entity_type,
            mesh_id_raw,
            mesh_id,
            mesh_id_list,
            composite_mentions
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
        INSERT INTO cdr.relations (pmid, relation_type, subject_id, object_id)
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

    for record in tqdm(parse_pubtator(source_file), desc=f"CDR {split_name}", unit="doc"):
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
            mesh_id_raw, mesh_id_single, mesh_id_list, composite_mentions = _mesh_fields(
                entity["identifier_raw"], entity.get("extra_info")
            )
            entity_rows.append(
                (
                    entity["pmid"],
                    entity["start_offset"],
                    entity["end_offset"],
                    entity["mention"],
                    entity["entity_type"],
                    mesh_id_raw,
                    mesh_id_single,
                    mesh_id_list,
                    composite_mentions,
                )
            )

        for relation in record["relations"]:
            relation_rows.append(
                (
                    relation["pmid"],
                    relation["relation_type"],
                    relation["subject_id"],
                    relation["object_id"],
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
    ensure_required_files(list(CDR_FILES.values()))

    with get_connection() as conn:
        with conn.cursor() as cursor:
            if args.truncate:
                cursor.execute(
                    """
                    TRUNCATE TABLE cdr.relations, cdr.entities, cdr.documents
                    RESTART IDENTITY CASCADE
                    """
                )
            for split_name, source_file in CDR_FILES.items():
                load_file(cursor, split_name, source_file, args.batch_size)

        conn.commit()

    print("CDR load complete.")


if __name__ == "__main__":
    main()
