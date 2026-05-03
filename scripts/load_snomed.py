from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Sequence

from tqdm import tqdm

from config import BATCH_SIZE, SNOMED_FILES
from utils import copy_rows, ensure_required_files, get_connection, parse_bool_flag, tsv_dict_reader, yyyymmdd_to_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load SNOMED Snapshot files into snomed schema.")
    parser.add_argument("--batch-size", type=int, default=max(BATCH_SIZE, 50000))
    parser.add_argument("--truncate", action="store_true", help="Truncate SNOMED tables before loading.")
    return parser.parse_args()


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.strip()
    return int(cleaned) if cleaned else None


def _to_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def load_table(
    cursor,
    source_file: Path,
    table: str,
    columns: Sequence[str],
    transform: Callable[[dict[str, str]], Sequence[object]],
    batch_size: int,
) -> None:
    rows: Iterable[Sequence[object]] = (
        transform(row) for row in tqdm(tsv_dict_reader(source_file), desc=f"SNOMED {table}", unit="row")
    )
    count = copy_rows(cursor, table, columns, rows, batch_size=batch_size)
    print(f"Loaded {count:,} rows into {table} from {source_file.name}")


def main() -> None:
    args = parse_args()
    ensure_required_files(list(SNOMED_FILES.values()))

    with get_connection() as conn:
        with conn.cursor() as cursor:
            if args.truncate:
                cursor.execute(
                    """
                    TRUNCATE TABLE
                        snomed.simple_map,
                        snomed.extended_map,
                        snomed.stated_relationships,
                        snomed.relationships,
                        snomed.text_definitions,
                        snomed.descriptions,
                        snomed.concepts
                    CASCADE
                    """
                )

            load_table(
                cursor,
                SNOMED_FILES["concepts"],
                "snomed.concepts",
                ("id", "effective_time", "active", "module_id", "definition_status_id"),
                lambda r: (
                    _to_int(r["id"]),
                    yyyymmdd_to_date(r["effectiveTime"]),
                    parse_bool_flag(r["active"]),
                    _to_int(r["moduleId"]),
                    _to_int(r["definitionStatusId"]),
                ),
                args.batch_size,
            )

            load_table(
                cursor,
                SNOMED_FILES["descriptions"],
                "snomed.descriptions",
                (
                    "id",
                    "effective_time",
                    "active",
                    "module_id",
                    "concept_id",
                    "language_code",
                    "type_id",
                    "term",
                    "case_significance_id",
                ),
                lambda r: (
                    _to_int(r["id"]),
                    yyyymmdd_to_date(r["effectiveTime"]),
                    parse_bool_flag(r["active"]),
                    _to_int(r["moduleId"]),
                    _to_int(r["conceptId"]),
                    _to_text(r["languageCode"]),
                    _to_int(r["typeId"]),
                    _to_text(r["term"]),
                    _to_int(r["caseSignificanceId"]),
                ),
                args.batch_size,
            )

            load_table(
                cursor,
                SNOMED_FILES["text_definitions"],
                "snomed.text_definitions",
                (
                    "id",
                    "effective_time",
                    "active",
                    "module_id",
                    "concept_id",
                    "language_code",
                    "type_id",
                    "term",
                    "case_significance_id",
                ),
                lambda r: (
                    _to_int(r["id"]),
                    yyyymmdd_to_date(r["effectiveTime"]),
                    parse_bool_flag(r["active"]),
                    _to_int(r["moduleId"]),
                    _to_int(r["conceptId"]),
                    _to_text(r["languageCode"]),
                    _to_int(r["typeId"]),
                    _to_text(r["term"]),
                    _to_int(r["caseSignificanceId"]),
                ),
                args.batch_size,
            )

            load_table(
                cursor,
                SNOMED_FILES["relationships"],
                "snomed.relationships",
                (
                    "id",
                    "effective_time",
                    "active",
                    "module_id",
                    "source_id",
                    "destination_id",
                    "relationship_group",
                    "type_id",
                    "characteristic_type_id",
                    "modifier_id",
                ),
                lambda r: (
                    _to_int(r["id"]),
                    yyyymmdd_to_date(r["effectiveTime"]),
                    parse_bool_flag(r["active"]),
                    _to_int(r["moduleId"]),
                    _to_int(r["sourceId"]),
                    _to_int(r["destinationId"]),
                    _to_int(r["relationshipGroup"]),
                    _to_int(r["typeId"]),
                    _to_int(r["characteristicTypeId"]),
                    _to_int(r["modifierId"]),
                ),
                args.batch_size,
            )

            load_table(
                cursor,
                SNOMED_FILES["stated_relationships"],
                "snomed.stated_relationships",
                (
                    "id",
                    "effective_time",
                    "active",
                    "module_id",
                    "source_id",
                    "destination_id",
                    "relationship_group",
                    "type_id",
                    "characteristic_type_id",
                    "modifier_id",
                ),
                lambda r: (
                    _to_int(r["id"]),
                    yyyymmdd_to_date(r["effectiveTime"]),
                    parse_bool_flag(r["active"]),
                    _to_int(r["moduleId"]),
                    _to_int(r["sourceId"]),
                    _to_int(r["destinationId"]),
                    _to_int(r["relationshipGroup"]),
                    _to_int(r["typeId"]),
                    _to_int(r["characteristicTypeId"]),
                    _to_int(r["modifierId"]),
                ),
                args.batch_size,
            )

            load_table(
                cursor,
                SNOMED_FILES["extended_map"],
                "snomed.extended_map",
                (
                    "id",
                    "effective_time",
                    "active",
                    "module_id",
                    "refset_id",
                    "referenced_component_id",
                    "map_group",
                    "map_priority",
                    "map_rule",
                    "map_advice",
                    "map_target",
                    "correlation_id",
                    "map_category_id",
                ),
                lambda r: (
                    _to_text(r["id"]),
                    yyyymmdd_to_date(r["effectiveTime"]),
                    parse_bool_flag(r["active"]),
                    _to_int(r["moduleId"]),
                    _to_int(r["refsetId"]),
                    _to_int(r["referencedComponentId"]),
                    _to_int(r["mapGroup"]),
                    _to_int(r["mapPriority"]),
                    _to_text(r["mapRule"]),
                    _to_text(r["mapAdvice"]),
                    _to_text(r["mapTarget"]),
                    _to_int(r["correlationId"]),
                    _to_int(r["mapCategoryId"]),
                ),
                args.batch_size,
            )

            load_table(
                cursor,
                SNOMED_FILES["simple_map"],
                "snomed.simple_map",
                (
                    "id",
                    "effective_time",
                    "active",
                    "module_id",
                    "refset_id",
                    "referenced_component_id",
                    "map_target",
                ),
                lambda r: (
                    _to_text(r["id"]),
                    yyyymmdd_to_date(r["effectiveTime"]),
                    parse_bool_flag(r["active"]),
                    _to_int(r["moduleId"]),
                    _to_int(r["refsetId"]),
                    _to_int(r["referencedComponentId"]),
                    _to_text(r["mapTarget"]),
                ),
                args.batch_size,
            )

        conn.commit()

    print("SNOMED load complete.")


if __name__ == "__main__":
    main()
