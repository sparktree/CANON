from __future__ import annotations

import csv
import io
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import psycopg2
from psycopg2.extensions import connection as PGConnection

from config import DB_CONFIG, PostgresConfig


def get_connection(db_config: PostgresConfig = DB_CONFIG) -> PGConnection:
    kwargs = asdict(db_config)
    if not kwargs["password"]:
        kwargs.pop("password")
    return psycopg2.connect(**kwargs)


def ensure_required_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Required files are missing:\n{formatted}")


def parse_pubtator(path: Path) -> Iterator[dict]:
    current: dict | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if current is not None:
                    yield current
                    current = None
                continue

            if "|t|" in line:
                pmid, title = line.split("|t|", 1)
                current = {
                    "pmid": int(pmid),
                    "title": title,
                    "abstract": "",
                    "entities": [],
                    "relations": [],
                }
                continue

            if "|a|" in line:
                if current is None:
                    raise ValueError(f"Found abstract before title in {path}")
                _, abstract = line.split("|a|", 1)
                current["abstract"] = abstract
                continue

            if current is None:
                raise ValueError(f"Found entity/relation before title in {path}")

            fields = line.split("\t")
            if len(fields) < 4:
                continue

            if len(fields) >= 6 and fields[1].isdigit():
                current["entities"].append(
                    {
                        "pmid": int(fields[0]),
                        "start_offset": int(fields[1]),
                        "end_offset": int(fields[2]),
                        "mention": fields[3],
                        "entity_type": fields[4],
                        "identifier_raw": fields[5],
                        "extra_info": fields[6] if len(fields) > 6 else None,
                    }
                )
            else:
                current["relations"].append(
                    {
                        "pmid": int(fields[0]),
                        "relation_type": fields[1],
                        "subject_id": fields[2],
                        "object_id": fields[3],
                        "novelty": fields[4] if len(fields) > 4 else "",
                    }
                )

    if current is not None:
        yield current


def normalize_mesh_id(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned in {"", "-1", "-", "NULL", "null"}:
        return None
    return cleaned


def split_composite_ids(value: str | None, delimiter: str = ",") -> list[str]:
    if value is None:
        return []
    items = [token.strip() for token in value.split(delimiter)]
    return [token for token in items if token]


def yyyymmdd_to_date(value: str) -> date:
    cleaned = value.strip()
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise ValueError(f"Invalid YYYYMMDD date: {value}")
    return date(int(cleaned[0:4]), int(cleaned[4:6]), int(cleaned[6:8]))


def parse_bool_flag(value: str) -> bool:
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "t"}:
        return True
    if cleaned in {"0", "false", "f"}:
        return False
    raise ValueError(f"Invalid boolean flag: {value}")


def tsv_dict_reader(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            yield row


def _encode_copy_value(value: object) -> str:
    if value is None:
        return r"\N"
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        escaped = [
            item.replace("\\", "\\\\").replace('"', '\\"') if item is not None else ""
            for item in value
        ]
        return "{" + ",".join(f'"{item}"' for item in escaped) + "}"
    text = str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def copy_rows(
    cursor,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    batch_size: int = 50000,
) -> int:
    buffer = io.StringIO()
    loaded = 0

    def flush() -> None:
        nonlocal buffer
        buffer.seek(0)
        copy_query = f"COPY {table} ({','.join(columns)}) FROM STDIN WITH NULL '\\N' DELIMITER '\t'"
        cursor.copy_expert(copy_query, buffer)
        buffer = io.StringIO()

    for row in rows:
        buffer.write("\t".join(_encode_copy_value(value) for value in row))
        buffer.write("\n")
        loaded += 1
        if loaded % batch_size == 0:
            flush()

    if buffer.tell() > 0:
        flush()

    return loaded
