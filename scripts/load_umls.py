from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tqdm import tqdm

from config import BATCH_SIZE, UMLS_FILES
from utils import copy_rows, ensure_required_files, get_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load UMLS RRF files into umls schema.")
    parser.add_argument("--batch-size", type=int, default=max(BATCH_SIZE, 50000))
    parser.add_argument("--truncate", action="store_true", help="Truncate UMLS tables before loading.")
    parser.add_argument("--english-only", action="store_true", help="Load only LAT='ENG' rows from MRCONSO.")
    parser.add_argument(
        "--sab-filter",
        default="",
        help="Optional comma-separated SAB filter (e.g. MSH,SNOMEDCT_US,HGNC).",
    )
    return parser.parse_args()


def _clean(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned if cleaned else None


def iter_rrf_rows(path: Path, expected_columns: int) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if not row:
                continue
            if len(row) == expected_columns + 1 and row[-1] == "":
                row = row[:-1]
            elif len(row) > expected_columns:
                row = row[:expected_columns]
            elif len(row) < expected_columns:
                row = row + [""] * (expected_columns - len(row))
            yield row


def load_table(
    cursor,
    source_file: Path,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    batch_size: int,
) -> None:
    count = copy_rows(cursor, table, columns, rows, batch_size=batch_size)
    print(f"Loaded {count:,} rows into {table} from {source_file.name}")


def main() -> None:
    args = parse_args()
    ensure_required_files(list(UMLS_FILES.values()))
    sab_filter = {item.strip() for item in args.sab_filter.split(",") if item.strip()}

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET maintenance_work_mem = '2GB'")
            if args.truncate:
                cursor.execute(
                    """
                    TRUNCATE TABLE
                        umls.mrrel,
                        umls.mrdef,
                        umls.mrsty,
                        umls.mrconso
                    CASCADE
                    """
                )

            mrconso_rows = (
                (
                    _clean(r[7]),
                    _clean(r[0]),
                    _clean(r[1]),
                    _clean(r[2]),
                    _clean(r[3]),
                    _clean(r[4]),
                    _clean(r[5]),
                    _clean(r[6]),
                    _clean(r[8]),
                    _clean(r[9]),
                    _clean(r[10]),
                    _clean(r[11]),
                    _clean(r[12]),
                    _clean(r[13]),
                    _clean(r[14]),
                    _clean(r[15]),
                    _clean(r[16]),
                    _clean(r[17]),
                )
                for r in tqdm(iter_rrf_rows(UMLS_FILES["mrconso"], 18), desc="UMLS mrconso", unit="row")
                if (not args.english_only or r[1] == "ENG") and (not sab_filter or r[11] in sab_filter)
            )
            load_table(
                cursor,
                UMLS_FILES["mrconso"],
                "umls.mrconso",
                (
                    "aui",
                    "cui",
                    "lat",
                    "ts",
                    "lui",
                    "stt",
                    "sui",
                    "ispref",
                    "saui",
                    "scui",
                    "sdui",
                    "sab",
                    "tty",
                    "code",
                    "str",
                    "srl",
                    "suppress",
                    "cvf",
                ),
                mrconso_rows,
                args.batch_size,
            )

            mrsty_rows = (
                (_clean(r[0]), _clean(r[1]), _clean(r[2]), _clean(r[3]), _clean(r[4]), _clean(r[5]))
                for r in tqdm(iter_rrf_rows(UMLS_FILES["mrsty"], 6), desc="UMLS mrsty", unit="row")
            )
            load_table(
                cursor,
                UMLS_FILES["mrsty"],
                "umls.mrsty",
                ("cui", "tui", "stn", "sty", "atui", "cvf"),
                mrsty_rows,
                args.batch_size,
            )

            mrdef_rows = (
                (
                    _clean(r[0]),
                    _clean(r[1]),
                    _clean(r[2]),
                    _clean(r[3]),
                    _clean(r[4]),
                    _clean(r[5]),
                    _clean(r[6]),
                    _clean(r[7]),
                )
                for r in tqdm(iter_rrf_rows(UMLS_FILES["mrdef"], 8), desc="UMLS mrdef", unit="row")
                if (not sab_filter or r[4] in sab_filter)
            )
            load_table(
                cursor,
                UMLS_FILES["mrdef"],
                "umls.mrdef",
                ("cui", "aui", "atui", "satui", "sab", "definition", "suppress", "cvf"),
                mrdef_rows,
                args.batch_size,
            )

            mrrel_rows = (
                (
                    _clean(r[0]),
                    _clean(r[1]),
                    _clean(r[2]),
                    _clean(r[3]),
                    _clean(r[4]),
                    _clean(r[5]),
                    _clean(r[6]),
                    _clean(r[7]),
                    _clean(r[8]),
                    _clean(r[9]),
                    _clean(r[10]),
                    _clean(r[11]),
                    _clean(r[12]),
                    _clean(r[13]),
                    _clean(r[14]),
                    _clean(r[15]),
                )
                for r in tqdm(iter_rrf_rows(UMLS_FILES["mrrel"], 16), desc="UMLS mrrel", unit="row")
                if (not sab_filter or r[10] in sab_filter)
            )
            load_table(
                cursor,
                UMLS_FILES["mrrel"],
                "umls.mrrel",
                (
                    "cui1",
                    "aui1",
                    "stype1",
                    "rel",
                    "cui2",
                    "aui2",
                    "stype2",
                    "rela",
                    "rui",
                    "srui",
                    "sab",
                    "sl",
                    "rg",
                    "dir",
                    "suppress",
                    "cvf",
                ),
                mrrel_rows,
                args.batch_size,
            )

        conn.commit()

    print("UMLS load complete.")


if __name__ == "__main__":
    main()
