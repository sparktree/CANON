"""Active-pipeline shared utilities.

Only :func:`parse_pubtator` is consumed by the in-memory CANON pipeline
(Phase 1.x). Legacy SQL helpers live in ``scripts/legacy_sql/utils.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def choose_torch_device(preferred: str = "auto"):
    """Return a torch device, preferring CUDA, then Apple MPS, then CPU."""
    import torch

    requested = (preferred or "auto").lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
