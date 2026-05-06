"""Unified document-level annotation schema (CANON Phase 2.1).

Each downstream phase consumes documents in this shape:

    Document
      - pmid, corpus, split
      - title, abstract, text       (text = title + " " + abstract; entity
                                     offsets index into `text`)
      - entities: list[EntityMention]
      - relations: list[Relation]

    EntityMention
      - id (stable within doc, "T1", "T2", ...)
      - span_start, span_end, surface_text
      - entity_type            (corpus-native label, e.g. "ChemicalEntity")
      - semantic_class         (entity_scope class: chemical, disease,
                                gene, variant, species, cell_line, or None)
      - original_code          (raw ID from the corpus; "MESH:" prefix kept)
      - mapped_snomed_id       (filled by 2.2, None at 2.1)
      - mapping_confidence     (filled by 2.2)
      - snomed_active          (filled by 2.2 from Phase 1.7 verified table)
      - non_snomed             (True iff entity_scope marks the type as NER-only;
                                set at 2.1 from the registry)
      - extra: dict            (corpus-specific fields preserved verbatim)

    Relation
      - subject_idx, object_idx  (indices into Document.entities)
      - source_relation_type     (raw label, e.g. "CID", "Negative_Correlation")
      - target_relation          (filled by 2.3)
      - tier                     (filled by 2.3, 1 or 2)
      - target_probability       (filled by 2.3)
      - novelty                  (BioRED only; None elsewhere)
      - extra: dict

JSON Lines is the on-disk format -- one Document per line, easy for streaming
into PyTorch DataLoaders.

Schema is stable across phases: bump SCHEMA_VERSION when fields change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


SCHEMA_VERSION = "2.1.0"


@dataclass
class EntityMention:
    id: str
    span_start: int
    span_end: int
    surface_text: str
    entity_type: str
    semantic_class: Optional[str] = None
    original_code: Optional[str] = None
    mapped_snomed_id: Optional[str] = None
    mapping_confidence: Optional[float] = None
    snomed_active: Optional[bool] = None
    non_snomed: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    subject_idx: int
    object_idx: int
    source_relation_type: str
    target_relation: Optional[str] = None
    tier: Optional[int] = None
    target_probability: Optional[float] = None
    novelty: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    pmid: str
    corpus: str
    split: str
    title: str
    abstract: str
    text: str
    entities: List[EntityMention] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        ents = [EntityMention(**e) for e in data.get("entities", [])]
        rels = [Relation(**r) for r in data.get("relations", [])]
        return cls(
            pmid=data["pmid"],
            corpus=data["corpus"],
            split=data["split"],
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            text=data.get("text", ""),
            entities=ents,
            relations=rels,
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


def write_jsonl(docs: Iterator[Document], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield Document.from_dict(json.loads(line))
