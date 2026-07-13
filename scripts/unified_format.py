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

JSON-LD is the canonical on-disk format. A mechanically derived JSON Lines
cache remains available for efficient streaming into PyTorch DataLoaders.

Schema is stable across phases: bump SCHEMA_VERSION when fields change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


SCHEMA_VERSION = "3.0.0"


@dataclass
class EntityMention:
    id: str
    span_start: int
    span_end: int
    surface_text: str
    entity_type: str
    semantic_class: Optional[str] = None
    ner_type: Optional[str] = None
    original_code: Optional[str] = None
    mapped_snomed_id: Optional[str] = None
    mapping_confidence: Optional[float] = None
    snomed_active: Optional[bool] = None
    non_snomed: bool = False
    source_concept_uri: Optional[str] = None
    normalized_concept_uri: Optional[str] = None
    mapping_property: Optional[str] = None
    umls_stys: List[str] = field(default_factory=list)
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
    confidence: float = 1.0
    target_candidates: List[Dict[str, Any]] = field(default_factory=list)
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

    def to_jsonld(self, context: Any = None) -> Dict[str, Any]:
        """Return the canonical Phase 2 JSON-LD representation.

        The @context is embedded inline by default (the Phase 2.0 context object),
        so every document is self-contained and resolves offline with no hosted
        URL. Instances and vocabulary terms share CANON's W3ID base. Resolution
        is not required at runtime because the context is embedded.
        """
        try:
            import skos_schema
        except ImportError:  # pragma: no cover - package-relative import
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import skos_schema
        if context is None:
            context = skos_schema.build_context()["@context"]
        doc_id = skos_schema.mint_document_uri(self.corpus, self.pmid)
        mentions = []
        mention_ids: List[str] = []
        for em in self.entities:
            mention_id = f"{doc_id}#mention-{em.id}"
            mention_ids.append(mention_id)
            row: Dict[str, Any] = {
                "@id": mention_id,
                "canon:span": {"start": em.span_start, "end": em.span_end},
                "canon:surface": em.surface_text,
                "canon:nerType": _ner_type_uri(em.ner_type or em.semantic_class),
                "canon:sourceEntityType": em.entity_type,
                "canon:sourceSemanticClass": em.semantic_class,
                "canon:sourceConcept": em.source_concept_uri or em.original_code,
                "canon:sourceNotation": em.original_code,
                "canon:confidence": em.mapping_confidence,
                "canon:snomedActive": em.snomed_active,
                "canon:nonSnomed": em.non_snomed,
                "canon:umlsSemanticTypes": list(em.umls_stys),
                "canon:extra": em.extra,
            }
            if em.normalized_concept_uri:
                row["canon:normalizedConcept"] = em.normalized_concept_uri
            if em.mapping_property:
                row["canon:mappingProperty"] = em.mapping_property
            mentions.append({k: v for k, v in row.items() if v is not None})

        relations = []
        for idx, rel in enumerate(self.relations, 1):
            if rel.subject_idx >= len(mention_ids) or rel.object_idx >= len(mention_ids):
                continue
            candidates = rel.target_candidates or rel.extra.get("target_candidates", [])
            jsonld_candidates = [{
                "canon:relation": _relation_uri(c.get("target_relation") or c.get("label")),
                "canon:tier": c.get("tier"),
                "canon:probability": c.get("probability", c.get("score")),
            } for c in candidates]
            relations.append({
                "@id": f"{doc_id}#relation-{idx}",
                "canon:subject": {"@id": mention_ids[rel.subject_idx]},
                "canon:object": {"@id": mention_ids[rel.object_idx]},
                "canon:sourceRelation": rel.source_relation_type,
                "canon:relation": _relation_uri(rel.target_relation),
                "canon:tier": rel.tier,
                "canon:probability": rel.target_probability,
                "canon:confidence": rel.confidence,
                "canon:candidates": jsonld_candidates,
                "canon:novelty": rel.novelty,
                "canon:extra": rel.extra,
            })

        return {
            "@context": context,
            "@id": doc_id,
            "canon:schemaVersion": SCHEMA_VERSION,
            "canon:sourceCorpus": self.corpus,
            "canon:split": self.split,
            "canon:pmid": self.pmid,
            "canon:title": self.title,
            "canon:abstract": self.abstract,
            "canon:text": self.text,
            "canon:mentions": mentions,
            "canon:relations": relations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        entity_fields = EntityMention.__dataclass_fields__
        relation_fields = Relation.__dataclass_fields__
        ents = [EntityMention(**{k: v for k, v in e.items() if k in entity_fields})
                for e in data.get("entities", [])]
        rels = [Relation(**{k: v for k, v in r.items() if k in relation_fields})
                for r in data.get("relations", [])]
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


def write_jsonld_documents(docs: Iterator[Document], root: Path, *, replace: bool = True) -> int:
    """Write one canonical JSON-LD file per document for streaming-friendly storage.

    Replacement removes only prior JSON-LD documents in the target directory,
    preventing deleted documents from surviving into a subsequently derived cache.
    """
    root.mkdir(parents=True, exist_ok=True)
    if replace:
        for old_path in root.glob("*.jsonld"):
            old_path.unlink()
    n = 0
    for doc in docs:
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in doc.pmid)
        safe_corpus = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in doc.corpus)
        path = root / f"{safe_corpus}__{safe_id}.jsonld"
        path.write_text(json.dumps(doc.to_jsonld(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        n += 1
    return n


def write_canonical(docs: Iterator[Document], jsonl_path: Path) -> int:
    """Standard Phase 2 stage output: canonical JSON-LD + derived JSONL cache.

    Writes one JSON-LD document per record under ``<jsonl_path.parent>/jsonld/
    <stem>/`` (the canonical, self-contained form) and derives the fast-loading
    JSONL view at *jsonl_path* (what the next stage reads). Returns the count.
    Single source of the pattern so every stage (2.1/2.2/2.3) is identical.
    """
    jsonld_dir = jsonl_path.parent / "jsonld" / jsonl_path.stem
    write_jsonld_documents(docs, jsonld_dir)
    return derive_jsonl_cache(jsonld_dir, jsonl_path)


def derive_jsonl_cache(jsonld_root: Path, output_path: Path) -> int:
    """Derive the legacy fast JSONL view from canonical JSON-LD documents."""
    def _iter() -> Iterator[Document]:
        for path in sorted(jsonld_root.glob("*.jsonld")):
            data = json.loads(path.read_text(encoding="utf-8"))
            mentions = data.get("canon:mentions", [])
            id_to_idx = {m["@id"]: i for i, m in enumerate(mentions)}
            entities = []
            for i, m in enumerate(mentions):
                span = m.get("canon:span", {})
                entities.append(EntityMention(
                    id=m.get("@id", f"T{i + 1}").rsplit("mention-", 1)[-1],
                    span_start=int(span.get("start", 0)), span_end=int(span.get("end", 0)),
                    surface_text=m.get("canon:surface", ""),
                    entity_type=m.get("canon:sourceEntityType", ""),
                    semantic_class=m.get("canon:sourceSemanticClass"),
                    ner_type=_ner_type_class(m.get("canon:nerType")),
                    original_code=m.get("canon:sourceNotation"),
                    mapped_snomed_id=(m.get("canon:normalizedConcept") or "").rsplit("/", 1)[-1] or None,
                    mapping_confidence=m.get("canon:confidence"),
                    snomed_active=m.get("canon:snomedActive"),
                    non_snomed=bool(m.get("canon:nonSnomed", False)),
                    source_concept_uri=m.get("canon:sourceConcept"),
                    normalized_concept_uri=m.get("canon:normalizedConcept"),
                    mapping_property=m.get("canon:mappingProperty"),
                    umls_stys=list(m.get("canon:umlsSemanticTypes", [])),
                    extra=m.get("canon:extra", {}),
                ))
            relations = []
            for r in data.get("canon:relations", []):
                sid = (r.get("canon:subject") or {}).get("@id")
                oid = (r.get("canon:object") or {}).get("@id")
                if sid not in id_to_idx or oid not in id_to_idx:
                    continue
                relations.append(Relation(
                    subject_idx=id_to_idx[sid], object_idx=id_to_idx[oid],
                    source_relation_type=r.get("canon:sourceRelation", ""),
                    target_relation=_relation_label(r.get("canon:relation")), tier=r.get("canon:tier"),
                    target_probability=r.get("canon:probability"),
                    confidence=float(r.get("canon:confidence", 1.0)),
                    target_candidates=[{
                        "target_relation": _relation_label(c.get("canon:relation")),
                        "tier": c.get("canon:tier"),
                        "probability": c.get("canon:probability"),
                    } for c in r.get("canon:candidates", [])],
                    novelty=r.get("canon:novelty"), extra=r.get("canon:extra", {}),
                ))
            yield Document(
                pmid=str(data.get("canon:pmid", "")), corpus=data.get("canon:sourceCorpus", ""),
                split=data.get("canon:split", ""), title=data.get("canon:title", ""),
                abstract=data.get("canon:abstract", ""), text=data.get("canon:text", ""),
                entities=entities, relations=relations,
                schema_version=data.get("canon:schemaVersion", SCHEMA_VERSION),
            )
    return write_jsonl(_iter(), output_path)


_NER_TYPE_URIS = {
    "clinical_finding": "http://snomed.info/id/404684003",
    "substance": "http://snomed.info/id/105590001",
    "pharmaceutical_product": "http://snomed.info/id/373873005",
    "disease": "http://snomed.info/id/404684003",
    "chemical": "http://snomed.info/id/105590001",
}


def _ner_type_uri(value: Optional[str]) -> Optional[str]:
    return _NER_TYPE_URIS.get(str(value), value) if value is not None else None


def _ner_type_class(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    reverse = {
        "http://snomed.info/id/404684003": "clinical_finding",
        "http://snomed.info/id/105590001": "substance",
        "http://snomed.info/id/373873005": "pharmaceutical_product",
    }
    return reverse.get(value, value)


def _relation_uri(value: Optional[str]) -> Optional[str]:
    if value is None or value.startswith("http://") or value.startswith("https://"):
        return value
    try:
        import skos_schema
    except ImportError:  # pragma: no cover - package-relative import
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import skos_schema
    return skos_schema.mint_canon_relation_uri(value)


def _relation_label(value: Optional[str]) -> Optional[str]:
    return value.rsplit("#", 1)[-1] if value else None
