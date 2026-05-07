"""Per-corpus PubTator / BioC XML -> unified-format converter (CANON Phase 2.1).

Drives the conversion of every available corpus split into JSON Lines
matching ``unified_format.Document``. SNOMED mapping slots and unified
relation slots are intentionally left empty here -- Phases 2.2 and 2.3
populate them.

Outputs (under outputs/phase2/unified/):
    BioRED/       train.jsonl, dev.jsonl, test.jsonl
    BC5CDR/       train.jsonl, dev.jsonl, test.jsonl
    NCBI_Disease/ ...   (optional; written only if PubTator files exist)
    NLM-Chem/     ...   (optional; BioC XML reader; written only if files exist)

Plus outputs/phase2/conversion_summary.json with per-corpus counts and a
flag for any (corpus, entity_type) pair not registered in entity_scope.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from config import BIORED_FILES, CDR_FILES, DATA_ROOT, REPO_ROOT, relative_to_repo
    from utils import parse_pubtator
    import entity_scope
    from unified_format import Document, EntityMention, Relation, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import BIORED_FILES, CDR_FILES, DATA_ROOT, REPO_ROOT, relative_to_repo
    from utils import parse_pubtator
    import entity_scope
    from unified_format import Document, EntityMention, Relation, write_jsonl


OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2" / "unified"
SUMMARY_JSON = REPO_ROOT / "outputs" / "phase2" / "conversion_summary.json"


# ---------------------------------------------------------------------------
# Per-corpus file resolution
# ---------------------------------------------------------------------------
def _ncbi_disease_files() -> Dict[str, Path]:
    base = DATA_ROOT / "NCBI_Disease"
    return {
        "train": base / "NCBItrainset_corpus.txt",
        "dev": base / "NCBIdevelopset_corpus.txt",
        "test": base / "NCBItestset_corpus.txt",
    }


def _nlm_chem_files() -> Dict[str, Path]:
    base = DATA_ROOT / "NLM-Chem"
    return {
        "train": base / "BC7T2-NLMChem-corpus-train.BioC.xml",
        "dev": base / "BC7T2-NLMChem-corpus-dev.BioC.xml",
        "test": base / "BC7T2-NLMChem-corpus-test.BioC.xml",
    }


def _existing(files: Dict[str, Path]) -> Dict[str, Path]:
    return {k: v for k, v in files.items() if v.exists()}


# ---------------------------------------------------------------------------
# PubTator entity / relation -> unified
# ---------------------------------------------------------------------------
def _pubtator_doc_to_unified(
    raw: dict,
    corpus: str,
    split: str,
    unregistered_counter: Counter,
) -> Document:
    """Build a unified Document from a parse_pubtator() raw dict."""
    title = raw.get("title", "") or ""
    abstract = raw.get("abstract", "") or ""
    # PubTator entity offsets index into `title + " " + abstract` (single space).
    text = title if not abstract else (title + " " + abstract)

    entities: List[EntityMention] = []
    code_to_first_idx: Dict[str, int] = {}
    for idx, ent in enumerate(raw.get("entities", [])):
        entity_type = ent.get("entity_type", "") or ""
        spec = entity_scope.lookup(corpus, entity_type)
        if spec is None and entity_type:
            unregistered_counter[(corpus, entity_type)] += 1
        semantic_class = spec.semantic_class if spec else None
        non_snomed = bool(spec and not spec.snomed_normalized)
        original_code = (ent.get("identifier_raw") or "").strip() or None

        em = EntityMention(
            id=f"T{idx + 1}",
            span_start=int(ent.get("start_offset", 0)),
            span_end=int(ent.get("end_offset", 0)),
            surface_text=ent.get("mention", "") or "",
            entity_type=entity_type,
            semantic_class=semantic_class,
            original_code=original_code,
            non_snomed=non_snomed,
            extra={"pubtator_extra": ent.get("extra_info")} if ent.get("extra_info") else {},
        )
        entities.append(em)
        if original_code and original_code not in code_to_first_idx:
            code_to_first_idx[original_code] = idx

    relations: List[Relation] = []
    for rel in raw.get("relations", []):
        subj_id = (rel.get("subject_id") or "").strip()
        obj_id = (rel.get("object_id") or "").strip()
        subj_idx = _resolve_relation_arg(subj_id, code_to_first_idx)
        obj_idx = _resolve_relation_arg(obj_id, code_to_first_idx)
        if subj_idx is None or obj_idx is None:
            # Skip dangling relations; record them in `extra` for audit.
            continue
        relations.append(
            Relation(
                subject_idx=subj_idx,
                object_idx=obj_idx,
                source_relation_type=rel.get("relation_type", "") or "",
                novelty=(rel.get("novelty") or None),
            )
        )

    return Document(
        pmid=str(raw.get("pmid", "")),
        corpus=corpus,
        split=split,
        title=title,
        abstract=abstract,
        text=text,
        entities=entities,
        relations=relations,
    )


def _resolve_relation_arg(arg: str, code_to_first_idx: Dict[str, int]) -> Optional[int]:
    """Match a relation argument string to an entity index.

    Tries an exact match first (so composite IDs like 'D003922,D003925' bind
    to a composite-coded entity if one exists), then falls back to any
    component of a comma-separated arg.
    """
    if not arg:
        return None
    if arg in code_to_first_idx:
        return code_to_first_idx[arg]
    for token in (t.strip() for t in arg.split(",")):
        if token in code_to_first_idx:
            return code_to_first_idx[token]
    return None


# ---------------------------------------------------------------------------
# BioC XML reader (used for NLM-Chem)
# ---------------------------------------------------------------------------
def _iter_bioc_documents(path: Path) -> Iterator[dict]:
    """Yield raw doc dicts mirroring the parse_pubtator() shape."""
    tree = ET.parse(str(path))
    root = tree.getroot()
    for doc_elem in root.findall("document"):
        pmid_elem = doc_elem.find("id")
        pmid = pmid_elem.text.strip() if (pmid_elem is not None and pmid_elem.text) else ""

        title = ""
        abstract_parts: List[str] = []
        passages = doc_elem.findall("passage")
        passage_offsets: List[Tuple[int, str]] = []  # (offset, text)
        annotations: List[Tuple[int, int, str, str, dict]] = []  # (start, end, mention, ann_id, infons)

        for psg in passages:
            offset_elem = psg.find("offset")
            text_elem = psg.find("text")
            psg_offset = int(offset_elem.text) if (offset_elem is not None and offset_elem.text) else 0
            psg_text = text_elem.text or "" if text_elem is not None else ""
            psg_type = ""
            for infon in psg.findall("infon"):
                if infon.get("key") == "type":
                    psg_type = (infon.text or "").lower().strip()
                    break
            passage_offsets.append((psg_offset, psg_text))
            if psg_type == "title" and not title:
                title = psg_text
            elif psg_type in ("abstract", "paragraph", "section", "") and psg_text:
                abstract_parts.append(psg_text)

            for ann in psg.findall("annotation"):
                ann_id = ann.get("id", "")
                infons = {
                    inf.get("key", ""): (inf.text or "")
                    for inf in ann.findall("infon")
                }
                loc = ann.find("location")
                if loc is None:
                    continue
                start = int(loc.get("offset", "0"))
                length = int(loc.get("length", "0"))
                txt = (ann.find("text").text or "") if ann.find("text") is not None else ""
                annotations.append((start, start + length, txt, ann_id, infons))

        abstract = " ".join(p for p in abstract_parts if p)
        # BioC offsets are absolute over the original full text. We
        # reconstruct a single text string in passage order to keep entity
        # offsets coherent with the PubTator path.
        full_text = title if not abstract else (title + " " + abstract)

        # Map BioC absolute offsets onto the reconstructed full_text. If the
        # passage layout matches title+" "+abstract this is a 1:1 mapping;
        # otherwise we project per-passage.
        offset_remap: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cursor = 0
        for psg_offset, psg_text in passage_offsets:
            psg_len = len(psg_text)
            offset_remap[(psg_offset, psg_offset + psg_len)] = (cursor, cursor + psg_len)
            cursor += psg_len + 1  # +1 for the inserted space separator

        def _project(start: int, end: int) -> Tuple[int, int]:
            for (po, pe), (co, _) in offset_remap.items():
                if po <= start <= pe:
                    return (co + (start - po), co + (end - po))
            return (start, end)

        entities = []
        ann_id_to_idx: Dict[str, int] = {}
        for i, (s, e, mention, ann_id, infons) in enumerate(annotations):
            ps, pe = _project(s, e)
            etype = (
                infons.get("type")
                or infons.get("ner_type")
                or infons.get("entity_type")
                or ""
            ).strip()
            # Preferred identifier keys: MESH > identifier > NCBI gene > generic id.
            ident = (
                infons.get("MESH")
                or infons.get("identifier")
                or infons.get("NCBI Gene")
                or infons.get("identifier_raw")
                or ""
            ).strip()
            entities.append(
                {
                    "pmid": pmid,
                    "start_offset": ps,
                    "end_offset": pe,
                    "mention": mention,
                    "entity_type": etype,
                    "identifier_raw": ident,
                    "extra_info": None,
                }
            )
            if ann_id:
                ann_id_to_idx[ann_id] = i

        relations = []
        for rel in doc_elem.findall("relation"):
            rinfons = {
                inf.get("key", ""): (inf.text or "")
                for inf in rel.findall("infon")
            }
            rtype = (rinfons.get("type") or rinfons.get("relation") or "").strip()
            nodes = rel.findall("node")
            subj_ref = None
            obj_ref = None
            for n in nodes:
                role = (n.get("role") or "").lower()
                ref = n.get("refid") or ""
                if role in ("subject", "arg1") or subj_ref is None:
                    if subj_ref is None:
                        subj_ref = ref
                if role in ("object", "arg2") or obj_ref is None:
                    if subj_ref is not None and ref != subj_ref and obj_ref is None:
                        obj_ref = ref
            if subj_ref and obj_ref:
                # Translate ref ids to original_code via the entity table; the
                # PubTator path expects subject_id / object_id to be codes,
                # but BioC uses annotation ids. We pass the codes through and
                # let _resolve_relation_arg fall back to component matching.
                subj_idx = ann_id_to_idx.get(subj_ref)
                obj_idx = ann_id_to_idx.get(obj_ref)
                subj_code = entities[subj_idx]["identifier_raw"] if subj_idx is not None else ""
                obj_code = entities[obj_idx]["identifier_raw"] if obj_idx is not None else ""
                relations.append(
                    {
                        "pmid": pmid,
                        "relation_type": rtype,
                        "subject_id": subj_code or subj_ref,
                        "object_id": obj_code or obj_ref,
                        "novelty": "",
                    }
                )

        yield {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "entities": entities,
            "relations": relations,
        }


# ---------------------------------------------------------------------------
# Per-corpus drivers
# ---------------------------------------------------------------------------
def _convert_split(
    corpus: str,
    split: str,
    path: Path,
    reader,  # callable: Path -> Iterator[dict]
    unregistered_counter: Counter,
) -> Tuple[Path, int, int, int]:
    docs: List[Document] = []
    n_ent = 0
    n_rel = 0
    for raw in reader(path):
        doc = _pubtator_doc_to_unified(raw, corpus, split, unregistered_counter)
        docs.append(doc)
        n_ent += len(doc.entities)
        n_rel += len(doc.relations)

    out_path = OUTPUT_DIR / corpus / f"{split}.jsonl"
    written = write_jsonl(iter(docs), out_path)
    return out_path, written, n_ent, n_rel


def convert_all(verbose: bool = True) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {"schema_version": "2.1.0", "corpora": {}, "unregistered_entity_types": []}
    unregistered_counter: Counter = Counter()

    plan = [
        ("BioRED", _existing(BIORED_FILES), parse_pubtator),
        ("BC5CDR", _existing(CDR_FILES), parse_pubtator),
        ("NCBI_Disease", _existing(_ncbi_disease_files()), parse_pubtator),
        ("NLM-Chem", _existing(_nlm_chem_files()), _iter_bioc_documents),
    ]

    for corpus, files, reader in plan:
        if not files:
            if verbose:
                print(f"[2.1] {corpus}: no input files present, skipping")
            summary["corpora"][corpus] = {"status": "absent"}
            continue

        per_split = {}
        for split, path in files.items():
            out_path, n_docs, n_ent, n_rel = _convert_split(
                corpus, split, path, reader, unregistered_counter
            )
            per_split[split] = {
                "documents": n_docs,
                "entities": n_ent,
                "relations": n_rel,
                "input": relative_to_repo(path),
                "output": relative_to_repo(out_path),
            }
            if verbose:
                print(
                    f"[2.1] {corpus:<14s} {split:<5s} -> "
                    f"{n_docs:>5,d} docs  {n_ent:>7,d} ents  {n_rel:>5,d} rels  "
                    f"({out_path.name})"
                )
        summary["corpora"][corpus] = {"status": "converted", "splits": per_split}

    summary["unregistered_entity_types"] = sorted(
        [
            {"corpus": c, "entity_type": t, "mentions": n}
            for (c, t), n in unregistered_counter.items()
        ],
        key=lambda r: -r["mentions"],
    )

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[2.1] summary -> {SUMMARY_JSON}")
        if summary["unregistered_entity_types"]:
            print("[2.1] WARNING: unregistered (corpus, entity_type) pairs encountered:")
            for r in summary["unregistered_entity_types"]:
                print(f"    {r['corpus']} / {r['entity_type']}  mentions={r['mentions']}")

    return summary


if __name__ == "__main__":
    convert_all(verbose=True)
