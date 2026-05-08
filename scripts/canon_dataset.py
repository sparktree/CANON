"""Shared dataloader for CANON Phase 3 training (3.3, 3.4, 3.6).

Streams unified-format documents from outputs/phase2/splits/*.jsonl, tokenizes
with a HuggingFace fast tokenizer, and produces per-document feature dicts
that the multi-task model consumes.

Outputs per document
--------------------
* input_ids, attention_mask, offset_mapping  (token tensors, length T)
* bio_labels        token-aligned BIO label IDs over 13 NER classes
                    (O + B-/I- x {cell_line, chemical, disease, gene, species, variant})
* entity_token_spans list of (start_tok, end_tok, semantic_class) per entity
                    that survived truncation
* norm_targets      soft target distributions per in-scope entity
                    (List[Dict[snomed_id -> probability]])
* norm_entity_idx   index into entity_token_spans for each norm target
* pair_indices      (P, 2) entity index pairs (i, j)
* pair_labels       (P,)  class id over the 13-class relation space
* pair_weights      (P,)  target_probability for golds, 1.0 for sampled negatives

The dataset is iterable; collate_docs pads tokens to the longest sequence in
the batch and stacks variable-length per-doc lists into nested Python lists
(the model unpacks them per element rather than relying on padded tensors --
each doc has a different number of entities/pairs).

Used by:
  scripts/train_stage1.py
  scripts/train_stage2.py
  scripts/train_stage3.py
  scripts/csp_solver.py (feature-extraction path)
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset

try:
    from unified_format import Document
    from entity_scope import SNOMED_NER_CLASSES, NON_SNOMED_NER_CLASSES
    from relation_schema import ALL_TARGET_RELATIONS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unified_format import Document
    from entity_scope import SNOMED_NER_CLASSES, NON_SNOMED_NER_CLASSES
    from relation_schema import ALL_TARGET_RELATIONS


# Stable BIO tag inventory (sorted alphabetically). 13 tags total.
SEMANTIC_CLASSES: Tuple[str, ...] = tuple(
    sorted(set(SNOMED_NER_CLASSES) | set(NON_SNOMED_NER_CLASSES))
)
BIO_LABELS: List[str] = ["O"]
for _cls in SEMANTIC_CLASSES:
    BIO_LABELS.append(f"B-{_cls}")
    BIO_LABELS.append(f"I-{_cls}")
BIO_LABEL_TO_ID: Dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
BIO_ID_TO_LABEL: Dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}
NUM_BIO_LABELS: int = len(BIO_LABELS)  # 13

# Stable relation-class inventory. 12 unified + 1 no-relation = 13.
RELATION_LABELS: List[str] = sorted(ALL_TARGET_RELATIONS) + ["no-relation"]
RELATION_LABEL_TO_ID: Dict[str, int] = {label: i for i, label in enumerate(RELATION_LABELS)}
NO_RELATION_ID: int = RELATION_LABEL_TO_ID["no-relation"]
NUM_RELATION_LABELS: int = len(RELATION_LABELS)  # 13

# Stable semantic-class inventory + "none" sentinel for relation-head type-pair embedding.
SEMANTIC_CLASS_LIST: List[str] = list(SEMANTIC_CLASSES) + ["none"]
SEMANTIC_CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(SEMANTIC_CLASS_LIST)}
NUM_SEMANTIC_CLASSES: int = len(SEMANTIC_CLASS_LIST)


@dataclass
class DocFeatures:
    pmid: str
    corpus: str
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    offset_mapping: torch.LongTensor
    bio_labels: torch.LongTensor
    entity_token_spans: List[Tuple[int, int, str]]    # (start_tok, end_tok_excl, semantic_class)
    entity_original: List[Dict]                        # echoes of each kept EntityMention as dict
    norm_targets: List[Dict[str, float]]
    norm_entity_idx: List[int]
    norm_weights: List[float]
    pair_indices: torch.LongTensor                     # (P, 2)
    pair_labels: torch.LongTensor                      # (P,)
    pair_weights: torch.FloatTensor                    # (P,)
    pair_semantic_classes: torch.LongTensor            # (P, 2) -> SEMANTIC_CLASS_TO_ID
    raw_doc: Dict = field(default_factory=dict)


def _normalize_mesh_code(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    if code.startswith("MESH:"):
        return code[len("MESH:") :]
    return code


def load_soft_lookup(path: Path) -> Dict[str, List[Dict]]:
    """Load Phase 2.4 soft mapping lookup (mesh_id -> ranked candidates)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_bio_targets(
    entities: Sequence[Dict],
    offset_mapping: Sequence[Tuple[int, int]],
    attention_mask: Sequence[int],
) -> Tuple[List[int], List[Tuple[int, int, str]], List[Dict]]:
    """Align entity character spans onto WordPiece tokens.

    Returns
    -------
    bio_labels : list[int] of length T (special tokens / pads get O = 0; we
                 do not use -100 because torchcrf does not accept it)
    entity_token_spans : list of (start_tok, end_tok_excl, semantic_class) for
                         every entity that has at least one overlapping content
                         token within the truncation window. Order preserved.
    surviving_entities : the EntityMention dicts that fit the window.
    """
    T = len(offset_mapping)
    bio = [0] * T  # all O initially
    spans: List[Tuple[int, int, str]] = []
    survivors: List[Dict] = []

    # Tokens with offset (0, 0) are special tokens (CLS/SEP) -- skip them.
    # The HF fast tokenizer also emits subword pieces with offsets pointing
    # into the surface text.
    for ent in entities:
        ent_start = int(ent["span_start"])
        ent_end = int(ent["span_end"])
        sclass = ent.get("semantic_class")
        if not sclass or sclass not in SEMANTIC_CLASSES:
            # Entities with no recognised semantic class don't get a BIO tag.
            continue

        first_tok = -1
        last_tok = -1
        for t, (s, e) in enumerate(offset_mapping):
            if attention_mask[t] == 0:
                continue
            if s == 0 and e == 0:
                continue
            # Token overlaps the entity span if [s, e) ∩ [ent_start, ent_end) != ∅
            if e > ent_start and s < ent_end:
                if first_tok < 0:
                    first_tok = t
                last_tok = t

        if first_tok < 0:
            # Entity fell outside the truncation window.
            continue

        b_label = BIO_LABEL_TO_ID[f"B-{sclass}"]
        i_label = BIO_LABEL_TO_ID[f"I-{sclass}"]
        bio[first_tok] = b_label
        for t in range(first_tok + 1, last_tok + 1):
            bio[t] = i_label

        spans.append((first_tok, last_tok + 1, sclass))
        survivors.append(ent)

    return bio, spans, survivors


def enumerate_pairs(
    surviving_entities: Sequence[Dict],
    raw_relations: Sequence[Dict],
    *,
    survivor_index: Dict[int, int],
    neg_ratio: float = 2.0,
    max_pairs: int = 64,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[float]]:
    """Build entity-pair training tuples for the relation head.

    survivor_index maps original entity index -> index into surviving_entities.

    Positives are kept whenever both endpoints survived truncation.
    Negatives are sampled from the remaining ordered pairs at neg_ratio x #positives,
    capped so the total never exceeds max_pairs (positives always retained).
    """
    rng = rng or random.Random(0)
    n = len(surviving_entities)
    pos_pairs: Dict[Tuple[int, int], Tuple[int, float]] = {}

    for rel in raw_relations:
        s_orig = rel["subject_idx"]
        o_orig = rel["object_idx"]
        if s_orig not in survivor_index or o_orig not in survivor_index:
            continue
        s = survivor_index[s_orig]
        o = survivor_index[o_orig]
        if s == o:
            continue
        target = rel.get("target_relation")
        if target is None or target not in RELATION_LABEL_TO_ID:
            continue
        prob = rel.get("target_probability") or 0.0
        if prob <= 0.0:
            prob = 1.0
        # Keep highest-probability assignment if duplicates land on the same pair.
        prev = pos_pairs.get((s, o))
        if prev is None or prob > prev[1]:
            pos_pairs[(s, o)] = (RELATION_LABEL_TO_ID[target], float(prob))

    indices: List[Tuple[int, int]] = []
    labels: List[int] = []
    weights: List[float] = []

    for (s, o), (lab, w) in pos_pairs.items():
        indices.append((s, o))
        labels.append(lab)
        weights.append(w)

    desired_neg = int(round(neg_ratio * len(pos_pairs)))
    desired_total = min(max_pairs, len(pos_pairs) + max(desired_neg, 0))
    desired_neg = max(0, desired_total - len(pos_pairs))

    if n >= 2 and desired_neg > 0:
        candidate_negs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (i, j) in pos_pairs:
                    continue
                candidate_negs.append((i, j))
        rng.shuffle(candidate_negs)
        for pair in candidate_negs[:desired_neg]:
            indices.append(pair)
            labels.append(NO_RELATION_ID)
            weights.append(1.0)

    return indices, labels, weights


class CanonDocDataset(IterableDataset):
    """Stream Documents from a JSONL split and yield DocFeatures.

    Parameters
    ----------
    jsonl_path : Path to outputs/phase2/splits/{train,dev,test}.jsonl.
    tokenizer  : HuggingFace fast tokenizer (e.g. AutoTokenizer.from_pretrained(...)).
    soft_lookup : Optional Dict[str, List[dict]] from load_soft_lookup. If None,
                  norm targets fall back to a delta on entity.mapped_snomed_id.
    max_length : token cap (default 512).
    max_docs   : optional cap on number of docs to yield (smoke-test mode).
    neg_ratio  : negative-pair sampling ratio (default 2.0 = 2x positives).
    max_pairs  : per-document cap on total entity pairs (default 64).
    seed       : RNG seed for deterministic negative sampling.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        soft_lookup: Optional[Dict[str, List[Dict]]] = None,
        *,
        max_length: int = 512,
        max_docs: Optional[int] = None,
        neg_ratio: float = 2.0,
        max_pairs: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.soft_lookup = soft_lookup or {}
        self.max_length = max_length
        self.max_docs = max_docs
        self.neg_ratio = neg_ratio
        self.max_pairs = max_pairs
        self.seed = seed

    def __iter__(self) -> Iterator[DocFeatures]:
        rng = random.Random(self.seed)
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if self.max_docs is not None and i >= self.max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                features = self._featurize(raw, rng)
                if features is None:
                    continue
                yield features

    def _featurize(self, raw: Dict, rng: random.Random) -> Optional[DocFeatures]:
        text = raw.get("text") or ((raw.get("title", "") + " " + raw.get("abstract", "")).strip())
        if not text:
            return None
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_attention_mask=True,
            padding=False,
        )
        offsets = encoded["offset_mapping"]
        attention = encoded["attention_mask"]
        input_ids = encoded["input_ids"]

        bio, spans, survivors = build_bio_targets(raw.get("entities", []), offsets, attention)

        # Map original-entity-index -> survivor-index (for relation enumeration).
        survivor_index: Dict[int, int] = {}
        kept = 0
        for orig_idx, ent in enumerate(raw.get("entities", [])):
            sclass = ent.get("semantic_class")
            if not sclass or sclass not in SEMANTIC_CLASSES:
                continue
            # Preserve order: if this ent has a span, it occupies the next survivor slot.
            if kept < len(survivors) and survivors[kept] is ent:
                survivor_index[orig_idx] = kept
                kept += 1

        # Concept-norm targets (only for in-scope SNOMED-normalized entities).
        norm_targets: List[Dict[str, float]] = []
        norm_entity_idx: List[int] = []
        norm_weights: List[float] = []
        for survivor_i, ent in enumerate(survivors):
            sclass = ent.get("semantic_class")
            if sclass not in SNOMED_NER_CLASSES:
                continue
            mesh = _normalize_mesh_code(ent.get("original_code"))
            target_dist: Dict[str, float] = {}
            if mesh and mesh in self.soft_lookup:
                for cand in self.soft_lookup[mesh]:
                    cid = str(cand.get("snomed_id"))
                    p = float(cand.get("prob", 0.0))
                    if cid and p > 0:
                        target_dist[cid] = target_dist.get(cid, 0.0) + p
            if not target_dist:
                snomed = ent.get("mapped_snomed_id")
                if snomed:
                    target_dist = {str(snomed): 1.0}
            if not target_dist:
                continue
            total = sum(target_dist.values()) or 1.0
            target_dist = {k: v / total for k, v in target_dist.items()}
            norm_targets.append(target_dist)
            norm_entity_idx.append(survivor_i)
            conf = ent.get("mapping_confidence")
            norm_weights.append(float(conf) if conf is not None else 1.0)

        pair_idx, pair_lab, pair_w = enumerate_pairs(
            survivors,
            raw.get("relations", []),
            survivor_index=survivor_index,
            neg_ratio=self.neg_ratio,
            max_pairs=self.max_pairs,
            rng=rng,
        )

        if pair_idx:
            pair_classes = []
            for (a, b) in pair_idx:
                ca = SEMANTIC_CLASS_TO_ID.get(survivors[a].get("semantic_class") or "none", SEMANTIC_CLASS_TO_ID["none"])
                cb = SEMANTIC_CLASS_TO_ID.get(survivors[b].get("semantic_class") or "none", SEMANTIC_CLASS_TO_ID["none"])
                pair_classes.append([ca, cb])
            pair_indices_t = torch.tensor(pair_idx, dtype=torch.long)
            pair_labels_t = torch.tensor(pair_lab, dtype=torch.long)
            pair_weights_t = torch.tensor(pair_w, dtype=torch.float)
            pair_classes_t = torch.tensor(pair_classes, dtype=torch.long)
        else:
            pair_indices_t = torch.zeros((0, 2), dtype=torch.long)
            pair_labels_t = torch.zeros((0,), dtype=torch.long)
            pair_weights_t = torch.zeros((0,), dtype=torch.float)
            pair_classes_t = torch.zeros((0, 2), dtype=torch.long)

        return DocFeatures(
            pmid=str(raw.get("pmid", "")),
            corpus=str(raw.get("corpus", "")),
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention, dtype=torch.long),
            offset_mapping=torch.tensor(offsets, dtype=torch.long),
            bio_labels=torch.tensor(bio, dtype=torch.long),
            entity_token_spans=spans,
            entity_original=survivors,
            norm_targets=norm_targets,
            norm_entity_idx=norm_entity_idx,
            norm_weights=norm_weights,
            pair_indices=pair_indices_t,
            pair_labels=pair_labels_t,
            pair_weights=pair_weights_t,
            pair_semantic_classes=pair_classes_t,
            raw_doc=raw,
        )


def collate_docs(batch: List[DocFeatures], pad_token_id: int = 0) -> Dict:
    """Pad token tensors to the longest sequence in the batch.

    Variable-length per-doc lists (entity_token_spans, norm_targets, pair_indices,
    pair_labels, pair_weights, pair_semantic_classes) are returned as Python lists
    of length B. Heads index into the encoder output per batch element.
    """
    B = len(batch)
    T = max(int(d.input_ids.shape[0]) for d in batch)

    input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    offset_mapping = torch.zeros((B, T, 2), dtype=torch.long)
    bio_labels = torch.zeros((B, T), dtype=torch.long)

    for i, d in enumerate(batch):
        L = int(d.input_ids.shape[0])
        input_ids[i, :L] = d.input_ids
        attention_mask[i, :L] = d.attention_mask
        offset_mapping[i, :L] = d.offset_mapping
        bio_labels[i, :L] = d.bio_labels

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
        "bio_labels": bio_labels,
        "entity_token_spans": [d.entity_token_spans for d in batch],
        "entity_original": [d.entity_original for d in batch],
        "norm_targets": [d.norm_targets for d in batch],
        "norm_entity_idx": [d.norm_entity_idx for d in batch],
        "norm_weights": [d.norm_weights for d in batch],
        "pair_indices": [d.pair_indices for d in batch],
        "pair_labels": [d.pair_labels for d in batch],
        "pair_weights": [d.pair_weights for d in batch],
        "pair_semantic_classes": [d.pair_semantic_classes for d in batch],
        "pmids": [d.pmid for d in batch],
        "corpora": [d.corpus for d in batch],
        "raw_docs": [d.raw_doc for d in batch],
    }


__all__ = [
    "BIO_LABELS",
    "BIO_LABEL_TO_ID",
    "BIO_ID_TO_LABEL",
    "NUM_BIO_LABELS",
    "RELATION_LABELS",
    "RELATION_LABEL_TO_ID",
    "NUM_RELATION_LABELS",
    "NO_RELATION_ID",
    "SEMANTIC_CLASSES",
    "SEMANTIC_CLASS_LIST",
    "SEMANTIC_CLASS_TO_ID",
    "NUM_SEMANTIC_CLASSES",
    "DocFeatures",
    "CanonDocDataset",
    "collate_docs",
    "load_soft_lookup",
    "build_bio_targets",
    "enumerate_pairs",
]
