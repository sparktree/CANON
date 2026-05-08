"""Multi-task heads on the SapBERT-pretrained encoder (CANON Phase 3.2).

Four heads share the encoder:
  * NERHead              -- 13-class BIO with a CRF decoder
  * ConceptNormHead      -- bi-encoder over a precomputed SNOMED matrix
  * RelationHead         -- 13-way MLP over span pair + [CLS] + type-pair emb
  * TemporalHead         -- Phase 5 placeholder (not implemented)

MultiTaskModel composes them and exposes a forward() that takes a collated
batch from canon_dataset.collate_docs() and returns per-task losses + raw
outputs (for downstream metric computation, CSP top-k extraction, etc.).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoModel

try:
    from canon_dataset import (
        NUM_BIO_LABELS,
        NUM_RELATION_LABELS,
        NUM_SEMANTIC_CLASSES,
        NO_RELATION_ID,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from canon_dataset import (
        NUM_BIO_LABELS,
        NUM_RELATION_LABELS,
        NUM_SEMANTIC_CLASSES,
        NO_RELATION_ID,
    )


# ---------------------------------------------------------------------------
# NER head
# ---------------------------------------------------------------------------


class NERHead(nn.Module):
    """Token classifier with a linear-chain CRF decoder.

    The encoder hidden states pass through a dropout + linear projection to
    NUM_BIO_LABELS scores. The CRF computes the negative log-likelihood for
    training and `decode` for inference.
    """

    def __init__(self, hidden_size: int, num_tags: int = NUM_BIO_LABELS, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        bio_labels: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emissions = self.classifier(self.dropout(hidden_states))
        mask_bool = attention_mask.bool()
        # torchcrf requires the first time step to be unmasked; pad tokens come last.
        out: Dict[str, torch.Tensor] = {"logits": emissions}
        if bio_labels is not None:
            ll = self.crf(emissions, bio_labels, mask=mask_bool, reduction="sum")
            denom = mask_bool.sum().clamp(min=1).float()
            out["loss"] = -ll / denom
        out["decoded"] = self.crf.decode(emissions, mask=mask_bool)
        return out


# ---------------------------------------------------------------------------
# Concept normalization head
# ---------------------------------------------------------------------------


class ConceptNormHead(nn.Module):
    """Bi-encoder concept-normalization head.

    Span representation: mean-pool of contextualized token embeddings inside
    the entity's token span, then a single linear projection to dim H.

    Candidate concept embeddings live as a (N_concepts, H) matrix loaded from
    CONCEPT_INDEX_DIR/concept_emb.safetensors. By default it is registered as
    a non-trainable buffer (Stage 1 + Stage 2). Setting train_concept_emb=True
    converts it into an nn.Parameter for Stage 3 fine-tuning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_concepts: int,
        *,
        train_concept_emb: bool = False,
        temperature_init: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        # Lazily filled in load_concept_index(); shape (num_concepts, hidden_size).
        if train_concept_emb:
            self.concept_emb: nn.Parameter = nn.Parameter(torch.zeros(num_concepts, hidden_size))
        else:
            self.register_buffer("concept_emb", torch.zeros(num_concepts, hidden_size))
        self.tau = float(temperature_init)
        self._concept_id_to_row: Dict[str, int] = {}
        self._concept_ids: List[str] = []
        self._train_concept_emb = train_concept_emb

    @property
    def concept_ids(self) -> List[str]:
        return self._concept_ids

    @property
    def concept_id_to_row(self) -> Dict[str, int]:
        return self._concept_id_to_row

    def load_concept_index(
        self,
        concept_ids_path: Path,
        concept_emb_path: Path,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """Replace concept_emb with the precomputed matrix from disk."""
        from safetensors.torch import load_file

        with Path(concept_ids_path).open("r", encoding="utf-8") as fh:
            self._concept_ids = json.load(fh)
        self._concept_id_to_row = {cid: i for i, cid in enumerate(self._concept_ids)}

        loaded = load_file(str(concept_emb_path))
        emb = loaded["embeddings"].float()
        if device is not None:
            emb = emb.to(device)
        if self._train_concept_emb:
            with torch.no_grad():
                self.concept_emb.data.copy_(emb)
        else:
            self.concept_emb = emb  # type: ignore[assignment]
            # Re-register as buffer if module was moved off-device by .to().
            self.register_buffer("concept_emb", emb)

    def span_repr(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        token_spans: Sequence[Sequence[Tuple[int, int, str]]],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Mean-pool encoder outputs over each entity span.

        Returns
        -------
        span_vecs : (E_total, H) projected span representations
        index     : list of (batch_idx, span_idx_within_doc) aligned with span_vecs rows.
        """
        outputs: List[torch.Tensor] = []
        idx: List[Tuple[int, int]] = []
        H = hidden_states.size(-1)
        for b, spans in enumerate(token_spans):
            for s_idx, (start, end, _sclass) in enumerate(spans):
                if end <= start:
                    pooled = hidden_states[b, start].unsqueeze(0)
                else:
                    pooled = hidden_states[b, start:end].mean(dim=0, keepdim=True)
                outputs.append(pooled)
                idx.append((b, s_idx))
        if not outputs:
            empty = hidden_states.new_zeros(0, H)
            return self.proj(self.dropout(empty)), []
        stacked = torch.cat(outputs, dim=0)
        return self.proj(self.dropout(stacked)), idx

    def score(self, span_vecs: torch.Tensor, candidate_subset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cosine-sim scores against the concept matrix.

        candidate_subset, if given, is a 1-D long tensor of concept rows to
        score against (CSP top-k path). Otherwise scores all rows.
        """
        if span_vecs.numel() == 0:
            return span_vecs.new_zeros(0, self.concept_emb.size(0) if candidate_subset is None else candidate_subset.numel())
        a = F.normalize(span_vecs, dim=-1)
        if candidate_subset is None:
            b = F.normalize(self.concept_emb, dim=-1)
        else:
            b = F.normalize(self.concept_emb[candidate_subset], dim=-1)
        return a @ b.T

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        token_spans: Sequence[Sequence[Tuple[int, int, str]]],
        norm_targets: Optional[Sequence[Sequence[Dict[str, float]]]] = None,
        norm_entity_idx: Optional[Sequence[Sequence[int]]] = None,
        norm_weights: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict:
        span_vecs, idx_map = self.span_repr(hidden_states, attention_mask, token_spans)
        out: Dict = {"span_vecs": span_vecs, "span_index": idx_map}
        if norm_targets is None or self.concept_emb.size(0) == 0 or span_vecs.size(0) == 0:
            out["loss"] = None
            return out

        # Build a soft-target matrix aligned with a subset of span_vecs rows.
        # For each (batch, entity_idx) listed in norm_entity_idx[b][k], we look
        # up the matching row in span_vecs (which iterates in the same order
        # span_repr produced them) and assemble per-row probability vectors.
        device = span_vecs.device
        scores = self.score(span_vecs)  # (E_total, N)
        N = scores.size(1)

        # Reverse map: (batch_idx, span_idx) -> row_in_span_vecs.
        position: Dict[Tuple[int, int], int] = {(b, s): i for i, (b, s) in enumerate(idx_map)}

        target_rows: List[int] = []
        target_dists: List[Dict[str, float]] = []
        target_weights: List[float] = []
        for b in range(len(token_spans)):
            ent_idx_list = norm_entity_idx[b] if norm_entity_idx is not None else []
            target_list = norm_targets[b] if norm_targets is not None else []
            weight_list = norm_weights[b] if norm_weights is not None else [1.0] * len(target_list)
            for k, ent_i in enumerate(ent_idx_list):
                key = (b, int(ent_i))
                if key not in position:
                    continue
                target_rows.append(position[key])
                target_dists.append(target_list[k])
                target_weights.append(float(weight_list[k]) if k < len(weight_list) else 1.0)

        if not target_rows:
            out["loss"] = None
            return out

        rows_t = torch.tensor(target_rows, dtype=torch.long, device=device)
        sub_scores = scores.index_select(0, rows_t)  # (E_targets, N)
        log_q = F.log_softmax(sub_scores / max(self.tau, 1e-6), dim=-1)

        target_p = torch.zeros_like(log_q)
        skipped = 0
        for r, dist in enumerate(target_dists):
            for cid, p in dist.items():
                row = self._concept_id_to_row.get(str(cid))
                if row is None or row >= N:
                    continue
                target_p[r, row] = p
            row_sum = target_p[r].sum()
            if row_sum.item() <= 0:
                skipped += 1
                continue
            target_p[r] = target_p[r] / row_sum

        weights_t = torch.tensor(target_weights, dtype=torch.float, device=device)
        # Exclude rows whose entire target lay outside the concept index.
        valid_mask = target_p.sum(dim=-1) > 0
        if valid_mask.any():
            per_row = -(target_p[valid_mask] * log_q[valid_mask]).sum(dim=-1)
            per_row = per_row * weights_t[valid_mask]
            out["loss"] = per_row.mean()
        else:
            out["loss"] = None
        out["scores"] = scores
        out["target_rows"] = rows_t
        return out


# ---------------------------------------------------------------------------
# Relation head
# ---------------------------------------------------------------------------


class RelationHead(nn.Module):
    """Per-pair classifier over the 13 unified relation labels (incl. no-relation).

    Input vector: [span_a || span_b || [CLS] || type_pair_emb]
                   (H + H + H + 2*type_emb_dim)
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int = NUM_RELATION_LABELS,
        type_emb_dim: int = 32,
        mlp_hidden: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(NUM_SEMANTIC_CLASSES, type_emb_dim)
        in_dim = 3 * hidden_size + 2 * type_emb_dim
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cls_repr: torch.Tensor,
        token_spans: Sequence[Sequence[Tuple[int, int, str]]],
        pair_indices: Sequence[torch.Tensor],
        pair_semantic_classes: Sequence[torch.Tensor],
        pair_labels: Optional[Sequence[torch.Tensor]] = None,
        pair_weights: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict:
        device = hidden_states.device
        H = hidden_states.size(-1)

        # Pre-pool span representations per doc (mean of token embeddings).
        per_doc_spans: List[torch.Tensor] = []
        for b, spans in enumerate(token_spans):
            if not spans:
                per_doc_spans.append(hidden_states.new_zeros(0, H))
                continue
            rows = []
            for (start, end, _) in spans:
                if end <= start:
                    rows.append(hidden_states[b, start].unsqueeze(0))
                else:
                    rows.append(hidden_states[b, start:end].mean(dim=0, keepdim=True))
            per_doc_spans.append(torch.cat(rows, dim=0))

        feats: List[torch.Tensor] = []
        labels_chunks: List[torch.Tensor] = []
        weights_chunks: List[torch.Tensor] = []
        per_doc_logits: List[torch.Tensor] = []

        for b, idx_t in enumerate(pair_indices):
            if idx_t.numel() == 0:
                per_doc_logits.append(hidden_states.new_zeros(0, self.mlp[-1].out_features))
                continue
            idx_t = idx_t.to(device)
            spans = per_doc_spans[b]
            if spans.size(0) == 0:
                per_doc_logits.append(hidden_states.new_zeros(0, self.mlp[-1].out_features))
                continue
            sa = spans.index_select(0, idx_t[:, 0])
            sb = spans.index_select(0, idx_t[:, 1])
            cls = cls_repr[b].unsqueeze(0).expand(sa.size(0), -1)
            class_ids = pair_semantic_classes[b].to(device)
            type_a = self.type_emb(class_ids[:, 0])
            type_b = self.type_emb(class_ids[:, 1])
            x = torch.cat([sa, sb, cls, type_a, type_b], dim=-1)
            logits = self.mlp(x)
            per_doc_logits.append(logits)
            feats.append(x)
            if pair_labels is not None:
                labels_chunks.append(pair_labels[b].to(device))
            if pair_weights is not None:
                weights_chunks.append(pair_weights[b].to(device))

        out: Dict = {"per_doc_logits": per_doc_logits}
        if pair_labels is not None and per_doc_logits and any(t.numel() for t in per_doc_logits):
            all_logits = torch.cat([t for t in per_doc_logits if t.numel()], dim=0)
            all_labels = torch.cat(labels_chunks, dim=0) if labels_chunks else hidden_states.new_zeros(0, dtype=torch.long)
            if pair_weights is not None and weights_chunks:
                all_weights = torch.cat(weights_chunks, dim=0)
            else:
                all_weights = torch.ones_like(all_labels, dtype=torch.float)
            if all_labels.numel() == 0:
                out["loss"] = None
            else:
                per_pair = F.cross_entropy(all_logits, all_labels, reduction="none")
                out["loss"] = (per_pair * all_weights).mean()
        else:
            out["loss"] = None
        return out


# ---------------------------------------------------------------------------
# Temporal head (Phase 5 placeholder)
# ---------------------------------------------------------------------------


class TemporalHead(nn.Module):
    """Phase 5 placeholder; not implemented in 3.2."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError("TemporalHead is deferred to Phase 5.")


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------


class MultiTaskModel(nn.Module):
    """Encoder + NER + ConceptNorm + Relation heads.

    Parameters
    ----------
    encoder_dir : path to a HF checkpoint (e.g. SAPBERT_ENCODER_DIR).
    num_concepts : number of rows in the concept-norm candidate matrix.
    train_concept_emb : if True, the candidate matrix becomes trainable.
    """

    def __init__(
        self,
        encoder_dir: str,
        num_concepts: int,
        *,
        train_concept_emb: bool = False,
        ner: bool = True,
        norm: bool = True,
        rel: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_dir)
        H = self.encoder.config.hidden_size
        self.hidden_size = H
        self.has_ner = ner
        self.has_norm = norm
        self.has_rel = rel
        if ner:
            self.ner_head = NERHead(H)
        if norm:
            self.norm_head = ConceptNormHead(H, num_concepts, train_concept_emb=train_concept_emb)
        if rel:
            self.rel_head = RelationHead(H)

    def freeze_encoder(self, frozen: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not frozen

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls = h[:, 0]
        return h, cls

    def forward(self, batch: Dict, *, active_heads: Optional[Sequence[str]] = None) -> Dict:
        active = set(active_heads) if active_heads is not None else {"ner", "norm", "rel"}
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bio_labels = batch.get("bio_labels")
        if isinstance(bio_labels, torch.Tensor):
            bio_labels = bio_labels.to(device)

        hidden, cls = self.encode(input_ids, attention_mask)
        outputs: Dict = {"hidden": hidden, "cls": cls, "losses": {}, "raw": {}}

        if self.has_ner and "ner" in active:
            ner_out = self.ner_head(hidden, attention_mask, bio_labels)
            outputs["raw"]["ner"] = ner_out
            if ner_out.get("loss") is not None:
                outputs["losses"]["ner"] = ner_out["loss"]

        if self.has_norm and "norm" in active:
            norm_out = self.norm_head(
                hidden,
                attention_mask,
                batch["entity_token_spans"],
                norm_targets=batch.get("norm_targets"),
                norm_entity_idx=batch.get("norm_entity_idx"),
                norm_weights=batch.get("norm_weights"),
            )
            outputs["raw"]["norm"] = norm_out
            if norm_out.get("loss") is not None:
                outputs["losses"]["norm"] = norm_out["loss"]

        if self.has_rel and "rel" in active:
            rel_out = self.rel_head(
                hidden,
                cls,
                batch["entity_token_spans"],
                batch["pair_indices"],
                batch["pair_semantic_classes"],
                pair_labels=batch.get("pair_labels"),
                pair_weights=batch.get("pair_weights"),
            )
            outputs["raw"]["rel"] = rel_out
            if rel_out.get("loss") is not None:
                outputs["losses"]["rel"] = rel_out["loss"]

        return outputs


__all__ = [
    "NERHead",
    "ConceptNormHead",
    "RelationHead",
    "TemporalHead",
    "MultiTaskModel",
]
