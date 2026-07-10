"""Differentiable expected-violation loss for CANON configuration (c)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from canon_dataset import BIO_LABEL_TO_ID, RELATION_LABELS, SEMANTIC_CLASSES
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
    import mrcm_validity
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from canon_dataset import BIO_LABEL_TO_ID, RELATION_LABELS, SEMANTIC_CLASSES
    from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
    import mrcm_validity


class ExpectedViolationLoss(nn.Module):
    """Penalize probability assigned to ontology-incompatible joint choices."""

    def __init__(self, tables: mrcm_validity.MRCMTables, concept_ids: Sequence[str],
                 top_k_concepts: int = 32) -> None:
        super().__init__()
        self.tables = tables
        self.top_k_concepts = top_k_concepts
        mask = torch.zeros((len(SEMANTIC_CLASSES), len(concept_ids)), dtype=torch.bool)
        for type_idx, sem_class in enumerate(SEMANTIC_CLASSES):
            if sem_class not in tables.type_to_anchors:
                mask[type_idx] = True
                continue
            mask[type_idx] = torch.tensor([
                mrcm_validity.concept_under_type(str(cid), sem_class, tables)
                for cid in concept_ids
            ], dtype=torch.bool)
        self.register_buffer("concept_validity", mask, persistent=False)

    @staticmethod
    def _span_type_probs(ner_logits: torch.Tensor, b: int, start: int, end: int) -> torch.Tensor:
        token_probs = torch.softmax(ner_logits[b, start:end], dim=-1)
        rows = []
        for sem_class in SEMANTIC_CLASSES:
            ids = [BIO_LABEL_TO_ID[f"B-{sem_class}"], BIO_LABEL_TO_ID[f"I-{sem_class}"]]
            rows.append(token_probs[:, ids].sum(dim=-1).mean())
        result = torch.stack(rows)
        return result / result.sum().clamp(min=1e-8)

    def forward(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        device = outputs["hidden"].device
        violations: List[torch.Tensor] = []
        ner_logits = outputs["raw"]["ner"]["logits"]
        spans = batch["entity_token_spans"]

        # Entity NER <-> concept expected compatibility.
        norm = outputs["raw"].get("norm", {})
        scores = norm.get("scores")
        span_index: List[Tuple[int, int]] = norm.get("span_index", [])
        eligible = {(b, int(e)) for b, indices in enumerate(batch.get("norm_entity_idx", [])) for e in indices}
        if scores is not None:
            k = min(self.top_k_concepts, scores.size(-1))
            topv, topi = torch.topk(scores, k, dim=-1)
            concept_probs = torch.softmax(topv, dim=-1)
            for row, (b, entity_idx) in enumerate(span_index):
                if (b, entity_idx) not in eligible:
                    continue
                start, end, _ = spans[b][entity_idx]
                type_probs = self._span_type_probs(ner_logits, b, start, end)
                validity = self.concept_validity[:, topi[row]].to(device).float()
                valid_mass = (type_probs[:, None] * concept_probs[row][None, :] * validity).sum()
                violations.append(-torch.log(valid_mass.clamp(min=1e-8)))

        # Relation expected compatibility, coupled to NER type probabilities.
        rel = outputs["raw"].get("rel", {})
        for b, logits in enumerate(rel.get("per_doc_logits", [])):
            if logits.numel() == 0:
                continue
            rel_probs = torch.softmax(logits, dim=-1)
            originals = batch["entity_original"][b]
            for row, pair in enumerate(batch["pair_indices"][b].tolist()):
                i, j = pair
                si, ei, _ = spans[b][i]
                sj, ej, _ = spans[b][j]
                pi = self._span_type_probs(ner_logits, b, si, ei)
                pj = self._span_type_probs(ner_logits, b, sj, ej)
                valid_by_rel = []
                for label in RELATION_LABELS:
                    if label == "no-relation":
                        valid_by_rel.append(rel_probs.new_tensor(1.0))
                    elif label in TIER1_RELATIONS:
                        mass = rel_probs.new_tensor(0.0)
                        for ai, ta in enumerate(SEMANTIC_CLASSES):
                            for bj, tb in enumerate(SEMANTIC_CLASSES):
                                if mrcm_validity.relation_pair_valid(label, ta, tb, self.tables):
                                    mass = mass + pi[ai] * pj[bj]
                        valid_by_rel.append(mass)
                    elif label in TIER2_RELATIONS:
                        stys_i = originals[i].get("umls_stys", []) if i < len(originals) else []
                        stys_j = originals[j].get("umls_stys", []) if j < len(originals) else []
                        valid = mrcm_validity.relation_pair_valid_sn(label, stys_i, stys_j, self.tables)
                        valid_by_rel.append(rel_probs.new_tensor(float(valid)))
                    else:
                        valid_by_rel.append(rel_probs.new_tensor(0.0))
                valid_mass = (rel_probs[row] * torch.stack(valid_by_rel)).sum()
                violations.append(-torch.log(valid_mass.clamp(min=1e-8)))

        return torch.stack(violations).mean() if violations else outputs["hidden"].sum() * 0.0


__all__ = ["ExpectedViolationLoss"]
