"""Coherence-vs-no-CSP evaluation for the CANON shrunk-plan run.

This is the proof-of-concept measurement the parent plan §4.3 describes:
for each (entity_type, concept) pair and each (subject, relation, object)
triple, check whether the prediction is ontologically coherent. Compare
two configurations:

    (a) Neural-only (argmax over each head)
    (b) Neural + CSP at inference   [the symbolic layer]

Coherence surfaces (per the parent plan amendment):
    surface (i)  -- entity-level NER<->concept consistency:
                    predicted concept must lie under predicted NER type's
                    SNOMED hierarchy anchor. Fires on every entity with a
                    predicted concept.
    surface (ii) -- relation-level domain/range:
                    predicted relation's MRCM domain/range must accept the
                    predicted entity types. Fires only when a Tier-1
                    relation is predicted; Tier-2 relations are vacuously
                    coherent.

Accuracy metrics (against gold dev):
    norm_top1_acc  -- predicted concept == gold concept on entities with
                      a verified gold mapping.
    rel_macro_f1   -- macro F1 over the 12 positive unified relation labels
                      (no-relation excluded), restricted to the gold pair
                      population.

Both metrics are reported neural-only and neural+CSP so the project's
coherence-performance tradeoff (parent plan §4.4) is measurable.

CLI
---
    python scripts/evaluate_coherence.py --model-dir outputs/phase3/stage2_local/best
        [--split dev|test] [--max-docs N] [--reuse-predictions PATH]

Outputs:
    outputs/phase3/coherence_eval/predictions.jsonl
    outputs/phase3/coherence_eval/coherence_summary.json
    outputs/phase3/coherence_eval/log.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    import config
    from canon_dataset import (
        BIO_LABELS,
        CanonDocDataset,
        NO_RELATION_ID,
        NUM_RELATION_LABELS,
        RELATION_LABELS,
        RELATION_LABEL_TO_ID,
        SEMANTIC_CLASSES,
        collate_docs,
        load_soft_lookup,
    )
    from csp_solver import (
        SEMANTIC_CLASSES as CSP_SEMANTIC_CLASSES,
        TIER1_RELATIONS,
        TIER2_RELATIONS,
        ConstraintTables,
        load_constraint_tables,
        model_predict,
        _neural_argmax,
        solve_document,
    )
    from heads import MultiTaskModel
    from unified_format import Document, read_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    from canon_dataset import (
        BIO_LABELS,
        CanonDocDataset,
        NO_RELATION_ID,
        NUM_RELATION_LABELS,
        RELATION_LABELS,
        RELATION_LABEL_TO_ID,
        SEMANTIC_CLASSES,
        collate_docs,
        load_soft_lookup,
    )
    from csp_solver import (
        SEMANTIC_CLASSES as CSP_SEMANTIC_CLASSES,
        TIER1_RELATIONS,
        TIER2_RELATIONS,
        ConstraintTables,
        load_constraint_tables,
        model_predict,
        _neural_argmax,
        solve_document,
    )
    from heads import MultiTaskModel
    from unified_format import Document, read_jsonl


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR     = config.PHASE3_OUTPUTS / "coherence_eval"
PREDICTIONS    = OUTPUT_DIR / "predictions.jsonl"
SUMMARY_OUT    = OUTPUT_DIR / "coherence_summary.json"
LOG_PATH       = OUTPUT_DIR / "log.txt"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("evaluate_coherence")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [coherence] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Coherence metric
# ---------------------------------------------------------------------------

def entity_coherent(sem_class: str, concept_id: Optional[str], tables: ConstraintTables) -> Optional[bool]:
    """True/False if the concept satisfies MRCM type-concept compatibility; None if N/A.

    Aligned with csp_solver.concept_under_type: the anchor concept itself is
    a valid member of its own type (an anchor IS the most general concept of
    that type). This matches the MRCM interpretation the CSP solver enforces.
    """
    if not concept_id or concept_id == "no-relation":
        return None
    anchors = tables.type_to_anchors.get(sem_class)
    if not anchors:
        return None  # 'gene', 'variant', 'species', 'cell_line', 'none' have no anchor
    cid = str(concept_id)
    for anc in anchors:
        if cid == str(anc):
            return True  # anchor-as-self counts per MRCM
        desc = tables.descendants.get(str(anc))
        if desc and cid in desc:
            return True
    return False


def entity_coherent_strict(sem_class: str, concept_id: Optional[str], tables: ConstraintTables) -> Optional[bool]:
    """Stricter coherence: concept must lie STRICTLY under (not equal to) the anchor.

    The CSP solver can satisfy MRCM coherence by retreating to the bare anchor
    concept (e.g., 105590001 Substance for any chemical). That is technically
    valid per MRCM but uninformative as a normalization. This metric makes the
    cost of such retreats visible: CSP entities counted compliant under
    entity_coherent but not entity_coherent_strict are anchor-retreats.
    """
    if not concept_id or concept_id == "no-relation":
        return None
    anchors = tables.type_to_anchors.get(sem_class)
    if not anchors:
        return None
    cid = str(concept_id)
    if cid in {str(a) for a in anchors}:
        return False  # anchor self-match excluded
    for anc in anchors:
        desc = tables.descendants.get(str(anc))
        if desc and cid in desc:
            return True
    return False


def relation_coherent(rel: str, type_a: str, type_b: str, tables: ConstraintTables) -> Optional[bool]:
    """True/False for Tier-1 relations against MRCM domain/range. None when
    Tier-2 or no-relation (those are vacuously coherent — surface (ii) doesn't
    apply)."""
    if rel == "no-relation":
        return None
    if rel in TIER2_RELATIONS:
        return None
    if rel in TIER1_RELATIONS:
        return bool(tables.valid_pair_for_relation.get((rel, type_a, type_b), False))
    return None


# ---------------------------------------------------------------------------
# Gold-truth alignment
# ---------------------------------------------------------------------------

def load_gold_dev(path: Path) -> Dict[str, Document]:
    """Index gold dev by pmid for fast lookup."""
    out: Dict[str, Document] = {}
    for doc in read_jsonl(path):
        out[str(doc.pmid)] = doc
    return out


def gold_concept_for_entity(doc: Document, entity_idx: int) -> Optional[str]:
    if entity_idx < 0 or entity_idx >= len(doc.entities):
        return None
    em = doc.entities[entity_idx]
    if em.mapped_snomed_id and em.snomed_active is True:
        return str(em.mapped_snomed_id)
    return None


def gold_relation_for_pair(doc: Document, i: int, j: int) -> str:
    """Return the gold target_relation for entities i,j (or 'no-relation' if absent)."""
    for r in doc.relations:
        if r.subject_idx == i and r.object_idx == j:
            return r.target_relation or "no-relation"
    return "no-relation"


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

class MetricAccum:
    """Per-configuration accumulators for one of {'neural', 'oracle', 'csp'}.

    All denominators are locked to the same evaluable-entity population
    (gold type has anchors AND non-empty concept candidates AND the
    underlying entity was scored by the model). When CSP flips an
    entity's type to 'none', that entity is now COUNTED as incoherent
    (rather than silently dropped from the denominator), so the rate
    reported here is faithful to the population a fair comparison
    requires.
    """
    def __init__(self) -> None:
        # Surface (i): MRCM-aligned coherence (anchor-inclusive). This is
        # what the CSP solver enforces -- a concept satisfies type
        # compatibility iff it equals OR is under the anchor.
        self.ent_total = 0
        self.ent_coherent = 0
        # Surface (i, strict): excludes anchor self-matches. Surfaces the
        # cost of "CSP retreats to anchor" -- ent_coherent - ent_strict
        # entities are compliant only because the chosen concept IS the
        # anchor (e.g. 105590001 Substance for chemical entities), which
        # is technically valid per MRCM but pragmatically uninformative.
        self.ent_strict_total = 0
        self.ent_strict_coherent = 0
        # Surface (ii): relation-level coherence (Tier-1 only)
        self.tier1_total = 0
        self.tier1_coherent = 0
        # Norm top-1
        self.norm_evaluable = 0
        self.norm_correct = 0
        # Relation F1: collected for macro F1 over positive labels
        self.rel_pred: List[int] = []
        self.rel_gold: List[int] = []
        # Tier breakdown of relation predictions
        self.pred_label_counts: Counter = Counter()

    def to_dict(self) -> Dict:
        ent_rate = self.ent_coherent / self.ent_total if self.ent_total else 0.0
        ent_strict_rate = (
            self.ent_strict_coherent / self.ent_strict_total
            if self.ent_strict_total else 0.0
        )
        tier1_rate = self.tier1_coherent / self.tier1_total if self.tier1_total else 0.0
        norm_acc = self.norm_correct / self.norm_evaluable if self.norm_evaluable else 0.0
        f1s = []
        for cls in range(NUM_RELATION_LABELS):
            if cls == NO_RELATION_ID:
                continue
            tp = sum(1 for p, g in zip(self.rel_pred, self.rel_gold) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(self.rel_pred, self.rel_gold) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(self.rel_pred, self.rel_gold) if p != cls and g == cls)
            if (tp + fp) and (tp + fn):
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f = 2 * p * r / (p + r) if (p + r) else 0.0
            else:
                f = 0.0
            f1s.append(f)
        rel_macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        return {
            "entity_validity_rate":      round(ent_rate, 4),
            "entity_evaluated":          self.ent_total,
            "entity_coherent":           self.ent_coherent,
            "entity_strict_validity_rate":      round(ent_strict_rate, 4),
            "entity_strict_evaluated":          self.ent_strict_total,
            "entity_strict_coherent":           self.ent_strict_coherent,
            "tier1_validity_rate":       round(tier1_rate, 4),
            "tier1_evaluated":           self.tier1_total,
            "tier1_coherent":            self.tier1_coherent,
            "norm_top1_accuracy":        round(norm_acc, 4),
            "norm_evaluated":            self.norm_evaluable,
            "norm_correct":              self.norm_correct,
            "relation_macro_f1":         round(rel_macro_f1, 4),
            "relation_pred_distribution": dict(self.pred_label_counts.most_common()),
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict:
    device = torch.device("cuda" if (args.device == "cuda" or
                                     (args.device == "auto" and torch.cuda.is_available()))
                          else "cpu")
    logger.info(f"device={device}")

    tables = load_constraint_tables(
        config.MRCM_CONSTRAINTS_JSON,
        config.SNOMED_ANCESTORS_PKL,
        logger=logger,
    )

    split_path = Path(args.split_path) if args.split_path else (
        config.PHASE2_SPLITS_DIR / f"{args.split}.jsonl"
    )
    if not split_path.exists():
        raise FileNotFoundError(f"split not found: {split_path}")

    logger.info(f"loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    pad_id = tokenizer.pad_token_id or 0

    with config.CONCEPT_INDEX_IDS.open() as fh:
        num_concepts = len(json.load(fh))
    model = MultiTaskModel(args.model_dir, num_concepts=num_concepts, ner=True, norm=True, rel=True)
    model.norm_head.load_concept_index(config.CONCEPT_INDEX_IDS, config.CONCEPT_INDEX_EMB)

    head_state_path = Path(args.model_dir) / "head_state.pt"
    if head_state_path.exists():
        sd = torch.load(head_state_path, map_location="cpu", weights_only=True)
        own = {n: p for n, p in model.named_parameters() if not n.startswith("encoder.")}
        loaded = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].data.copy_(v)
                loaded += 1
        logger.info(f"loaded {loaded} head params from {head_state_path}")
    else:
        logger.info("no head_state.pt found alongside encoder; head weights remain initial")

    model.to(device).eval()

    soft = load_soft_lookup(config.SOFT_MAPPING_LOOKUP)
    ds = CanonDocDataset(
        split_path, tokenizer, soft,
        max_length=args.max_length, max_docs=args.max_docs,
        max_pairs=args.max_pairs, seed=0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=lambda b: collate_docs(b, pad_token_id=pad_id))

    logger.info("running neural inference + CSP solving over dev split ...")
    t0 = time.time()
    neural_records = model_predict(
        model, loader, device,
        top_k_types=2, top_k_concepts=args.top_k_concepts, top_k_relations=3,
    )
    logger.info(f"  neural inference done in {time.time() - t0:.1f}s; {len(neural_records)} docs")

    gold_index = load_gold_dev(split_path)

    neural_metrics = MetricAccum()
    csp_metrics = MetricAccum()
    oracle_metrics = MetricAccum()  # top-K oracle: is ANY top-K concept compliant?

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    overrides_concept = 0
    overrides_relation = 0
    csp_status_counts: Counter = Counter()
    type_flips_csp = 0  # CSP forced sem_class -> 'none' due to no compliant candidate

    t0 = time.time()
    with PREDICTIONS.open("w", encoding="utf-8") as out_fh:
        for rec in neural_records:
            pmid = str(rec["pmid"])
            gold = gold_index.get(pmid)
            neural = _neural_argmax(rec["entities"], rec["pairs"])
            sol = solve_document(rec, tables, timeout_ms=args.timeout_ms)
            csp_assn = sol.get("assignment") if sol.get("status") == "sat" else neural
            csp_status_counts[sol.get("status", "unknown")] += 1

            for e_idx, (n_e, c_e, neural_ent_rec) in enumerate(zip(
                neural["entities"], csp_assn["entities"], rec["entities"]
            )):
                # Lock the denominator on the gold type. Both neural and CSP
                # are scored against the SAME population: entities whose gold
                # type has hierarchy anchors AND that have at least one
                # concept candidate. This eliminates the "denominator drift"
                # that the original metric had when CSP flipped to type=none.
                gold_type = neural_ent_rec.get("gold_sem_class") or n_e["type"]
                concept_cands = neural_ent_rec.get("concept_candidates") or []
                anchors = tables.type_to_anchors.get(gold_type)
                evaluable = bool(anchors) and bool(concept_cands)

                # Track CSP type-flips for transparency.
                csp_type_flipped = (c_e["type"] != gold_type)
                if csp_type_flipped:
                    type_flips_csp += 1

                if evaluable:
                    # MRCM-aligned (anchor-inclusive): the notion the CSP enforces.
                    # Neural: top-1 compliance under gold type.
                    n_compliant = entity_coherent(gold_type, n_e.get("concept"), tables)
                    n_strict   = entity_coherent_strict(gold_type, n_e.get("concept"), tables)
                    neural_metrics.ent_total += 1
                    neural_metrics.ent_coherent += int(bool(n_compliant))
                    neural_metrics.ent_strict_total += 1
                    neural_metrics.ent_strict_coherent += int(bool(n_strict))

                    # Oracle: any of the top-K satisfy under gold type?
                    any_compliant = any(
                        entity_coherent(gold_type, cand["id"], tables)
                        for cand in concept_cands
                    )
                    any_strict = any(
                        entity_coherent_strict(gold_type, cand["id"], tables)
                        for cand in concept_cands
                    )
                    oracle_metrics.ent_total += 1
                    oracle_metrics.ent_coherent += int(bool(any_compliant))
                    oracle_metrics.ent_strict_total += 1
                    oracle_metrics.ent_strict_coherent += int(bool(any_strict))

                    # CSP: faithful accounting -- if CSP flipped type to !=gold,
                    # count as incoherent (it failed to find a coherent
                    # assignment under the gold type). If CSP kept gold type,
                    # check whether its chosen concept is hierarchy-compliant.
                    csp_metrics.ent_total += 1
                    csp_metrics.ent_strict_total += 1
                    if csp_type_flipped:
                        csp_metrics.ent_coherent += 0
                        csp_metrics.ent_strict_coherent += 0
                    else:
                        c_compliant = entity_coherent(gold_type, c_e.get("concept"), tables)
                        c_strict    = entity_coherent_strict(gold_type, c_e.get("concept"), tables)
                        csp_metrics.ent_coherent += int(bool(c_compliant))
                        csp_metrics.ent_strict_coherent += int(bool(c_strict))

                # Norm top-1 against gold (entities with verified gold concept only).
                gold_cid = gold_concept_for_entity(gold, e_idx) if gold else None
                if gold_cid is not None:
                    if n_e.get("concept"):
                        neural_metrics.norm_evaluable += 1
                        if str(n_e["concept"]) == gold_cid:
                            neural_metrics.norm_correct += 1
                    if c_e.get("concept"):
                        csp_metrics.norm_evaluable += 1
                        if str(c_e["concept"]) == gold_cid:
                            csp_metrics.norm_correct += 1
                    # Oracle for norm: did the gold concept appear anywhere in top-K?
                    if concept_cands:
                        oracle_metrics.norm_evaluable += 1
                        if any(str(cand["id"]) == gold_cid for cand in concept_cands):
                            oracle_metrics.norm_correct += 1

                if n_e.get("concept") != c_e.get("concept"):
                    overrides_concept += 1

            for n_p, c_p in zip(neural["pairs"], csp_assn["pairs"]):
                i, j = n_p["i"], n_p["j"]
                # Resolve entity types from each configuration's entities list.
                ta_n = neural["entities"][i]["type"] if i < len(neural["entities"]) else "none"
                tb_n = neural["entities"][j]["type"] if j < len(neural["entities"]) else "none"
                ta_c = csp_assn["entities"][i]["type"] if i < len(csp_assn["entities"]) else "none"
                tb_c = csp_assn["entities"][j]["type"] if j < len(csp_assn["entities"]) else "none"

                neural_rc = relation_coherent(n_p["relation"], ta_n, tb_n, tables)
                csp_rc = relation_coherent(c_p["relation"], ta_c, tb_c, tables)
                if neural_rc is not None:
                    neural_metrics.tier1_total += 1
                    neural_metrics.tier1_coherent += int(neural_rc)
                if csp_rc is not None:
                    csp_metrics.tier1_total += 1
                    csp_metrics.tier1_coherent += int(csp_rc)

                # Relation F1 against gold
                neural_metrics.pred_label_counts[n_p["relation"]] += 1
                csp_metrics.pred_label_counts[c_p["relation"]] += 1
                if gold is not None:
                    g = gold_relation_for_pair(gold, i, j)
                    g_id = RELATION_LABEL_TO_ID.get(g, NO_RELATION_ID)
                    n_id = RELATION_LABEL_TO_ID.get(n_p["relation"], NO_RELATION_ID)
                    c_id = RELATION_LABEL_TO_ID.get(c_p["relation"], NO_RELATION_ID)
                    neural_metrics.rel_pred.append(n_id)
                    neural_metrics.rel_gold.append(g_id)
                    csp_metrics.rel_pred.append(c_id)
                    csp_metrics.rel_gold.append(g_id)

                if n_p["relation"] != c_p["relation"]:
                    overrides_relation += 1

            row = {
                "pmid": pmid,
                "neural": neural,
                "csp": csp_assn,
                "csp_status": sol.get("status", "unknown"),
            }
            out_fh.write(json.dumps(row) + "\n")

    logger.info(f"  CSP + accounting done in {time.time() - t0:.1f}s")

    n_dict = neural_metrics.to_dict()
    c_dict = csp_metrics.to_dict()
    o_dict = oracle_metrics.to_dict()
    summary = {
        "model_dir":     args.model_dir,
        "split":         args.split,
        "split_path":    str(split_path),
        "documents":     len(neural_records),
        "csp_status":    dict(csp_status_counts),
        "concept_overrides_csp_vs_neural":  overrides_concept,
        "relation_overrides_csp_vs_neural": overrides_relation,
        "csp_type_flips_to_non_gold":       type_flips_csp,
        "methodology": {
            "denominator": (
                "Locked: entities whose gold sem_class has hierarchy anchors AND "
                "has >= 1 concept candidate. Both neural and CSP are scored on the "
                "same population. CSP type-flips (gold->none) count as incoherent, "
                "not as denominator drops."
            ),
            "neural":  "top-1 cosine concept; checks whether it lies under gold type's anchor.",
            "oracle":  "top-K compliance: does ANY top-K concept lie under gold type's anchor? Upper bound for what CSP can extract.",
            "csp":     "Z3-chosen concept under gold type's anchor. Type-flips counted as failures.",
            "top_k_concepts": args.top_k_concepts,
        },
        "neural":  n_dict,
        "oracle":  o_dict,
        "csp":     c_dict,
        "delta_csp_minus_neural": {
            "entity_validity_rate":  round(c_dict["entity_validity_rate"]  - n_dict["entity_validity_rate"], 4),
            "tier1_validity_rate":   round(c_dict["tier1_validity_rate"]   - n_dict["tier1_validity_rate"], 4),
            "norm_top1_accuracy":    round(c_dict["norm_top1_accuracy"]    - n_dict["norm_top1_accuracy"], 4),
            "relation_macro_f1":     round(c_dict["relation_macro_f1"]     - n_dict["relation_macro_f1"], 4),
        },
        "delta_oracle_minus_neural": {
            "entity_validity_rate":  round(o_dict["entity_validity_rate"]  - n_dict["entity_validity_rate"], 4),
            "norm_top1_accuracy":    round(o_dict["norm_top1_accuracy"]    - n_dict["norm_top1_accuracy"], 4),
        },
        "delta_csp_minus_oracle": {
            "entity_validity_rate":  round(c_dict["entity_validity_rate"]  - o_dict["entity_validity_rate"], 4),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"summary -> {SUMMARY_OUT}")

    # Console table
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    logger.info("=" * 92)
    logger.info(f"{'Metric':<32}{'Neural top-1':>14}{'Oracle top-K':>14}{'CSP':>14}{'CSP - Neural':>16}")
    logger.info("-" * 92)
    for label, key in [
        ("entity validity (MRCM)",    "entity_validity_rate"),
        ("entity validity (strict)",  "entity_strict_validity_rate"),
        ("tier-1 validity rate (ii)", "tier1_validity_rate"),
        ("norm top-1 accuracy",       "norm_top1_accuracy"),
        ("relation macro F1",         "relation_macro_f1"),
    ]:
        n = n_dict[key]; o = o_dict[key]; c = c_dict[key]
        logger.info(f"{label:<32}{_fmt(n):>14}{_fmt(o):>14}{_fmt(c):>14}{_fmt(c - n):>16}")
    logger.info("=" * 92)
    logger.info(f"  evaluable entity population (locked denominator): {n_dict['entity_evaluated']:,}")
    logger.info(f"  concept overrides (CSP vs neural top-1): {overrides_concept:,}")
    logger.info(f"  relation overrides (CSP vs neural top-1): {overrides_relation:,}")
    logger.info(f"  CSP type-flips (gold sem_class -> non-gold): {type_flips_csp:,}")
    logger.info(f"  CSP status: {dict(csp_status_counts)}")
    logger.info(f"  ORACLE - NEURAL = headroom in the bi-encoder's top-K")
    logger.info(f"  CSP    - ORACLE = symbolic layer's contribution beyond what bi-encoder already surfaces")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(config.STAGE2_DIR / "best"),
                        help="Stage-2 (or Stage-3) model directory.")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--split-path", default=None,
                        help="Override split path (default: outputs/phase2/splits/<split>.jsonl).")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k-concepts", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    logger = setup_logging(LOG_PATH)
    evaluate(args, logger)


if __name__ == "__main__":
    main()
