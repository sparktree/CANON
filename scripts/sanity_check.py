"""Read-only sanity check for the CANON pipeline (Phases 1.1 - 2.7).

Runs seven tiers of cheap structural checks against the artifacts already on
disk and emits ``outputs/sanity_check_report.json`` plus a colored stdout
summary. Designed to be the regression gate before each phase boundary --
takes seconds, never modifies state.

Tiers:
    1. Existence & non-empty -- every claimed artifact is present.
    2. Counts vs summaries  -- summary JSONs agree with actual JSONL line counts.
    3. Schema integrity     -- every Document parses; spans + indices in range.
    4. Cross-phase invariants -- the high-value checks (verified mappings,
                                 tier-1 relations, hierarchy coverage, split
                                 disjointness, dev/test verification).
    5. Distributional spot-checks -- entity/relation/confidence histograms.
    6. Known gaps inventory -- static checklist of deferred or absent work.
    7. Code health         -- every scripts/ module imports cleanly.

Run with ``python main.py --only sanity`` once registered in main.STEPS.
"""

from __future__ import annotations

import csv
import importlib
import json
import pickle
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from config import REPO_ROOT, relative_to_repo
    from unified_format import Document, read_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import REPO_ROOT, relative_to_repo
    from unified_format import Document, read_jsonl


# ---------------------------------------------------------------------------
# ANSI color helpers (degrade to no-op if not a tty)
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty()
GREEN  = "\033[32m" if _COLOR else ""
RED    = "\033[31m" if _COLOR else ""
YELLOW = "\033[33m" if _COLOR else ""
DIM    = "\033[2m"  if _COLOR else ""
RESET  = "\033[0m"  if _COLOR else ""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

P1 = REPO_ROOT / "outputs" / "phase1"
P2 = REPO_ROOT / "outputs" / "phase2"

REPORT_PATH = REPO_ROOT / "outputs" / "sanity_check_report.json"

GOLD_CORPORA = ("BioRED", "BC5CDR")
SPLITS = ("train", "dev", "test")


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }


def _ok(name: str, detail: str = "", **metrics: Any) -> CheckResult:
    return CheckResult(name=name, passed=True, detail=detail, metrics=metrics)


def _fail(name: str, detail: str, **metrics: Any) -> CheckResult:
    return CheckResult(name=name, passed=False, detail=detail, metrics=metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_count(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def _exists_and_nonempty(path: Path) -> Tuple[bool, int]:
    if not path.exists():
        return False, 0
    return True, path.stat().st_size


# ---------------------------------------------------------------------------
# Tier 1 -- Existence & non-empty
# ---------------------------------------------------------------------------

def tier_1_existence() -> List[CheckResult]:
    expected = [
        # Phase 1
        P1 / "mesh_to_snomed_verified.csv",
        P1 / "mesh_to_snomed.csv",
        P1 / "mesh_to_snomed_inactive.csv",
        P1 / "mesh_to_snomed_unmapped.csv",
        P1 / "coverage_stats.csv",
        P1 / "low_confidence_top100.csv",
        P1 / "snomed_hierarchy.pkl",
        P1 / "snomed_ancestors.pkl",
        P1 / "snomed_hierarchy_stats.json",
        P1 / "mrcm_constraints.json",
        P1 / "relation_schema_alignment.csv",
        P1 / "mapping_verification_summary.json",
        P1 / "entity_scope_audit.csv",
        P1 / "entity_scope_summary.csv",
        # Phase 2 summaries
        P2 / "conversion_summary.json",
        P2 / "mapping_application_summary.json",
        P2 / "relation_mapping_summary.json",
        P2 / "soft_mapping_summary.json",
        P2 / "soft_mapping_lookup.json",
        # Phase 2 augmentation
        P2 / "synthetic" / "train.jsonl",
        P2 / "silver" / "PubTator3" / "train.jsonl",
        P2 / "silver" / "pubtator3_silver_summary.json",
        # Phase 2.7
        P2 / "splits" / "train.jsonl",
        P2 / "splits" / "dev.jsonl",
        P2 / "splits" / "test.jsonl",
        P2 / "splits" / "split_summary.json",
    ]
    # Per-corpus per-split jsonls
    for stage in ("unified", "mapped", "relation_mapped"):
        for corpus in GOLD_CORPORA:
            for split in SPLITS:
                expected.append(P2 / stage / corpus / f"{split}.jsonl")

    results: List[CheckResult] = []
    missing: List[str] = []
    empty: List[str] = []
    sizes: Dict[str, int] = {}
    for p in expected:
        ok, size = _exists_and_nonempty(p)
        rel = relative_to_repo(p)
        sizes[rel] = size
        if not ok:
            missing.append(rel)
        elif size == 0:
            empty.append(rel)

    if missing or empty:
        results.append(_fail(
            "tier1.artifacts_present",
            f"missing={len(missing)}, empty={len(empty)}",
            missing=missing, empty=empty, total_expected=len(expected),
        ))
    else:
        results.append(_ok(
            "tier1.artifacts_present",
            f"all {len(expected)} expected artifacts present and non-empty",
            total_expected=len(expected),
            total_bytes=sum(sizes.values()),
        ))
    return results


# ---------------------------------------------------------------------------
# Tier 2 -- Counts vs summaries
# ---------------------------------------------------------------------------

def tier_2_counts() -> List[CheckResult]:
    results: List[CheckResult] = []

    def _check_corpus_summary(summary_path: Path, jsonl_dir: Path,
                              label: str) -> CheckResult:
        if not summary_path.exists():
            return _fail(f"tier2.{label}", f"{summary_path.name} missing")
        summary = json.loads(summary_path.read_text())
        mismatches: List[Dict[str, Any]] = []
        seen = 0
        for corpus_name, corpus_data in summary.get("corpora", {}).items():
            if corpus_data.get("status") == "absent":
                continue
            splits_block = corpus_data.get("splits") or {
                k: v for k, v in corpus_data.items()
                if isinstance(v, dict) and "documents" in v
            }
            for split_name, split_data in splits_block.items():
                claimed = split_data.get("documents")
                jsonl = jsonl_dir / corpus_name / f"{split_name}.jsonl"
                if not jsonl.exists():
                    mismatches.append({"corpus": corpus_name, "split": split_name,
                                       "claimed": claimed, "actual": None,
                                       "reason": "jsonl_missing"})
                    continue
                actual = _line_count(jsonl)
                seen += 1
                if actual != claimed:
                    mismatches.append({"corpus": corpus_name, "split": split_name,
                                       "claimed": claimed, "actual": actual})
        if mismatches:
            return _fail(f"tier2.{label}",
                         f"{len(mismatches)} count mismatch(es)",
                         mismatches=mismatches)
        return _ok(f"tier2.{label}",
                   f"all {seen} (corpus, split) counts match summary")

    results.append(_check_corpus_summary(
        P2 / "conversion_summary.json", P2 / "unified", "conversion"))
    results.append(_check_corpus_summary(
        P2 / "mapping_application_summary.json", P2 / "mapped", "mapping"))

    # Phase 2.3 relation_mapping_summary uses the mapped jsonl as input. The
    # output relation_mapped jsonl per (corpus, split) preserves the document
    # count from Phase 2.2, so re-use the mapping summary's counts.
    rm_summary_path = P2 / "mapping_application_summary.json"
    if rm_summary_path.exists():
        summary = json.loads(rm_summary_path.read_text())
        mismatches = []
        seen = 0
        for corpus_name, corpus_data in summary.get("corpora", {}).items():
            if corpus_data.get("status") == "absent":
                continue
            splits_block = corpus_data.get("splits") or {
                k: v for k, v in corpus_data.items()
                if isinstance(v, dict) and "documents" in v
            }
            for split_name, split_data in splits_block.items():
                claimed = split_data.get("documents")
                jsonl = P2 / "relation_mapped" / corpus_name / f"{split_name}.jsonl"
                if not jsonl.exists():
                    mismatches.append({"corpus": corpus_name, "split": split_name,
                                       "claimed": claimed, "actual": None,
                                       "reason": "jsonl_missing"})
                    continue
                actual = _line_count(jsonl)
                seen += 1
                if actual != claimed:
                    mismatches.append({"corpus": corpus_name, "split": split_name,
                                       "claimed": claimed, "actual": actual})
        if mismatches:
            results.append(_fail("tier2.relation_mapped_counts",
                                 f"{len(mismatches)} mismatch(es)",
                                 mismatches=mismatches))
        else:
            results.append(_ok("tier2.relation_mapped_counts",
                               f"all {seen} relation_mapped counts match"))

    # Silver
    silver_summary = P2 / "silver" / "pubtator3_silver_summary.json"
    silver_jsonl   = P2 / "silver" / "PubTator3" / "train.jsonl"
    if silver_summary.exists() and silver_jsonl.exists():
        s = json.loads(silver_summary.read_text())
        claimed = s.get("documents_written")
        actual = _line_count(silver_jsonl)
        if claimed == actual:
            results.append(_ok("tier2.silver_count",
                               f"{actual:,} silver docs match summary"))
        else:
            results.append(_fail("tier2.silver_count",
                                 f"summary says {claimed}, jsonl has {actual}",
                                 claimed=claimed, actual=actual))

    # Splits
    split_summary = P2 / "splits" / "split_summary.json"
    if split_summary.exists():
        s = json.loads(split_summary.read_text())
        claimed = s.get("documents_written", {})
        actual = {sp: _line_count(P2 / "splits" / f"{sp}.jsonl")
                  for sp in SPLITS
                  if (P2 / "splits" / f"{sp}.jsonl").exists()}
        if claimed == actual:
            results.append(_ok("tier2.splits_count",
                               f"split counts match: {actual}",
                               counts=actual))
        else:
            results.append(_fail("tier2.splits_count",
                                 f"claimed={claimed} actual={actual}",
                                 claimed=claimed, actual=actual))
    return results


# ---------------------------------------------------------------------------
# Tier 3 -- Schema integrity
# ---------------------------------------------------------------------------

def _iter_phase2_jsonls() -> Iterable[Path]:
    for stage in ("unified", "mapped", "relation_mapped"):
        for corpus in GOLD_CORPORA:
            for split in SPLITS:
                p = P2 / stage / corpus / f"{split}.jsonl"
                if p.exists():
                    yield p
    for p in (P2 / "synthetic" / "train.jsonl",
              P2 / "silver" / "PubTator3" / "train.jsonl",
              P2 / "splits" / "train.jsonl",
              P2 / "splits" / "dev.jsonl",
              P2 / "splits" / "test.jsonl"):
        if p.exists():
            yield p


def tier_3_schema() -> List[CheckResult]:
    results: List[CheckResult] = []
    bad: List[Dict[str, Any]] = []
    seen = 0
    for path in _iter_phase2_jsonls():
        rel = relative_to_repo(path)
        for line_no, doc in enumerate(read_jsonl(path), 1):
            seen += 1
            if not doc.pmid or not doc.corpus or not doc.split:
                bad.append({"file": rel, "line": line_no,
                            "issue": "missing pmid/corpus/split"})
                continue
            text_len = len(doc.text)
            for em in doc.entities:
                if not (0 <= em.span_start < em.span_end <= text_len):
                    bad.append({"file": rel, "line": line_no,
                                "issue": f"bad span [{em.span_start},{em.span_end}] vs text_len={text_len}",
                                "ent_id": em.id})
                    break
            else:
                # Span check passed; now check relation indices.
                n_ent = len(doc.entities)
                for r in doc.relations:
                    if not (0 <= r.subject_idx < n_ent and 0 <= r.object_idx < n_ent):
                        bad.append({"file": rel, "line": line_no,
                                    "issue": f"relation idx out of range "
                                             f"({r.subject_idx},{r.object_idx}) vs n_ent={n_ent}"})
                        break
            if len(bad) >= 25:  # short-circuit: report first 25
                break
        if len(bad) >= 25:
            break
    if bad:
        results.append(_fail("tier3.schema_integrity",
                             f"{len(bad)} bad doc(s) in {seen:,} scanned",
                             docs_scanned=seen,
                             first_failures=bad[:25]))
    else:
        results.append(_ok("tier3.schema_integrity",
                           f"{seen:,} documents validated",
                           docs_scanned=seen))
    return results


# ---------------------------------------------------------------------------
# Tier 4 -- Cross-phase invariants
# ---------------------------------------------------------------------------

def _load_verified_table() -> Dict[Tuple[str, str], bool]:
    """Return {(mesh_id, snomed_id): snomed_active}."""
    table: Dict[Tuple[str, str], bool] = {}
    csv_path = P1 / "mesh_to_snomed_verified.csv"
    if not csv_path.exists():
        return table
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            mesh = (row.get("mesh_id") or "").strip()
            snomed = (row.get("snomed_id") or "").strip()
            active = (row.get("snomed_active") or "").lower() == "true"
            if mesh and snomed:
                table[(mesh, snomed)] = active
    return table


def _load_tier1_relations() -> Set[str]:
    csv_path = P1 / "relation_schema_alignment.csv"
    if not csv_path.exists():
        return set()
    out: Set[str] = set()
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                tier = int(row.get("tier", "0"))
            except ValueError:
                continue
            if tier == 1:
                tgt = (row.get("target_relation") or "").strip()
                if tgt:
                    out.add(tgt)
    return out


def _check_verified_consistency(table: Dict[Tuple[str, str], bool]) -> CheckResult:
    if not table:
        return _fail("tier4.verified_consistency",
                     "verified table empty -- cannot check")
    bad: List[Dict[str, Any]] = []
    seen = 0
    for corpus in GOLD_CORPORA:
        for split in SPLITS:
            p = P2 / "mapped" / corpus / f"{split}.jsonl"
            if not p.exists():
                continue
            for doc in read_jsonl(p):
                for em in doc.entities:
                    if em.mapped_snomed_id is None:
                        continue
                    code = (em.original_code or "").strip()
                    if code in ("", "-", "-1"):
                        continue
                    # BioRED/BC5CDR encode composite codes (one mention covering
                    # multiple MeSH descriptors) with either "," or "|" as the
                    # separator depending on the source PubTator file. Phase
                    # 2.2 picks the SNOMED ID for one component, so the
                    # (component, snomed) pair -- not the composite -- is what
                    # should match the verified table.
                    raw_components = code.replace("|", ",").split(",")
                    components = [c.strip() for c in raw_components if c.strip()]
                    seen += 1
                    snomed = em.mapped_snomed_id
                    matched_active: Optional[bool] = None
                    for comp in components:
                        active = table.get((comp, snomed))
                        if active is not None:
                            matched_active = active
                            break
                    if matched_active is None:
                        bad.append({"corpus": corpus, "split": split,
                                    "pmid": doc.pmid, "ent_id": em.id,
                                    "mesh": code, "snomed": snomed,
                                    "issue": "pair_not_in_verified_table"})
                    elif matched_active != bool(em.snomed_active):
                        bad.append({"corpus": corpus, "split": split,
                                    "pmid": doc.pmid, "ent_id": em.id,
                                    "mesh": code, "snomed": snomed,
                                    "issue": f"active_mismatch table={matched_active} entity={em.snomed_active}"})
                    if len(bad) >= 25:
                        return _fail("tier4.verified_consistency",
                                     f">=25 mismatches in {seen:,} entities scanned",
                                     entities_scanned=seen,
                                     first_failures=bad[:25])
    if bad:
        return _fail("tier4.verified_consistency",
                     f"{len(bad)} mismatch(es) in {seen:,} entities",
                     entities_scanned=seen, first_failures=bad[:25])
    return _ok("tier4.verified_consistency",
               f"{seen:,} mapped entities all consistent with verified table",
               entities_scanned=seen)


def _check_tier1_legitimacy(tier1_set: Set[str]) -> CheckResult:
    if not tier1_set:
        return _fail("tier4.tier1_legitimacy", "no tier-1 relations in schema")
    bad: List[Dict[str, Any]] = []
    seen = 0
    for corpus in GOLD_CORPORA:
        for split in SPLITS:
            p = P2 / "relation_mapped" / corpus / f"{split}.jsonl"
            if not p.exists():
                continue
            for doc in read_jsonl(p):
                for r in doc.relations:
                    if r.tier == 1:
                        seen += 1
                        if r.target_relation not in tier1_set:
                            bad.append({"corpus": corpus, "split": split,
                                        "pmid": doc.pmid,
                                        "target_relation": r.target_relation,
                                        "issue": "claims_tier1_but_not_in_schema"})
    if bad:
        return _fail("tier4.tier1_legitimacy",
                     f"{len(bad)} bogus tier-1 relation(s)",
                     tier1_relations_seen=seen, first_failures=bad[:25])
    return _ok("tier4.tier1_legitimacy",
               f"{seen:,} tier-1 relations all in {sorted(tier1_set)}",
               tier1_relations_seen=seen)


def _check_hierarchy_coverage() -> CheckResult:
    pkl = P1 / "snomed_hierarchy.pkl"
    if not pkl.exists():
        return _fail("tier4.hierarchy_coverage", "snomed_hierarchy.pkl missing")
    payload = pickle.loads(pkl.read_bytes())
    graph = payload["graph"] if isinstance(payload, dict) else payload
    nodes: Set[str] = {str(n) for n in graph.nodes()}
    seen_active: Set[str] = set()
    seen_inactive: Set[str] = set()
    for corpus in GOLD_CORPORA:
        for split in SPLITS:
            p = P2 / "mapped" / corpus / f"{split}.jsonl"
            if not p.exists():
                continue
            for doc in read_jsonl(p):
                for em in doc.entities:
                    sid = em.mapped_snomed_id
                    if not sid:
                        continue
                    if em.snomed_active:
                        seen_active.add(str(sid))
                    else:
                        seen_inactive.add(str(sid))
    missing_active = sorted(seen_active - nodes)
    missing_inactive = sorted(seen_inactive - nodes)
    if missing_active:
        return _fail("tier4.hierarchy_coverage",
                     f"{len(missing_active)}/{len(seen_active)} active SNOMED IDs absent from hierarchy",
                     active_missing=missing_active[:25],
                     inactive_missing_count=len(missing_inactive),
                     active_seen=len(seen_active),
                     inactive_seen=len(seen_inactive),
                     graph_nodes=len(nodes))
    return _ok("tier4.hierarchy_coverage",
               f"all {len(seen_active):,} active mapped SNOMED IDs present in hierarchy "
               f"({len(missing_inactive)} inactive IDs absent, expected)",
               active_seen=len(seen_active),
               inactive_seen=len(seen_inactive),
               inactive_missing_from_graph=len(missing_inactive),
               graph_nodes=len(nodes))


def _check_split_disjointness() -> CheckResult:
    """Disjointness at (corpus, pmid) granularity.

    Raw PMID overlap across splits is expected: BioRED is built on top of
    BC5CDR, so the same PubMed paper appears in both corpora with distinct
    annotation sets. The training-time invariant we actually need is that
    within a single corpus no PMID appears in more than one split. The
    raw overlap is reported as a warning so it stays visible.
    """
    keys_per_split: Dict[str, Set[Tuple[str, str]]] = {}
    raw_pmids_per_split: Dict[str, Set[str]] = {}
    for split in SPLITS:
        p = P2 / "splits" / f"{split}.jsonl"
        if not p.exists():
            return _fail("tier4.split_disjointness", f"{p} missing")
        keys: Set[Tuple[str, str]] = set()
        pmids: Set[str] = set()
        for doc in read_jsonl(p):
            keys.add((doc.corpus, doc.pmid))
            pmids.add(doc.pmid)
        keys_per_split[split] = keys
        raw_pmids_per_split[split] = pmids

    overlaps: Dict[str, int] = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        common = keys_per_split[a] & keys_per_split[b]
        if common:
            overlaps[f"{a}_x_{b}"] = len(common)

    raw_overlaps: Dict[str, int] = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        common = raw_pmids_per_split[a] & raw_pmids_per_split[b]
        if common:
            raw_overlaps[f"{a}_x_{b}"] = len(common)

    warnings: List[str] = []
    if raw_overlaps:
        warnings.append(
            f"raw PMID overlap (expected, BioRED ⊆ BC5CDR papers): {raw_overlaps}"
        )

    if overlaps:
        return CheckResult(
            name="tier4.split_disjointness",
            passed=False,
            detail=f"(corpus, pmid) overlap: {overlaps}",
            metrics={
                "key_overlaps": overlaps,
                "raw_pmid_overlaps": raw_overlaps,
                "split_sizes_keys": {k: len(v) for k, v in keys_per_split.items()},
            },
            warnings=warnings,
        )
    return CheckResult(
        name="tier4.split_disjointness",
        passed=True,
        detail="all 3 splits disjoint at (corpus, pmid) granularity",
        metrics={
            "raw_pmid_overlaps": raw_overlaps,
            "split_sizes_keys": {k: len(v) for k, v in keys_per_split.items()},
        },
        warnings=warnings,
    )


def _check_dev_test_purity() -> CheckResult:
    bad_corpus: List[Dict[str, Any]] = []
    bad_unverified = 0
    bad_unverified_examples: List[Dict[str, Any]] = []
    for split in ("dev", "test"):
        p = P2 / "splits" / f"{split}.jsonl"
        if not p.exists():
            continue
        for doc in read_jsonl(p):
            if doc.corpus not in GOLD_CORPORA:
                bad_corpus.append({"split": split, "pmid": doc.pmid,
                                   "corpus": doc.corpus})
            for em in doc.entities:
                if em.non_snomed:
                    continue
                if em.semantic_class not in ("chemical", "disease"):
                    continue
                code = (em.original_code or "").strip()
                if code in ("", "-", "-1"):
                    continue
                if em.mapped_snomed_id is None or em.snomed_active is not True:
                    bad_unverified += 1
                    if len(bad_unverified_examples) < 10:
                        bad_unverified_examples.append({
                            "split": split, "pmid": doc.pmid,
                            "ent_id": em.id, "code": code,
                            "snomed": em.mapped_snomed_id,
                            "active": em.snomed_active,
                        })
    if bad_corpus or bad_unverified:
        return _fail("tier4.dev_test_purity",
                     f"non_gold_corpus={len(bad_corpus)} unverified_entities={bad_unverified}",
                     non_gold_corpus_examples=bad_corpus[:25],
                     unverified_examples=bad_unverified_examples)
    return _ok("tier4.dev_test_purity",
               "dev/test contain only gold corpora and verified SNOMED entities")


def tier_4_cross_phase() -> List[CheckResult]:
    results: List[CheckResult] = []
    table = _load_verified_table()
    tier1 = _load_tier1_relations()
    results.append(_check_verified_consistency(table))
    results.append(_check_tier1_legitimacy(tier1))
    results.append(_check_hierarchy_coverage())
    results.append(_check_split_disjointness())
    results.append(_check_dev_test_purity())
    return results


# ---------------------------------------------------------------------------
# Tier 5 -- Distributional spot-checks
# ---------------------------------------------------------------------------

def tier_5_distributions() -> List[CheckResult]:
    results: List[CheckResult] = []

    def _scan(path: Path) -> Dict[str, Any]:
        sem_classes: Counter = Counter()
        target_relations: Counter = Counter()
        tier_counts: Counter = Counter()
        confidences: List[float] = []
        corpus_counts: Counter = Counter()
        for doc in read_jsonl(path):
            corpus_counts[doc.corpus] += 1
            for em in doc.entities:
                sem_classes[em.semantic_class or "<none>"] += 1
                if em.mapping_confidence is not None:
                    confidences.append(em.mapping_confidence)
            for r in doc.relations:
                target_relations[r.target_relation or "<none>"] += 1
                tier_counts[r.tier or 0] += 1
        # Confidence summary (no numpy required)
        if confidences:
            confidences.sort()
            n = len(confidences)
            conf_summary = {
                "n": n,
                "min": round(confidences[0], 3),
                "p25": round(confidences[n // 4], 3),
                "median": round(confidences[n // 2], 3),
                "p75": round(confidences[(3 * n) // 4], 3),
                "max": round(confidences[-1], 3),
                "mean": round(sum(confidences) / n, 3),
            }
        else:
            conf_summary = {"n": 0}
        return {
            "corpus_counts": dict(corpus_counts),
            "semantic_class_distribution": dict(sem_classes),
            "target_relation_distribution": dict(target_relations.most_common(15)),
            "tier_distribution": dict(tier_counts),
            "confidence_summary": conf_summary,
        }

    flagged_warnings: List[str] = []
    distros: Dict[str, Any] = {}
    for split in SPLITS:
        p = P2 / "splits" / f"{split}.jsonl"
        if not p.exists():
            continue
        d = _scan(p)
        distros[split] = d
        # Smoke checks
        sem = d["semantic_class_distribution"]
        total_ents = sum(sem.values())
        if total_ents == 0:
            flagged_warnings.append(f"{split}: zero entities total")
        else:
            for cls, n in sem.items():
                if cls in ("chemical", "disease") and n / total_ents > 0.95:
                    flagged_warnings.append(
                        f"{split}: {cls} = {n/total_ents:.0%} of entities (>95%)")
        if d["target_relation_distribution"] and \
                sum(d["target_relation_distribution"].values()) == 0:
            flagged_warnings.append(f"{split}: zero relations")

    results.append(CheckResult(
        name="tier5.distributions",
        passed=not flagged_warnings,
        detail=("clean" if not flagged_warnings
                else f"{len(flagged_warnings)} warning(s)"),
        metrics={"per_split": distros},
        warnings=flagged_warnings,
    ))
    return results


# ---------------------------------------------------------------------------
# Tier 6 -- Known gaps inventory (static)
# ---------------------------------------------------------------------------

def tier_6_known_gaps() -> List[CheckResult]:
    gaps: List[str] = []

    # Corpora that the plan calls for but Phase 2.1 marked absent
    conv_path = P2 / "conversion_summary.json"
    if conv_path.exists():
        s = json.loads(conv_path.read_text())
        for corpus, data in s.get("corpora", {}).items():
            if data.get("status") == "absent":
                gaps.append(f"corpus_absent:{corpus}")

    # Phase 3 / 4 deliverables that don't exist yet
    phase3_paths = {
        "encoder_checkpoint": REPO_ROOT / "outputs" / "phase3" / "encoder",
        "csp_module":         REPO_ROOT / "outputs" / "phase3" / "csp_solver",
        "stage1_logs":        REPO_ROOT / "outputs" / "phase3" / "training",
    }
    for label, p in phase3_paths.items():
        if not p.exists():
            gaps.append(f"phase3_pending:{label}")

    phase4_path = REPO_ROOT / "outputs" / "phase4"
    if not phase4_path.exists():
        gaps.append("phase4_pending:evaluation")

    return [CheckResult(
        name="tier6.known_gaps",
        passed=True,  # informational only
        detail=f"{len(gaps)} known gap(s) -- these are expected, not failures",
        metrics={"gaps": gaps},
        warnings=gaps,
    )]


# ---------------------------------------------------------------------------
# Tier 7 -- Code health
# ---------------------------------------------------------------------------

def tier_7_code_health() -> List[CheckResult]:
    scripts_dir = Path(__file__).resolve().parent
    failures: List[Dict[str, str]] = []
    succeeded: List[str] = []
    for py in sorted(scripts_dir.glob("*.py")):
        if py.stem in ("__init__", "sanity_check"):
            continue
        try:
            importlib.import_module(py.stem)
            succeeded.append(py.stem)
        except Exception as e:
            failures.append({
                "module": py.stem,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3),
            })
    if failures:
        return [_fail("tier7.imports",
                      f"{len(failures)} module(s) failed to import",
                      imported=succeeded, failures=failures)]
    return [_ok("tier7.imports",
                f"all {len(succeeded)} scripts/ modules import cleanly",
                imported=succeeded)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

TIERS = [
    ("Tier 1 -- Existence",          tier_1_existence),
    ("Tier 2 -- Counts",             tier_2_counts),
    ("Tier 3 -- Schema",             tier_3_schema),
    ("Tier 4 -- Cross-phase",        tier_4_cross_phase),
    ("Tier 5 -- Distributions",      tier_5_distributions),
    ("Tier 6 -- Known gaps",         tier_6_known_gaps),
    ("Tier 7 -- Code health",        tier_7_code_health),
]


def run(verbose: bool = True) -> Dict[str, Any]:
    report: Dict[str, Any] = {"tiers": {}}
    total_pass = 0
    total_fail = 0
    total_warn = 0
    for label, fn in TIERS:
        if verbose:
            print(f"\n{label}")
        try:
            tier_results = fn()
        except Exception as e:
            tier_results = [_fail(f"{label}.crashed", f"{type(e).__name__}: {e}")]
        tier_block = [r.to_dict() for r in tier_results]
        report["tiers"][label] = tier_block
        for r in tier_results:
            mark = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
            warn = f" {YELLOW}({len(r.warnings)} warn){RESET}" if r.warnings else ""
            if verbose:
                print(f"  {mark}  {r.name}{warn}  {DIM}{r.detail}{RESET}")
                if not r.passed:
                    for k, v in r.metrics.items():
                        if isinstance(v, list) and v:
                            print(f"        {k}: showing first {min(5, len(v))} of {len(v)}")
                            for item in v[:5]:
                                print(f"          - {item}")
                for w in r.warnings[:5]:
                    print(f"        {YELLOW}warn{RESET}: {w}")
            if r.passed:
                total_pass += 1
            else:
                total_fail += 1
            total_warn += len(r.warnings)

    report["totals"] = {"passed": total_pass, "failed": total_fail,
                         "warnings": total_warn}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")

    if verbose:
        bar = "=" * 60
        overall = (f"{GREEN}OK{RESET}" if total_fail == 0
                   else f"{RED}FAILURES PRESENT{RESET}")
        print(f"\n{bar}")
        print(f"sanity check {overall}: {total_pass} passed, "
              f"{total_fail} failed, {total_warn} warnings")
        print(f"report -> {relative_to_repo(REPORT_PATH)}")
        print(bar)

    return report


if __name__ == "__main__":
    summary = run(verbose=True)
    sys.exit(0 if summary["totals"]["failed"] == 0 else 1)
