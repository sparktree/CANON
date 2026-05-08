"""CANON top-level entry point.

Runs every implemented phase step end-to-end. Update this file as new steps
land so a single command exercises the whole pipeline.

Currently implemented:
    Phase 1.1 -- UMLS RRF parser + pickle cache         (umls_query.py)
    Phase 1.2 -- MeSH -> SNOMED mapping pipeline        (mesh_to_snomed.py)
    Phase 1.3 -- Non-MeSH vocabulary scoping audit      (entity_scope.py + scope_audit.py)
    Phase 1.4 -- Relation schema alignment table        (relation_schema.py)
    Phase 1.5 -- MRCM constraint dictionary             (mrcm.py)
    Phase 1.6 -- SNOMED hierarchy graph                 (snomed_hierarchy.py)
    Phase 1.7 -- Active-release mapping verification    (mapping_verify.py)
    Phase 2.1 -- Unified annotation format + converters (unified_format.py + corpus_convert.py)
    Phase 2.2 -- Apply SNOMED concept mappings          (concept_map.py)
    Phase 2.3 -- Apply unified relation labels          (relation_map.py)
    Phase 2.4 -- Soft mapping preprocessing             (soft_map.py)
    Phase 2.5 -- SNOMED-derived synthetic Tier-1 data   (snomed_synth.py)
    Phase 2.6 -- PubTator3 silver-data acquisition      (silver_pubtator.py)
                 [gated by env var CANON_DOWNLOAD_SILVER=1]
    Phase 2.7 -- Train/Dev/Test split assembly          (assemble_splits.py)
    Phase 3.1 -- SapBERT-style pre-training of BioLinkBERT (sapbert_pretrain.py)
                 [HPC-targeted; main.py runs --smoke-test mode for orchestration sanity]
    Phase 3.2 -- Concept index + multi-task model self-test  (build_concept_index.py + heads.py)
    Phase 3.3 -- Stage 1 per-head training                   (train_stage1.py)
    Phase 3.4 -- Stage 2 joint multi-task training           (train_stage2.py)
    Phase 3.5 -- CSP solver (Z3 + MRCM constraints)          (csp_solver.py)
    Phase 3.6 -- Stage 3 CSP-feedback fine-tune              (train_stage3.py)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import assemble_splits  # noqa: E402
import concept_map  # noqa: E402
import corpus_convert  # noqa: E402
import entity_scope  # noqa: E402
import mapping_verify  # noqa: E402
import mesh_to_snomed  # noqa: E402
import mrcm  # noqa: E402
import relation_map  # noqa: E402
import relation_schema  # noqa: E402
import scope_audit  # noqa: E402
import silver_pubtator  # noqa: E402
import snomed_hierarchy  # noqa: E402
import snomed_synth  # noqa: E402
import soft_map  # noqa: E402
import umls_query  # noqa: E402


def _banner(text: str) -> None:
    bar = "=" * len(text)
    print(f"\n{bar}\n{text}\n{bar}", flush=True)


def step_1_1(force_reparse: bool) -> None:
    _banner("Phase 1.1 -- UMLS RRF parser + pickle cache")
    t0 = time.time()
    umls_query.preload(force=force_reparse)
    print(
        f"[1.1] CUIs={len(umls_query.cui_to_atoms):,}  "
        f"(sab,code) keys={len(umls_query.code_to_cuis):,}  "
        f"CUIs-with-rels={len(umls_query.cui_to_rels):,}  "
        f"CUIs-with-stys={len(umls_query.cui_to_stys):,}  "
        f"MRMAP from-keys={len(umls_query.mrmap_entries):,}"
    )
    print(f"[1.1] elapsed {time.time() - t0:.1f}s")


def step_1_2() -> None:
    _banner("Phase 1.2 -- MeSH -> SNOMED concept mapping")
    t0 = time.time()
    result = mesh_to_snomed.build_mapping(verbose=True)
    mapped = len(result["mapping_rows"])
    unmapped = len(result["unmapped_rows"])
    total = mapped + unmapped
    pct = (mapped / total * 100) if total else 0.0
    print(f"[1.2] mapped {mapped:,} / {total:,} ({pct:.1f}%); unmapped {unmapped:,}")
    print(f"[1.2] outputs in {mesh_to_snomed.OUTPUT_DIR}")
    print(f"[1.2] elapsed {time.time() - t0:.1f}s")


def step_1_3() -> None:
    _banner("Phase 1.3 -- Non-MeSH vocabulary scoping audit")
    t0 = time.time()
    in_scope = sum(1 for s in entity_scope.iter_specs() if s.snomed_normalized)
    out_scope = sum(1 for s in entity_scope.iter_specs() if not s.snomed_normalized)
    print(f"[1.3] registry: {in_scope} SNOMED-normalized types, {out_scope} NER-only types")
    scope_audit.run(verbose=True)
    print(f"[1.3] elapsed {time.time() - t0:.1f}s")


def step_1_4() -> None:
    _banner("Phase 1.4 -- Relation schema alignment")
    t0 = time.time()
    rows = list(relation_schema.iter_rows())
    tier1 = sum(1 for r in rows if r.tier == 1)
    tier2 = sum(1 for r in rows if r.tier == 2)
    print(f"[1.4] {len(rows)} mapping rows  ({tier1} Tier-1, {tier2} Tier-2)")
    out = relation_schema.dump_csv()
    print(f"[1.4] CSV written to {out}")
    print(f"[1.4] elapsed {time.time() - t0:.1f}s")


def step_1_5() -> None:
    _banner("Phase 1.5 -- MRCM constraint dictionary")
    t0 = time.time()
    out = mrcm.main(verbose=True)
    print(f"[1.5] JSON written to {out}")
    print(f"[1.5] elapsed {time.time() - t0:.1f}s")


def step_1_6(force_reparse: bool = False) -> None:
    _banner("Phase 1.6 -- SNOMED Hierarchy Graph")
    t0 = time.time()
    out = snomed_hierarchy.main(force=force_reparse, verbose=True)
    print(f"[1.6] stats written to {out}")
    print(f"[1.6] elapsed {time.time() - t0:.1f}s")


def step_1_7() -> None:
    _banner("Phase 1.7 -- Active-release mapping verification")
    t0 = time.time()
    summary = mapping_verify.verify(verbose=True)
    if summary["mapped_rows_inactive"]:
        print(
            f"[1.7] {summary['mapped_rows_inactive']} mapped concepts are retired in the loaded "
            f"SNOMED release; Phase 2.2 must apply a policy (drop / redirect / keep)."
        )
    print(f"[1.7] elapsed {time.time() - t0:.1f}s")


def step_2_1() -> None:
    _banner("Phase 2.1 -- Unified annotation format + converters")
    t0 = time.time()
    summary = corpus_convert.convert_all(verbose=True)
    converted = sum(
        1 for v in summary["corpora"].values() if v.get("status") == "converted"
    )
    print(f"[2.1] {converted} corpus paths converted to unified JSONL")
    print(f"[2.1] elapsed {time.time() - t0:.1f}s")


def step_2_2() -> None:
    _banner("Phase 2.2 -- Apply SNOMED concept mappings")
    t0 = time.time()
    summary = concept_map.apply_all(verbose=True)
    total_docs = sum(
        split_data.get("documents", 0)
        for corpus_splits in summary["corpora"].values()
        for split_data in corpus_splits.values()
    )
    print(f"[2.2] {total_docs:,} documents stamped with SNOMED mappings")
    print(f"[2.2] elapsed {time.time() - t0:.1f}s")


def step_2_3() -> None:
    _banner("Phase 2.3 -- Apply unified relation labels")
    t0 = time.time()
    summary = relation_map.apply_all(verbose=True)
    agg = summary["aggregate"]
    print(
        f"[2.3] {agg['total_relations']:,} relations stamped  "
        f"(tier1={agg['tier1']:,}, tier2={agg['tier2']:,}, "
        f"multi-candidate={agg['multi_candidate']:,})"
    )
    print(f"[2.3] elapsed {time.time() - t0:.1f}s")


def step_2_4() -> None:
    _banner("Phase 2.4 -- Soft mapping preprocessing")
    t0 = time.time()
    out = soft_map.apply_all(verbose=True)
    print(f"[2.4] lookup -> {out}")
    print(f"[2.4] elapsed {time.time() - t0:.1f}s")


def step_2_5() -> None:
    _banner("Phase 2.5 -- SNOMED-derived synthetic Tier-1 data")
    t0 = time.time()
    summary = snomed_synth.generate_all(verbose=True)
    for attr, n in summary["counts_by_attribute"].items():
        print(f"[2.5]   {attr}: {n:,} triples")
    print(f"[2.5] total: {summary['total_documents']:,} synthetic documents")
    print(f"[2.5] elapsed {time.time() - t0:.1f}s")


def step_2_6() -> None:
    _banner("Phase 2.6 -- PubTator3 silver-data acquisition")
    t0 = time.time()
    summary = silver_pubtator.apply_all(verbose=True)
    if summary.get("status") == "completed":
        print(f"[2.6] {summary['documents_written']:,} silver documents written")
    print(f"[2.6] elapsed {time.time() - t0:.1f}s")


def step_2_7() -> None:
    _banner("Phase 2.7 -- Train/Dev/Test split assembly")
    t0 = time.time()
    summary = assemble_splits.assemble(verbose=True)
    counts = summary["documents_written"]
    print(f"[2.7] train={counts['train']:,}  dev={counts['dev']:,}  test={counts['test']:,}")
    print(f"[2.7] elapsed {time.time() - t0:.1f}s")


def step_3_1() -> None:
    """Run a SapBERT smoke test locally so the orchestration is self-testable.

    Full SapBERT pre-training is HPC-only; submit slurm/sapbert.slurm on
    BigRed200 for the production run. Locally we exercise the pipeline at
    1 epoch / ~1K pairs / batch 8 (~5 min on CPU) so any code-path error
    surfaces here rather than after a Slurm queue wait.
    """
    _banner("Phase 3.1 -- SapBERT pre-training (SMOKE TEST)")
    print(
        "[3.1] Running smoke test only (1 epoch, ~1K pairs, batch 8). "
        "Submit slurm/sapbert.slurm for production training on BigRed200."
    )
    import subprocess
    t0 = time.time()
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "sapbert_pretrain.py"),
        "--smoke-test",
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[3.1] smoke test failed (rc={rc})")
    print(f"[3.1] elapsed {time.time() - t0:.1f}s")


def _run_subprocess(script: str, label: str, extra_args: list[str] | None = None) -> int:
    import subprocess
    cmd = [sys.executable, str(SCRIPTS_DIR / script), "--smoke-test"]
    if extra_args:
        cmd.extend(extra_args)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[{label}] smoke test failed (rc={rc})")
    return rc


def step_3_2() -> None:
    """Build the concept index (smoke-cap 5K) and assemble the multi-task model.

    Production indexing of all ~370K active SNOMED concepts runs via
    slurm/build_concept_index.slurm. Locally we build a small cache and verify
    that MultiTaskModel forward()s a real dev document without exception.
    """
    _banner("Phase 3.2 -- Concept index + multi-task model self-test")
    t0 = time.time()
    rc = _run_subprocess("build_concept_index.py", "3.2")
    if rc != 0:
        print(f"[3.2] elapsed {time.time() - t0:.1f}s")
        return
    # Forward self-test
    try:
        from transformers import AutoTokenizer
        import canon_dataset  # type: ignore
        import config  # type: ignore
        import heads  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(str(config.SAPBERT_ENCODER_DIR))
        soft = canon_dataset.load_soft_lookup(config.SOFT_MAPPING_LOOKUP)
        ds = canon_dataset.CanonDocDataset(
            config.PHASE2_SPLITS_DIR / "dev.jsonl",
            tokenizer,
            soft,
            max_docs=1,
            max_pairs=16,
        )
        feats = list(ds)
        if not feats:
            print("[3.2] dev split produced no features; skipping self-test")
            return
        batch = canon_dataset.collate_docs(feats, pad_token_id=tokenizer.pad_token_id or 0)
        import json as _json
        with open(config.CONCEPT_INDEX_IDS) as fh:
            num_concepts = len(_json.load(fh))
        model = heads.MultiTaskModel(str(config.SAPBERT_ENCODER_DIR), num_concepts=num_concepts)
        model.norm_head.load_concept_index(config.CONCEPT_INDEX_IDS, config.CONCEPT_INDEX_EMB)
        model.eval()
        out = model(batch)
        print(f"[3.2] forward self-test OK; losses: { {k: float(v) for k,v in out['losses'].items()} }")
    except Exception as exc:  # noqa: BLE001
        print(f"[3.2] forward self-test failed: {exc}")
    print(f"[3.2] elapsed {time.time() - t0:.1f}s")


def step_3_3() -> None:
    _banner("Phase 3.3 -- Stage 1 per-head training (SMOKE TEST)")
    t0 = time.time()
    for head in ("ner", "norm", "rel"):
        print(f"[3.3] training head: {head}")
        _run_subprocess("train_stage1.py", "3.3", ["--head", head])
    print(f"[3.3] elapsed {time.time() - t0:.1f}s")


def step_3_4() -> None:
    _banner("Phase 3.4 -- Stage 2 joint multi-task training (SMOKE TEST)")
    t0 = time.time()
    _run_subprocess("train_stage2.py", "3.4")
    print(f"[3.4] elapsed {time.time() - t0:.1f}s")


def step_3_5() -> None:
    _banner("Phase 3.5 -- CSP solver (SMOKE TEST)")
    t0 = time.time()
    _run_subprocess("csp_solver.py", "3.5")
    print(f"[3.5] elapsed {time.time() - t0:.1f}s")


def step_3_6() -> None:
    _banner("Phase 3.6 -- Stage 3 CSP-feedback fine-tune (SMOKE TEST)")
    t0 = time.time()
    _run_subprocess("train_stage3.py", "3.6")
    print(f"[3.6] elapsed {time.time() - t0:.1f}s")


STEPS = {
    "1.1": step_1_1,
    "1.2": step_1_2,
    "1.3": step_1_3,
    "1.4": step_1_4,
    "1.5": step_1_5,
    "1.6": step_1_6,
    "1.7": step_1_7,
    "2.1": step_2_1,
    "2.2": step_2_2,
    "2.3": step_2_3,
    "2.4": step_2_4,
    "2.5": step_2_5,
    "2.6": step_2_6,
    "2.7": step_2_7,
    "3.1": step_3_1,
    "3.2": step_3_2,
    "3.3": step_3_3,
    "3.4": step_3_4,
    "3.5": step_3_5,
    "3.6": step_3_6,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all implemented CANON steps.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(STEPS),
        help="Run only the listed step IDs (default: all).",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Ignore pickled caches and re-parse from source files (affects 1.1 and 1.6).",
    )
    args = parser.parse_args()

    selected = args.only or sorted(STEPS)
    overall = time.time()
    for step_id in selected:
        if step_id in ("1.1", "1.6"):
            STEPS[step_id](force_reparse=args.force_reparse)  # type: ignore[call-arg]
        else:
            STEPS[step_id]()  # type: ignore[operator]
    print(f"\n[main] all selected steps done in {time.time() - overall:.1f}s")


if __name__ == "__main__":
    main()
