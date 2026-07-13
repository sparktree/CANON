"""Step-2 targeted PubTator3 retrieval for real-context Tier-1 grounding.

Step 1 (contextualize_synthetic.py) grounds Tier-1 relations in the train + silver
documents that already exist. Step 2 raises coverage by retrieving MORE real
abstracts: it scans the bulk PubTator3 files (local) for PubMed abstracts that
co-mention a chemical-disease (or disease-disease) pair SNOMED attests as
causative-agent / due-to / after (ancestor-aware), fetches those abstracts' text
from the PubTator3 API, and writes them as unified documents. contextualize then
grounds them the same way it grounds train/silver -- co-mention plus a SNOMED
stated attribute, real text, no label leakage.

Reuse: the Phase 2.6 machinery (footprint, bulk scan, BioC-XML fetch/parse) and
the calibration's SNOMED attestation index. The scan and the attestation match
are local; the network is used only to fetch text for matched PMIDs. Gated by
CANON_DOWNLOAD_PUBTATOR_CONTEXT=1, mirroring Phase 2.6's silver gate, so a default
run stays offline.

There is no coverage target or gate: the amount retrieved is whatever the caps
(abstracts per attested pair, total PMIDs) and the data allow, reported as
evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import concept_map
    import config
    import silver_pubtator as sp
    from calibrate_relation_priors import load_ancestors, load_attribute_index
    from unified_format import Document, EntityMention, read_jsonl, write_jsonl
    import entity_scope
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import concept_map
    import config
    import silver_pubtator as sp
    from calibrate_relation_priors import load_ancestors, load_attribute_index
    from unified_format import Document, EntityMention, read_jsonl, write_jsonl
    import entity_scope


_ATTR_CAUSATIVE_AGENT = "246075003"
_ATTR_DUE_TO = "42752001"
_ATTR_AFTER = "255234002"

ENV_GATE = "CANON_DOWNLOAD_PUBTATOR_CONTEXT"

# Caps (configurable). Bound the fetch volume while keeping diversity.
MAX_PER_PAIR = 25        # abstracts per attested (disease, chemical/disease) pair
MAX_PMIDS = 20_000       # overall ceiling on PMIDs fetched

CACHE_DIR = sp.SILVER_DIR / "raw" / "biocxml_context"
OUTPUT_JSONL = config.PHASE2_DIR / "silver" / "PubTator3" / "retrieved_context.jsonl"
SUMMARY_JSON = config.PHASE2_DIR / "silver" / "pubtator3_context_summary.json"

_MAP_MIN_CONF = sp.ENTITY_MAP_MIN_CONF
_CONTEXT_CONFIDENCE = 0.6  # distant-supervision weight (matches Step-1 grounding)


# ---------------------------------------------------------------------------
# MeSH -> SNOMED resolution + attestation
# ---------------------------------------------------------------------------
def _mesh_to_sctid(mesh: str, table: Dict[str, "concept_map._MappingEntry"]) -> Optional[str]:
    entry = concept_map._best_entry_for_code(mesh, table)
    if entry is not None and entry.confidence >= _MAP_MIN_CONF and entry.active:
        return entry.snomed_id
    return None


class _AttestedPairs:
    """Precomputed MeSH-pair attestation sets, so per-PMID selection is O(1).

    Built once from the corpus footprint by resolving each MeSH to SNOMED and
    checking the ancestor-aware stated-relationship index. Also exposes the MeSH
    that participate in some attested pair, used to narrow the bulk scan to the
    relevant footprint (instead of every corpus chemical/disease, which matches
    ~10M abstracts).
    """

    def __init__(self, table, attr_index, ancestors):
        ca = attr_index[_ATTR_CAUSATIVE_AGENT]
        dt = attr_index[_ATTR_DUE_TO]
        af = attr_index[_ATTR_AFTER]

        chems, diseases = sp.build_target_footprint()
        chem_sct = {m: s for m in chems if (s := _mesh_to_sctid(m, table))}
        dis_sct = {m: s for m in diseases if (s := _mesh_to_sctid(m, table))}

        _anc_cache: Dict[str, Set[str]] = {}

        def sa(c: str) -> Set[str]:
            r = _anc_cache.get(c)
            if r is None:
                r = {c}
                r.update(ancestors.get(c, ()))
                _anc_cache[c] = r
            return r

        def targets(sct: str, idx) -> Set[str]:
            out: Set[str] = set()
            for s in sa(sct):
                d = idx.get(s)
                if d:
                    out.update(d)
            return out

        self.ca_pairs: Set[Tuple[str, str]] = set()   # (disease_mesh, chemical_mesh)
        self.dt_pairs: Set[Tuple[str, str]] = set()   # (disease_mesh, disease_mesh)
        self.af_pairs: Set[Tuple[str, str]] = set()
        self.part_chem: Set[str] = set()
        self.part_dis: Set[str] = set()

        dis_ca = {s: targets(s, ca) for s in set(dis_sct.values())}
        for dm, ds in dis_sct.items():
            tgt = dis_ca.get(ds)
            if not tgt:
                continue
            for cm, cs in chem_sct.items():
                if sa(cs) & tgt:
                    self.ca_pairs.add((dm, cm))
                    self.part_dis.add(dm)
                    self.part_chem.add(cm)

        dis_dt = {s: targets(s, dt) for s in set(dis_sct.values())}
        dis_af = {s: targets(s, af) for s in set(dis_sct.values())}
        dis_items = list(dis_sct.items())
        for am, asct in dis_items:
            for bm, bsct in dis_items:
                if am == bm:
                    continue
                bset = sa(bsct)
                if bset & dis_dt.get(asct, set()):
                    self.dt_pairs.add((am, bm))
                    self.part_dis.update((am, bm))
                if bset & dis_af.get(asct, set()):
                    self.af_pairs.add((am, bm))
                    self.part_dis.update((am, bm))

    def pairs_in_pmid(self, chem_meshes: Tuple[str, ...],
                      dis_meshes: Tuple[str, ...]) -> List[Tuple[str, str, str]]:
        out: List[Tuple[str, str, str]] = []
        for dm in dis_meshes:
            for cm in chem_meshes:
                if (dm, cm) in self.ca_pairs:
                    out.append(("causative-agent", dm, cm))
        for am in dis_meshes:
            for bm in dis_meshes:
                if am == bm:
                    continue
                if (am, bm) in self.dt_pairs:
                    out.append(("due-to", am, bm))
                if (am, bm) in self.af_pairs:
                    out.append(("after", am, bm))
        return out


# ---------------------------------------------------------------------------
# PMID selection (local: scan + attestation match)
# ---------------------------------------------------------------------------
def select_target_pmids(
    max_per_pair: int = MAX_PER_PAIR,
    max_pmids: int = MAX_PMIDS,
    verbose: bool = True,
) -> Tuple[List[str], dict]:
    table = concept_map.load_verified_table()
    attr_index = load_attribute_index()
    ancestors = load_ancestors()

    if verbose:
        print("[2.6b] precomputing SNOMED-attested MeSH pairs ...", flush=True)
    ap = _AttestedPairs(table, attr_index, ancestors)
    if verbose:
        print(f"[2.6b] attested pairs: causative-agent={len(ap.ca_pairs):,} "
              f"due-to={len(ap.dt_pairs):,} after={len(ap.af_pairs):,}; "
              f"participating MeSH: chem={len(ap.part_chem):,} disease={len(ap.part_dis):,}",
              flush=True)

    # Narrow the bulk scan to only MeSH that participate in an attested pair.
    chem_pmids, dis_pmids, scan_stats = sp.filter_pmids_in_domain(
        ap.part_chem, ap.part_dis, verbose=verbose)

    # Exclude PMIDs already fetched as silver (they are already Step-1 context).
    existing: Set[str] = set()
    if sp.OUTPUT_JSONL.exists():
        for doc in read_jsonl(sp.OUTPUT_JSONL):
            if doc.pmid:
                existing.add(str(doc.pmid))

    pair_counts: Counter = Counter()
    attr_pair_seen: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    selected: List[str] = []
    selected_set: Set[str] = set()

    # Richest PMIDs first so early picks maximise pair diversity.
    candidates = sorted(chem_pmids.keys(),
                        key=lambda p: len(chem_pmids.get(p, ())) + len(dis_pmids.get(p, ())),
                        reverse=True)
    for pmid in candidates:
        if len(selected) >= max_pmids:
            break
        if pmid in existing or pmid in selected_set:
            continue
        pairs = ap.pairs_in_pmid(chem_pmids.get(pmid, ()), dis_pmids.get(pmid, ()))
        if not pairs:
            continue
        if not any(pair_counts[(attr, sm, om)] < max_per_pair for attr, sm, om in pairs):
            continue
        selected.append(pmid)
        selected_set.add(pmid)
        for attr, sm, om in pairs:
            pair_counts[(attr, sm, om)] += 1
            attr_pair_seen[attr].add((sm, om))

    del chem_pmids, dis_pmids
    gc.collect()

    stats = {
        "scan": scan_stats,
        "attested_pairs": {"causative-agent": len(ap.ca_pairs),
                           "due-to": len(ap.dt_pairs), "after": len(ap.af_pairs)},
        "existing_silver_pmids_excluded": len(existing),
        "selected_pmids": len(selected),
        "distinct_attested_pairs_covered": {k: len(v) for k, v in attr_pair_seen.items()},
        "max_per_pair": max_per_pair,
        "max_pmids": max_pmids,
    }
    if verbose:
        print(f"[2.6b] selected {len(selected):,} PMIDs covering "
              f"{ {k: len(v) for k, v in attr_pair_seen.items()} } attested pairs", flush=True)
    return selected, stats


# ---------------------------------------------------------------------------
# Build unified documents (entities only; no relation requirement)
# ---------------------------------------------------------------------------
def _build_docs(xml_paths: List[Path], target_pmids: Set[str],
                table, verbose: bool = True) -> Tuple[List[Document], dict]:
    docs: List[Document] = []
    n_docs = 0
    n_kept = 0
    for xml_path in xml_paths:
        for raw in sp._parse_biocxml(xml_path):
            n_docs += 1
            pmid = raw["pmid"]
            if pmid not in target_pmids:
                continue
            entities: List[EntityMention] = []
            for i, ent in enumerate(raw["entities"]):
                etype, sem_class = sp._entity_type_to_class(ent["entity_type"])
                non_snomed = sem_class in entity_scope.NON_SNOMED_NER_CLASSES
                code_norm = sp._strip_mesh(ent["identifier_raw"]) if ent["identifier_raw"] else None
                mapped_id = None
                active = None
                conf = None
                if code_norm and not non_snomed and sem_class in {"chemical", "disease"}:
                    mapped_id = _mesh_to_sctid(code_norm, table)
                    if mapped_id is not None:
                        active = True
                        conf = _CONTEXT_CONFIDENCE
                entities.append(EntityMention(
                    id=f"T{i + 1}", span_start=ent["start"], span_end=ent["end"],
                    surface_text=ent["mention"], entity_type=etype,
                    semantic_class=sem_class or None, original_code=code_norm,
                    mapped_snomed_id=mapped_id, mapping_confidence=conf,
                    snomed_active=active, non_snomed=non_snomed,
                    extra={"retrieved_context": True, "source": "PubTator3"},
                ))
            has_chem = any(e.mapped_snomed_id and e.semantic_class == "chemical" for e in entities)
            has_dis = any(e.mapped_snomed_id and e.semantic_class == "disease" for e in entities)
            if not (has_chem and has_dis):
                continue
            docs.append(Document(
                pmid=pmid, corpus="PubTator3_context", split="train",
                title=raw["title"], abstract=raw["abstract"], text=raw["text"],
                entities=entities, relations=[],
            ))
            n_kept += 1
    if verbose:
        print(f"[2.6b] parsed {n_docs:,} XML docs -> kept {n_kept:,} retrieved-context docs", flush=True)
    return docs, {"xml_docs_parsed": n_docs, "docs_kept": n_kept}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def apply_all(verbose: bool = True) -> dict:
    if os.getenv(ENV_GATE) != "1":
        if verbose:
            print(f"[2.6b] {ENV_GATE} != 1; skipping targeted retrieval (offline default).",
                  flush=True)
        return {"status": "skipped", "reason": f"{ENV_GATE} not set"}

    selected, sel_stats = select_target_pmids(verbose=verbose)
    if not selected:
        return {"status": "no_pmids", "selection": sel_stats}

    xml_paths, fetch_stats = sp.fetch_biocxml(selected, verbose=verbose, cache_dir=CACHE_DIR)
    table = concept_map.load_verified_table()
    docs, build_stats = _build_docs(xml_paths, set(selected), table, verbose=verbose)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_written = write_jsonl(iter(docs), OUTPUT_JSONL)

    summary = {
        "status": "completed",
        "documents_written": n_written,
        "output": str(OUTPUT_JSONL),
        "selection": sel_stats,
        "fetch": fetch_stats,
        "build": build_stats,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"[2.6b] wrote {n_written:,} retrieved-context docs -> {OUTPUT_JSONL}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-2 targeted PubTator3 context retrieval.")
    parser.add_argument("--max-per-pair", type=int, default=MAX_PER_PAIR)
    parser.add_argument("--max-pmids", type=int, default=MAX_PMIDS)
    parser.add_argument("--select-only", action="store_true",
                        help="run the local scan/selection and print stats without fetching")
    args = parser.parse_args()
    if args.select_only:
        _, stats = select_target_pmids(args.max_per_pair, args.max_pmids)
        print(json.dumps(stats, indent=2))
    else:
        print(json.dumps(apply_all(), indent=2))


if __name__ == "__main__":
    main()
