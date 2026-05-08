"""PubTator 3.0 silver-data acquisition (CANON Phase 2.6).

Five-pass pipeline:

    1. Build target MeSH footprint -- read mesh_to_snomed_verified.csv,
       keep mesh_ids with confidence >= 0.8 and snomed_active. Split into
       target_chemicals and target_diseases by entity_classes column.
    2. Stream-filter the bulk PubTator3 per-bioconcept .gz files
       (chemical2pubtator3.gz + disease2pubtator3.gz). Build per-PMID
       chemical / disease concept sets, retaining only PMIDs that mention
       at least one in-footprint chemical AND one in-footprint disease.
    3. Sample <= SAMPLE_SIZE PMIDs by stratified-greedy set-cover so that
       each in-footprint MeSH descriptor appears in at least
       COVERAGE_TARGET sampled abstracts (long-tail safe-guard); fill any
       remaining slots by abstract richness.
    4. Fetch BioC-XML for sampled PMIDs from the PubTator 3.0 REST API in
       PMIDS_PER_BATCH-sized batches, cached on disk so re-runs are free.
    5. Parse, apply confidence filters, stamp SNOMED + relation mappings,
       set silver confidence weight, and write outputs/phase2/silver/
       PubTator3/train.jsonl.

Confidence filters (per the plan, refined by what the API actually exposes):

  * Entity:   mesh_to_snomed_verified.csv mapping_confidence >= 0.8 + active.
              PubTator 3.0's BioC-XML does NOT expose per-entity normalizer
              scores, so we rely on the MeSH->SNOMED filter alone for
              entity quality.
  * Relation: PubTator 3.0 BioC-XML emits <infon key="score">x</infon> per
              relation -- threshold REL_SCORE_THRESHOLD (default 0.7).

After filtering, every retained entity has its mapping_confidence overwritten
with SILVER_CONFIDENCE = 0.4 (mid of plan's 0.3-0.5 range), tagging the
document as silver tier for confidence-weighted loss in Phase 3.

Gating:
    Step 2.6 makes a network call (PubTator REST API). It is skipped by
    default unless the environment variable CANON_DOWNLOAD_SILVER=1 is set,
    so `python main.py` runs offline by default. The bulk .gz files must
    already be present at Data/PubTator3/.

Outputs:
    outputs/phase2/silver/raw/biocxml/batch_NNNN.xml  -- cached API responses
    outputs/phase2/silver/PubTator3/train.jsonl       -- final unified corpus
    outputs/phase2/silver/pubtator3_silver_summary.json
"""

from __future__ import annotations

import csv
import gc
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

try:
    from config import DATA_ROOT, REPO_ROOT, relative_to_repo
    import concept_map
    import entity_scope
    import relation_schema
    from unified_format import Document, EntityMention, Relation, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import DATA_ROOT, REPO_ROOT, relative_to_repo
    import concept_map
    import entity_scope
    import relation_schema
    from unified_format import Document, EntityMention, Relation, write_jsonl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL              = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml"
API_USER_AGENT       = "CANON/1.0 (research; pubtator3 silver acquisition)"
PMIDS_PER_BATCH      = 100
API_RATE_SLEEP       = 0.4   # seconds between batches; ~2.5 req/s, well under NCBI's 3 req/s
API_TIMEOUT_SECONDS  = 60
API_MAX_RETRIES      = 3

SAMPLE_SIZE          = 25_000  # 24.5% retention observed -> ~6,100 silver docs (>= plan's 5K floor)
COVERAGE_TARGET      = 3      # min sampled PMIDs per in-footprint MeSH descriptor
ENTITY_MAP_MIN_CONF  = 0.8    # plan threshold on Phase 1.2 mapping confidence
REL_SCORE_THRESHOLD  = 0.7    # PubTator3 relation `score` infon threshold
SILVER_CONFIDENCE    = 0.4    # silver-tier mapping confidence (plan range 0.3-0.5)

ENV_GATE             = "CANON_DOWNLOAD_SILVER"

# File paths
PUBTATOR_DIR     = DATA_ROOT / "PubTator3"
CHEM_GZ          = PUBTATOR_DIR / "chemical2pubtator3.gz"
DISEASE_GZ       = PUBTATOR_DIR / "disease2pubtator3.gz"

VERIFIED_CSV     = REPO_ROOT / "outputs" / "phase1" / "mesh_to_snomed_verified.csv"

SILVER_DIR       = REPO_ROOT / "outputs" / "phase2" / "silver"
CACHE_DIR        = SILVER_DIR / "raw" / "biocxml"
OUTPUT_JSONL     = SILVER_DIR / "PubTator3" / "train.jsonl"
SUMMARY_JSON     = SILVER_DIR / "pubtator3_silver_summary.json"


def _strip_mesh(cid: str) -> str:
    return cid[5:] if cid.startswith("MESH:") else cid


# ---------------------------------------------------------------------------
# Pass 1: target MeSH footprint
# ---------------------------------------------------------------------------

def build_target_footprint(min_confidence: float = ENTITY_MAP_MIN_CONF) -> Tuple[Set[str], Set[str]]:
    if not VERIFIED_CSV.exists():
        raise FileNotFoundError(
            f"{VERIFIED_CSV} not found; run Phase 1.7 (mapping_verify.py) first."
        )
    chems: Set[str] = set()
    diseases: Set[str] = set()
    with VERIFIED_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                conf = float(row.get("confidence") or 0.0)
            except ValueError:
                continue
            if conf < min_confidence:
                continue
            if (row.get("snomed_active") or "").lower() != "true":
                continue
            mesh = (row.get("mesh_id") or "").strip()
            if not mesh:
                continue
            classes = {c.strip() for c in (row.get("entity_classes") or "").split(",") if c.strip()}
            if "chemical" in classes:
                chems.add(mesh)
            if "disease" in classes:
                diseases.add(mesh)
    return chems, diseases


# ---------------------------------------------------------------------------
# Pass 2: bulk filter
# ---------------------------------------------------------------------------

# Pre-intern target MeSH IDs so all list inserts of them point at the
# canonical interned string. Combined with sys.intern() on PMIDs in the
# streaming loop, this slashes per-entry memory in the per-PMID dicts.
def _intern_set(items: Set[str]) -> Set[str]:
    return {sys.intern(s) for s in items}


def _stream_concept_file(
    path: Path,
    target: Set[str],
    label: str,
    verbose: bool,
    pmid_filter: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[Dict[str, Tuple[str, ...]], int, int]:
    """Stream a per-bioconcept .gz file, building {pmid: tuple(in-target MeSH ids)}.

    Per-PMID concept storage is a tuple, not a set: tuples carry roughly
    1/5 the per-object overhead of a Python set (no hash table backing
    array), and the average MeSH-IDs-per-PMID payload is small (~4 ids)
    so a linear membership check during construction is cheaper than a
    set's hash overhead for these cardinalities. Net savings on
    chem_pmids alone are ~5-6 GB at peak vs. the prior set-of-strings
    storage.

    ``pmid_filter`` (optional) is a dict whose keys define the membership
    set. Rows whose PMID isn't in those keys are dropped on the spot; the
    disease pass uses ``chem_pmids`` directly here so we avoid allocating
    a separate snapshot set.

    PMID and MeSH-ID strings are passed through ``sys.intern`` so the same
    Python string object is shared across both passes' dicts, halving
    string-storage cost across pass 1 + pass 2.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; expected the bulk PubTator3 file under Data/PubTator3/."
        )
    # During streaming we accumulate into mutable lists for cheap append
    # + linear-dedupe. After streaming we convert every value to a tuple
    # in a single pass and let GC reclaim the list backing arrays.
    pmid_concepts_lists: Dict[str, List[str]] = defaultdict(list)
    n_total = 0
    n_kept = 0
    if verbose:
        gate = " (filtered to chemical-PMIDs)" if pmid_filter is not None else ""
        print(f"[2.6] streaming {path.name}{gate} ...", flush=True)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pmid_raw = parts[0]
            if pmid_filter is not None and pmid_raw not in pmid_filter:
                continue
            cid_raw = _strip_mesh(parts[2].strip())
            if cid_raw in target:
                pmid = sys.intern(pmid_raw)
                cid = sys.intern(cid_raw)
                bucket = pmid_concepts_lists[pmid]
                # Defensive dedupe -- the bulk concept files normally have
                # one row per (PMID, MeSH) tuple but this guards against
                # any duplicates and is O(k) where k is small (~4 avg).
                if cid not in bucket:
                    bucket.append(cid)
                    n_kept += 1
            if verbose and n_total % 5_000_000 == 0:
                print(f"[2.6]   {label}: {n_total:,} rows scanned, "
                      f"{n_kept:,} kept across {len(pmid_concepts_lists):,} PMIDs",
                      flush=True)
    if verbose:
        print(f"[2.6]   {label}: {n_total:,} total rows, {n_kept:,} in-footprint mentions, "
              f"{len(pmid_concepts_lists):,} distinct PMIDs", flush=True)

    # Freeze each list into a tuple in-place so the caller gets the
    # compact representation. The old lists become unreferenced and get
    # reclaimed by gc.collect() shortly afterward.
    pmid_concepts: Dict[str, Tuple[str, ...]] = pmid_concepts_lists  # type: ignore[assignment]
    for p in pmid_concepts_lists:
        pmid_concepts[p] = tuple(pmid_concepts_lists[p])  # type: ignore[index]
    return pmid_concepts, n_total, n_kept


def filter_pmids_in_domain(
    target_chems: Set[str],
    target_diseases: Set[str],
    verbose: bool = True,
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]], dict]:
    """Build the chemical-disease intersection of in-footprint PMIDs.

    Memory profile (16 GB-RAM safe, with ~3x improvement over the
    set-of-strings variant):
      * Pass 1 builds chem_pmids as Dict[str, Tuple[str, ...]] (~15M
        entries, ~1.5-2 GB after interning + tuple compaction).
      * Pass 2 streams disease.gz with chem_pmids itself as the filter (no
        separate set snapshot); dis_pmids is born intersection-sized
        (~10M entries, ~1-1.5 GB).
      * Post-pass-2 we rebuild chem_pmids restricted to the intersection
        (single dict-comp); the original 15M-entry dict is then deleted
        and gc.collect() reclaims its hash-table backing. Peak working
        set: ~3-4 GB instead of the prior ~8-9 GB.
    """
    target_chems_i = _intern_set(target_chems)
    target_diseases_i = _intern_set(target_diseases)

    chem_pmids, total_chem, kept_chem = _stream_concept_file(
        CHEM_GZ, target_chems_i, "chemical", verbose
    )
    chemical_pmid_count = len(chem_pmids)
    gc.collect()  # reclaim list backing arrays freed by the tuple-conversion

    # Pass chem_pmids directly as the filter dict; its keys() membership
    # is exactly what we need. No extra snapshot allocation.
    dis_pmids, total_disease, kept_disease = _stream_concept_file(
        DISEASE_GZ, target_diseases_i, "disease", verbose,
        pmid_filter=chem_pmids,
    )
    disease_pmid_count = len(dis_pmids)
    gc.collect()

    # The disease dict's keys ARE the intersection.
    intersection: Set[str] = set(dis_pmids.keys())
    if verbose:
        print(f"[2.6] PMIDs with >=1 in-footprint chemical AND >=1 in-footprint disease: "
              f"{len(intersection):,}", flush=True)

    # Rebuild the chemical dict trimmed to the intersection, then drop the
    # original. Peak briefly holds both, but only the intersection-sized
    # version survives.
    chem_pmids_inter: Dict[str, Tuple[str, ...]] = {p: chem_pmids[p] for p in intersection}
    del chem_pmids
    gc.collect()

    stats = {
        "chemical_rows_total": total_chem,
        "chemical_rows_in_footprint": kept_chem,
        "chemical_pmids_in_footprint": chemical_pmid_count,
        "disease_rows_total": total_disease,
        "disease_rows_in_footprint": kept_disease,
        "disease_pmids_in_footprint_post_filter": disease_pmid_count,
        "intersection_pmids": len(intersection),
    }
    return chem_pmids_inter, dis_pmids, stats


# ---------------------------------------------------------------------------
# Pass 3: stratified sampling
# ---------------------------------------------------------------------------

def sample_pmids_stratified(
    pmid_chems: Dict[str, Tuple[str, ...]],
    pmid_diseases: Dict[str, Tuple[str, ...]],
    n: int = SAMPLE_SIZE,
    coverage_target: int = COVERAGE_TARGET,
    verbose: bool = True,
) -> Tuple[List[str], dict]:
    """Greedy set-cover sampling biased toward rare-MeSH coverage.

    Phase A: walk PMIDs in richness-descending order, take any PMID that
    helps at least one currently-under-covered concept (count below
    coverage_target). This guarantees rare-concept representation.

    Phase B: fill remaining slots up to n with the next richest PMIDs,
    regardless of marginal coverage gain. Ensures we hit the volume target
    even if the coverage objective saturates early.

    Per-PMID concept storage is tuple-of-strings; the helper below builds
    a single small set per iteration for the union+coverage update.
    """

    def _ents_for(pmid: str) -> Set[str]:
        merged = set(pmid_chems.get(pmid, ()))
        merged.update(pmid_diseases.get(pmid, ()))
        return merged

    candidates = list(pmid_chems.keys())
    richness: Dict[str, int] = {
        p: len(pmid_chems.get(p, ())) + len(pmid_diseases.get(p, ()))
        for p in candidates
    }
    sorted_pmids = sorted(candidates, key=lambda p: -richness[p])

    coverage: Counter = Counter()
    chosen: List[str] = []
    chosen_set: Set[str] = set()

    # Phase A: stratified
    for pmid in sorted_pmids:
        if len(chosen) >= n:
            break
        ents = _ents_for(pmid)
        helps = sum(1 for c in ents if coverage[c] < coverage_target)
        if helps >= 1:
            chosen.append(pmid)
            chosen_set.add(pmid)
            coverage.update(ents)

    # Phase B: fill remaining slots by pure richness
    if len(chosen) < n:
        for pmid in sorted_pmids:
            if len(chosen) >= n:
                break
            if pmid in chosen_set:
                continue
            chosen.append(pmid)
            chosen_set.add(pmid)
            coverage.update(_ents_for(pmid))

    # Coverage diagnostics
    target_concepts: Set[str] = set()
    for p in candidates:
        target_concepts.update(pmid_chems.get(p, ()))
        target_concepts.update(pmid_diseases.get(p, ()))
    covered_at_target = sum(1 for c in target_concepts if coverage[c] >= coverage_target)
    covered_at_least_one = sum(1 for c in target_concepts if coverage[c] >= 1)

    stats = {
        "sample_size_target": n,
        "sample_size_actual": len(chosen),
        "coverage_target_per_concept": coverage_target,
        "in_footprint_concepts_in_pool": len(target_concepts),
        "concepts_covered_at_target": covered_at_target,
        "concepts_covered_at_least_one": covered_at_least_one,
        "concepts_uncovered": len(target_concepts) - covered_at_least_one,
    }
    if verbose:
        print(f"[2.6] sampled {len(chosen):,} PMIDs; "
              f"{covered_at_target:,}/{len(target_concepts):,} concepts hit "
              f">= {coverage_target} times, "
              f"{covered_at_least_one:,} hit >= 1 time", flush=True)
    return chosen, stats


# ---------------------------------------------------------------------------
# Pass 4: API fetch with disk cache
# ---------------------------------------------------------------------------

def fetch_biocxml(
    pmids: List[str],
    batch_size: int = PMIDS_PER_BATCH,
    sleep: float = API_RATE_SLEEP,
    verbose: bool = True,
) -> Tuple[List[Path], dict]:
    """Fetch BioC-XML batches; cache to CACHE_DIR. Idempotent on re-run."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pmids_sorted = sorted(set(pmids), key=int)
    n_batches = (len(pmids_sorted) + batch_size - 1) // batch_size
    cache_hits = 0
    fetched = 0
    failed_batches: List[int] = []
    paths: List[Path] = []

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        batch = pmids_sorted[start : start + batch_size]
        cache_path = CACHE_DIR / f"batch_{batch_idx + 1:04d}.xml"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            paths.append(cache_path)
            cache_hits += 1
            continue

        url = f"{API_URL}?pmids={','.join(batch)}"
        ok = False
        for attempt in range(API_MAX_RETRIES):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": API_USER_AGENT})
                with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
                    data = resp.read()
                cache_path.write_bytes(data)
                paths.append(cache_path)
                fetched += 1
                ok = True
                if verbose and ((batch_idx + 1) % 10 == 0 or batch_idx == 0):
                    print(f"[2.6] API batch {batch_idx + 1}/{n_batches} ({len(batch)} PMIDs) "
                          f"-> {len(data):,} bytes", flush=True)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt + 1 < API_MAX_RETRIES:
                    backoff = 2 ** attempt * 5
                    if verbose:
                        print(f"[2.6]   batch {batch_idx + 1}: HTTP {e.code}, "
                              f"retrying in {backoff}s ...", flush=True)
                    time.sleep(backoff)
                    continue
                if verbose:
                    print(f"[2.6]   batch {batch_idx + 1}: HTTP {e.code} -- skipping",
                          flush=True)
                break
            except (urllib.error.URLError, OSError) as e:
                if attempt + 1 < API_MAX_RETRIES:
                    time.sleep(2 ** attempt * 2)
                    continue
                if verbose:
                    print(f"[2.6]   batch {batch_idx + 1}: {type(e).__name__}: {e} -- skipping",
                          flush=True)
                break
        if not ok:
            failed_batches.append(batch_idx + 1)
        time.sleep(sleep)

    stats = {
        "batches_total": n_batches,
        "batches_cache_hits": cache_hits,
        "batches_fetched": fetched,
        "batches_failed": len(failed_batches),
        "failed_batch_indices": failed_batches[:20],  # cap for the JSON
    }
    if verbose:
        print(f"[2.6] API: {cache_hits:,} cache hits + {fetched:,} fetched "
              f"+ {len(failed_batches)} failed = {n_batches:,} batches", flush=True)
    return paths, stats


# ---------------------------------------------------------------------------
# Pass 5: parse + filter + convert
# ---------------------------------------------------------------------------

# PubTator3's BioCXML emits BioRED-vocabulary relation labels directly
# (Association, Negative_Correlation, Positive_Correlation, Bind, Cotreatment,
# Drug_Interaction, Conversion, Comparison). Reusing the BioRED key in
# relation_schema is intentional -- the labels are identical.
_PUBTATOR_RELATION_CORPUS = "BioRED"


def _parse_biocxml(xml_path: Path) -> Iterator[dict]:
    """Yield raw doc dicts from a PubTator3 BioCXML response file."""
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        print(f"[2.6]   WARNING: parse error in {xml_path.name}: {e}", flush=True)
        return
    root = tree.getroot()

    for doc_elem in root.findall("document"):
        pmid_elem = doc_elem.find("id")
        pmid = (pmid_elem.text or "").strip() if pmid_elem is not None else ""
        if not pmid:
            continue

        title = ""
        abstract_parts: List[str] = []
        # Project BioC absolute offsets onto our reconstructed text. Each
        # passage contributes (its bioc_offset, our_cursor) so we can map any
        # BioC offset >= bioc_offset_n into our text reliably.
        passage_offsets: List[Tuple[int, int]] = []
        cursor = 0

        entities: List[dict] = []

        for psg in doc_elem.findall("passage"):
            psg_type = ""
            for inf in psg.findall("infon"):
                if inf.get("key") == "type":
                    psg_type = (inf.text or "").lower().strip()
                    break
            offset_el = psg.find("offset")
            text_el = psg.find("text")
            psg_offset = int(offset_el.text) if (offset_el is not None and offset_el.text) else 0
            psg_text = (text_el.text if text_el is not None else "") or ""

            passage_offsets.append((psg_offset, cursor))
            if psg_type == "title" and not title:
                title = psg_text
            elif psg_text:
                abstract_parts.append(psg_text)
            cursor += len(psg_text) + 1  # +1 for the inserted separator

            for ann in psg.findall("annotation"):
                infons = {inf.get("key", ""): (inf.text or "") for inf in ann.findall("infon")}
                loc = ann.find("location")
                if loc is None:
                    continue
                start = int(loc.get("offset", "0"))
                length = int(loc.get("length", "0"))
                txt_elem = ann.find("text")
                surface = (txt_elem.text or "") if txt_elem is not None else ""

                # Project absolute BioC offset onto our reconstructed text.
                proj_start = start
                for po, co in passage_offsets:
                    if po <= start:
                        proj_start = co + (start - po)

                etype = (infons.get("type") or "").strip()
                ident_raw = (infons.get("identifier") or infons.get("MESH") or "").strip()

                entities.append({
                    "start": proj_start,
                    "end": proj_start + length,
                    "mention": surface,
                    "entity_type": etype,
                    "identifier_raw": ident_raw,
                })

        # Relations
        relations: List[dict] = []
        for rel in doc_elem.findall("relation"):
            r_infons = {inf.get("key", ""): (inf.text or "") for inf in rel.findall("infon")}
            rtype = (r_infons.get("type") or "").strip()
            score_str = (r_infons.get("score") or "").strip()
            try:
                score = float(score_str) if score_str else None
            except ValueError:
                score = None
            role1 = (r_infons.get("role1") or "").strip()
            role2 = (r_infons.get("role2") or "").strip()
            relations.append({
                "type": rtype,
                "score": score,
                "role1": role1,  # "Chemical|MESH:D001234"
                "role2": role2,
            })

        text = title if not abstract_parts else (title + " " + " ".join(abstract_parts))
        yield {
            "pmid": pmid,
            "title": title,
            "abstract": " ".join(abstract_parts),
            "text": text,
            "entities": entities,
            "relations": relations,
        }


def _role_to_code(role: str) -> str:
    """'Chemical|MESH:D001234' -> 'D001234'."""
    if not role:
        return ""
    parts = role.split("|", 1)
    raw = parts[1] if len(parts) == 2 else parts[0]
    return _strip_mesh(raw.strip())


def _entity_type_to_class(etype: str) -> Tuple[str, str]:
    """Map PubTator3 entity_type -> (canonical_corpus_label, semantic_class).

    PubTator3 uses 'Chemical' and 'Disease' identical to BC5CDR, but also
    'Gene', 'Species', 'CellLine', 'Mutation', etc. Map the SNOMED-scope
    types via entity_scope's BC5CDR/BioRED rows; everything else is
    non_snomed (NER-only, like BioRED genes/variants in Phase 1.3).
    """
    # Try BC5CDR labels first (Chemical, Disease).
    spec = entity_scope.lookup("BC5CDR", etype)
    if spec is not None:
        return etype, spec.semantic_class
    # PubTator3 also produces BioRED-style labels in some cases.
    spec = entity_scope.lookup("BioRED", etype)
    if spec is not None:
        return etype, spec.semantic_class
    # Map common PubTator3 labels into the BioRED registry by substring.
    table = {
        "Gene":     ("GeneOrGeneProduct",         "gene"),
        "Species":  ("OrganismTaxon",             "species"),
        "CellLine": ("CellLine",                  "cell_line"),
        "Mutation": ("SequenceVariant",           "variant"),
    }
    if etype in table:
        return table[etype]
    return etype, ""  # unknown types fall through; will be flagged in summary


def convert_to_unified(
    xml_paths: List[Path],
    mapping_table: Dict[str, concept_map._MappingEntry],
    sampled_pmids: Set[str],
    verbose: bool = True,
) -> Tuple[List[Document], dict]:
    """Parse cached XML, apply filters, return Documents + stats."""
    docs: List[Document] = []
    n_xml_docs = 0
    n_kept = 0
    n_dropped_no_subj_or_obj = 0
    n_dropped_no_relations = 0
    n_dropped_not_sampled = 0
    n_relations_seen = 0
    n_relations_kept_score = 0
    n_relations_kept_mapping = 0
    n_entities_seen = 0
    n_entities_mapped = 0
    n_entities_unmapped = 0
    unknown_types: Counter = Counter()
    relation_label_counter: Counter = Counter()

    for xml_path in xml_paths:
        for raw in _parse_biocxml(xml_path):
            n_xml_docs += 1
            pmid = raw["pmid"]
            if pmid not in sampled_pmids:
                # Defensive: PubTator3 may return PMIDs we didn't request.
                n_dropped_not_sampled += 1
                continue

            # ---- Build entities, applying mapping filter ----
            entities: List[EntityMention] = []
            code_to_first_idx: Dict[str, int] = {}
            for i, ent in enumerate(raw["entities"]):
                n_entities_seen += 1
                etype, sem_class = _entity_type_to_class(ent["entity_type"])
                if not sem_class and ent["entity_type"]:
                    unknown_types[ent["entity_type"]] += 1
                non_snomed = (sem_class in entity_scope.NON_SNOMED_NER_CLASSES)

                code_raw = ent["identifier_raw"] or None
                code_normalized = _strip_mesh(code_raw) if code_raw else None
                mapped_id: str | None = None
                snomed_active: bool | None = None
                mapping_conf: float | None = None

                if code_normalized and not non_snomed and sem_class in {"chemical", "disease"}:
                    entry = concept_map._best_entry_for_code(code_normalized, mapping_table)
                    if entry is not None and entry.confidence >= ENTITY_MAP_MIN_CONF and entry.active:
                        mapped_id = entry.snomed_id
                        snomed_active = True
                        # Silver weight overrides per-entry confidence (plan tier).
                        mapping_conf = SILVER_CONFIDENCE
                        n_entities_mapped += 1
                    else:
                        n_entities_unmapped += 1
                else:
                    if not non_snomed:
                        n_entities_unmapped += 1

                em = EntityMention(
                    id=f"T{i + 1}",
                    span_start=ent["start"],
                    span_end=ent["end"],
                    surface_text=ent["mention"],
                    entity_type=etype,
                    semantic_class=sem_class or None,
                    original_code=code_normalized,
                    mapped_snomed_id=mapped_id,
                    mapping_confidence=mapping_conf,
                    snomed_active=snomed_active,
                    non_snomed=non_snomed,
                    extra={"silver": True, "source": "PubTator3"},
                )
                entities.append(em)
                if code_normalized and code_normalized not in code_to_first_idx:
                    code_to_first_idx[code_normalized] = i

            # ---- Build relations, applying score filter ----
            relations: List[Relation] = []
            for rel in raw["relations"]:
                n_relations_seen += 1
                relation_label_counter[rel["type"]] += 1
                score = rel["score"]
                if score is None or score < REL_SCORE_THRESHOLD:
                    continue
                n_relations_kept_score += 1

                subj_code = _role_to_code(rel["role1"])
                obj_code = _role_to_code(rel["role2"])
                subj_idx = code_to_first_idx.get(subj_code)
                obj_idx = code_to_first_idx.get(obj_code)
                if subj_idx is None or obj_idx is None:
                    continue

                subj = entities[subj_idx]
                obj = entities[obj_idx]
                subj_class = subj.semantic_class or "unknown"
                obj_class = obj.semantic_class or "unknown"

                # Run through the same Phase 1.4 schema BioRED uses (PubTator3
                # emits identical relation labels). Default 'associated-with'
                # Tier-2 if the (subj, obj) pair is unmapped in the schema.
                mappings = relation_schema.get_mappings(
                    _PUBTATOR_RELATION_CORPUS, rel["type"], subj_class, obj_class
                )
                if mappings:
                    best = max(mappings, key=lambda m: m.probability)
                    target_relation = best.target_relation
                    tier = best.tier
                    target_prob = best.probability
                    candidates = [
                        {"target_relation": m.target_relation,
                         "tier": m.tier,
                         "probability": m.probability}
                        for m in mappings
                    ]
                    default_used = False
                else:
                    target_relation = "associated-with"
                    tier = 2
                    target_prob = 1.0
                    candidates = [{"target_relation": "associated-with",
                                   "tier": 2,
                                   "probability": 1.0}]
                    default_used = True

                relations.append(Relation(
                    subject_idx=subj_idx,
                    object_idx=obj_idx,
                    source_relation_type=rel["type"],
                    target_relation=target_relation,
                    tier=tier,
                    target_probability=target_prob,
                    novelty=None,
                    extra={
                        "silver": True,
                        "subject_class": subj_class,
                        "object_class": obj_class,
                        "pubtator_score": score,
                        "target_candidates": candidates,
                        "default_used": default_used,
                    },
                ))
                n_relations_kept_mapping += 1

            # ---- Document-level keep rule ----
            # Need at least one mapped chemical AND one mapped disease entity.
            mapped_chems    = any(e.mapped_snomed_id and e.semantic_class == "chemical" for e in entities)
            mapped_diseases = any(e.mapped_snomed_id and e.semantic_class == "disease"  for e in entities)
            if not (mapped_chems and mapped_diseases):
                n_dropped_no_subj_or_obj += 1
                continue
            if not relations:
                n_dropped_no_relations += 1
                continue

            docs.append(Document(
                pmid=pmid,
                corpus="PubTator3_silver",
                split="train",
                title=raw["title"],
                abstract=raw["abstract"],
                text=raw["text"],
                entities=entities,
                relations=relations,
            ))
            n_kept += 1

    stats = {
        "xml_docs_seen": n_xml_docs,
        "documents_kept": n_kept,
        "dropped_not_sampled": n_dropped_not_sampled,
        "dropped_no_mapped_chem_disease_pair": n_dropped_no_subj_or_obj,
        "dropped_no_relations": n_dropped_no_relations,
        "entities_seen": n_entities_seen,
        "entities_mapped": n_entities_mapped,
        "entities_unmapped_or_dropped": n_entities_unmapped,
        "relations_seen": n_relations_seen,
        "relations_kept_score_filter": n_relations_kept_score,
        "relations_kept_after_mapping": n_relations_kept_mapping,
        "relation_label_distribution": dict(relation_label_counter.most_common()),
        "unknown_entity_types": dict(unknown_types.most_common(20)),
    }
    if verbose:
        print(f"[2.6] parsed {n_xml_docs:,} XML docs -> kept {n_kept:,} silver documents",
              flush=True)
    return docs, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def apply_all(verbose: bool = True) -> dict:
    if not os.environ.get(ENV_GATE):
        msg = (
            f"Phase 2.6 is gated. Set environment variable {ENV_GATE}=1 to enable "
            "the PubTator3 silver pipeline (one network roundtrip required)."
        )
        if verbose:
            print(f"[2.6] SKIPPED -- {msg}", flush=True)
        return {"status": "skipped", "reason": msg}

    if verbose:
        print("[2.6] building target MeSH footprint from Phase 1.7 verified table ...",
              flush=True)
    target_chems, target_diseases = build_target_footprint(min_confidence=ENTITY_MAP_MIN_CONF)
    if verbose:
        print(f"[2.6] footprint: {len(target_chems):,} chemicals, "
              f"{len(target_diseases):,} diseases (>= {ENTITY_MAP_MIN_CONF} confidence)",
              flush=True)

    pmid_chems, pmid_diseases, filter_stats = filter_pmids_in_domain(
        target_chems, target_diseases, verbose=verbose
    )
    if not pmid_chems:
        raise RuntimeError(
            "No PMIDs survived the in-footprint filter; verify "
            "Data/PubTator3/*.gz files are populated."
        )

    sampled, sample_stats = sample_pmids_stratified(
        pmid_chems, pmid_diseases, n=SAMPLE_SIZE, coverage_target=COVERAGE_TARGET,
        verbose=verbose,
    )
    if not sampled:
        raise RuntimeError("Stratified sampler returned no PMIDs.")
    sampled_set = set(sampled)

    xml_paths, fetch_stats = fetch_biocxml(sampled, verbose=verbose)

    mapping_table = concept_map.load_verified_table()
    docs, parse_stats = convert_to_unified(
        xml_paths, mapping_table, sampled_set, verbose=verbose
    )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_written = write_jsonl(iter(docs), OUTPUT_JSONL)

    summary = {
        "status": "completed",
        "policy": {
            "entity_mapping_min_confidence": ENTITY_MAP_MIN_CONF,
            "relation_score_threshold": REL_SCORE_THRESHOLD,
            "silver_confidence": SILVER_CONFIDENCE,
            "sample_size_target": SAMPLE_SIZE,
            "coverage_target": COVERAGE_TARGET,
        },
        "footprint": {
            "target_chemicals": len(target_chems),
            "target_diseases": len(target_diseases),
        },
        "bulk_filter": filter_stats,
        "sampling": sample_stats,
        "api_fetch": fetch_stats,
        "parsing_and_filtering": parse_stats,
        "documents_written": n_written,
        "outputs": {
            "train_jsonl": relative_to_repo(OUTPUT_JSONL),
            "biocxml_cache_dir": relative_to_repo(CACHE_DIR),
        },
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[2.6] wrote {n_written:,} silver documents -> {relative_to_repo(OUTPUT_JSONL)}",
              flush=True)
        print(f"[2.6] summary -> {relative_to_repo(SUMMARY_JSON)}", flush=True)
    return summary


if __name__ == "__main__":
    apply_all(verbose=True)
