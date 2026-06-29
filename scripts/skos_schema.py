"""SKOS prefixes, URI minting, and method-to-property mapping for CANON.

Seed of the Phase 2.0 SKOS Foundation deliverable. Right now it carries only
what Phase 1.2 (MeSH -> SNOMED mapping) needs to emit SKOS-typed rows: a
small prefix table, URI mint helpers for the vocabularies used in Phase 1, the
SKOS mapping-property constants, and the mapping_method -> skos_property table.

This module will be extended in Phase 2.0 with the JSON-LD ``@context`` writer,
the NetworkX -> RDF exporter, and a thin SKOS reader/writer; producers landed
before 2.0 (Phase 1.2 today, Phase 1.4 next) import only the URI helpers and
the property table so the schema delta stays in one place.

Prefixes follow standard published URIs where one exists:

    mesh:           http://id.nlm.nih.gov/mesh/{descriptor_id}      (NLM)
    snomed:         http://snomed.info/id/{sctid}                   (IHTSDO)
    skos:           http://www.w3.org/2004/02/skos/core#            (W3C)

The remaining prefixes (canon:, biored:, bc5cdr:, ncbi_gene:, dbsnp:,
ncbi_taxon:, cellosaurus:) are CANON-local. They are listed here so the
Phase 2.0 ``@context`` file can be generated mechanically from this table.
"""

from __future__ import annotations

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Namespace prefixes
# ---------------------------------------------------------------------------
PREFIXES: Dict[str, str] = {
    "canon":       "https://canon.iu.edu/ns#",
    "skos":        "http://www.w3.org/2004/02/skos/core#",
    "prov":        "http://www.w3.org/ns/prov#",
    "mesh":        "http://id.nlm.nih.gov/mesh/",
    "snomed":      "http://snomed.info/id/",
    "biored":      "https://canon.iu.edu/corpus/biored#",
    "bc5cdr":      "https://canon.iu.edu/corpus/bc5cdr#",
    "ncbi_gene":   "https://www.ncbi.nlm.nih.gov/gene/",
    "dbsnp":       "https://www.ncbi.nlm.nih.gov/snp/",
    "ncbi_taxon":  "https://www.ncbi.nlm.nih.gov/taxonomy/",
    "cellosaurus": "https://www.cellosaurus.org/",
}


# ---------------------------------------------------------------------------
# SKOS mapping-property constants
# ---------------------------------------------------------------------------
SKOS_EXACT_MATCH   = "skos:exactMatch"
SKOS_CLOSE_MATCH   = "skos:closeMatch"
SKOS_BROAD_MATCH   = "skos:broadMatch"
SKOS_NARROW_MATCH  = "skos:narrowMatch"
SKOS_RELATED_MATCH = "skos:relatedMatch"


# Phase 1.2 method name -> SKOS mapping property.
# The Phase 1.2 four-priority pipeline produces these methods:
#   mrmap_curated     -- NLM-curated 1:1 mapping, interchangeable
#   shared_cui        -- same UMLS atom-level concept; close but not identical
#   mrrel_sy          -- UMLS SY: synonymous
#   mrrel_rq          -- UMLS RQ: related and possibly synonymous
#   mrrel_rb          -- UMLS RB: source is narrower than target (broader-than)
#   mrrel_rn          -- UMLS RN: source is broader than target (narrower-than)
#   sty_fallback:*    -- semantic-type root fallback (last resort)
#
# Per CANON_Plan.txt 1.2 the SKOS property is one of
# {exactMatch | closeMatch | broadMatch | narrowMatch | relatedMatch}.
# RB/RN map directly to broad/narrow per the plan; SY/RQ are tighter than
# relatedMatch but looser than exactMatch, so closeMatch fits SKOS semantics
# (interchangeable in some applications, not in general).
METHOD_TO_SKOS_PROPERTY: Dict[str, str] = {
    "mrmap_curated": SKOS_EXACT_MATCH,
    "shared_cui":    SKOS_CLOSE_MATCH,
    "mrrel_sy":      SKOS_CLOSE_MATCH,
    "mrrel_rq":      SKOS_CLOSE_MATCH,
    "mrrel_rb":      SKOS_BROAD_MATCH,
    "mrrel_rn":      SKOS_NARROW_MATCH,
}


def skos_property_for_method(method: str) -> str:
    """Resolve a Phase 1.2 mapping_method to its SKOS property.

    Semantic-type fallback methods are recorded as ``sty_fallback:<sty>``;
    they all collapse to skos:relatedMatch regardless of the trailing tag.
    """
    if method.startswith("sty_fallback"):
        return SKOS_RELATED_MATCH
    try:
        return METHOD_TO_SKOS_PROPERTY[method]
    except KeyError as exc:
        raise ValueError(f"Unknown Phase 1.2 mapping_method: {method!r}") from exc


# ---------------------------------------------------------------------------
# URI mint helpers
# ---------------------------------------------------------------------------
def mint_mesh_uri(mesh_id: Optional[str]) -> str:
    """Return the NLM MeSH URI for *mesh_id*; empty string for falsy input."""
    if not mesh_id:
        return ""
    return f"{PREFIXES['mesh']}{mesh_id}"


def mint_snomed_uri(sctid: Optional[str]) -> str:
    """Return the SNOMED CT URI for *sctid*; empty string for falsy input."""
    if not sctid:
        return ""
    return f"{PREFIXES['snomed']}{sctid}"
