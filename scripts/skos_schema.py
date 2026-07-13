"""SKOS prefixes, URI minting, and method-to-property mapping for CANON.

The Phase 2.0 SKOS Foundation. It carries: the prefix table; URI mint helpers
for the vocabularies used in Phases 1 and 2; the SKOS mapping-property
constants; the mapping_method -> skos_property table; the JSON-LD ``@context``
writer (Deliverable 1); a thin SKOS reader/writer plus CURIE expand/compact
helpers; and the NetworkX -> RDF exporter that streams ``skos:broader`` triples
from the Phase 1.6 SNOMED hierarchy on demand (Deliverable 2).

The in-memory pipeline stays on compact identifiers (integer/string SCTIDs,
MeSH descriptor IDs, bare relation labels) for performance; URI expansion
happens only at write/read boundaries through the mint / ``expand_curie`` /
``write_jsonld`` helpers here, so the schema delta stays in one place.

Prefixes follow standard published URIs where one exists:

    mesh:           http://id.nlm.nih.gov/mesh/{descriptor_id}      (NLM)
    snomed:         http://snomed.info/id/{sctid}                   (IHTSDO)
    skos:           http://www.w3.org/2004/02/skos/core#            (W3C)

The remaining prefixes (canon:, biored:, bc5cdr:, ncbi_gene:, dbsnp:,
ncbi_taxon:, cellosaurus:) are CANON-local. They are listed here so the
Phase 2.0 ``@context`` file can be generated mechanically from this table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


# ---------------------------------------------------------------------------
# Namespace prefixes
# ---------------------------------------------------------------------------
CANON_BASE_URI = "https://w3id.org/canon/"
CANON_VOCAB_URI = f"{CANON_BASE_URI}ns#"


PREFIXES: Dict[str, str] = {
    "canon":       CANON_VOCAB_URI,
    "skos":        "http://www.w3.org/2004/02/skos/core#",
    "prov":        "http://www.w3.org/ns/prov#",
    "mesh":        "http://id.nlm.nih.gov/mesh/",
    "snomed":      "http://snomed.info/id/",
    "biored":      f"{CANON_BASE_URI}corpus/biored#",
    "bc5cdr":      f"{CANON_BASE_URI}corpus/bc5cdr#",
    "ncbi_gene":   "https://www.ncbi.nlm.nih.gov/gene/",
    "dbsnp":       "https://www.ncbi.nlm.nih.gov/snp/",
    "ncbi_taxon":  "https://www.ncbi.nlm.nih.gov/taxonomy/",
    "cellosaurus": "https://www.cellosaurus.org/",
}


# ---------------------------------------------------------------------------
# SKOS mapping-property constants
# ---------------------------------------------------------------------------
# CANON deliberately omits skos:exactMatch from its property vocabulary. SKOS
# exactMatch carries commitments (transitivity, application-independent
# interchangeability) that none of the available CANON evidence sources can
# verify: UMLS CUI grouping is a curator-assisted synonymy claim, not a
# cross-application equivalence guarantee; MeSH descriptors are bibliographic
# indexing terms whose scope is documentary; SNOMED concepts are clinical
# recording terms whose scope is operational. A shared name across the two
# vocabularies ("Sodium", "Serotonin") can refer to the substance in MeSH and
# to one of several distinguishable clinical senses in SNOMED (the substance,
# the measurand, the ion). exactMatch would overclaim. closeMatch is the
# correct ceiling under the SKOS spec for everything we can produce.
SKOS_CLOSE_MATCH   = "skos:closeMatch"
SKOS_BROAD_MATCH   = "skos:broadMatch"
SKOS_NARROW_MATCH  = "skos:narrowMatch"
SKOS_RELATED_MATCH = "skos:relatedMatch"


# Phase 1.2 method name -> SKOS mapping property.
# The Phase 1.2 three-priority pipeline produces these methods:
#   shared_cui_strict -- same UMLS concept, single SNOMED concept and single
#                        MeSH MH per CUI, both atoms preferred terms, and
#                        normalized strings match. High-confidence closeMatch
#                        (conf 0.93). Still closeMatch -- the four-condition
#                        test is a strong heuristic for substitutability in
#                        many contexts but does not warrant exactMatch.
#   shared_cui        -- same UMLS atom-level concept; close but not identical
#   mrrel_sy          -- UMLS SY: synonymous
#   mrrel_rq          -- UMLS RQ: related and possibly synonymous
#   mrrel_rb          -- UMLS RB: source is narrower than target (broader-than)
#   mrrel_rn          -- UMLS RN: source is broader than target (narrower-than)
#   sty_fallback:*    -- semantic-type root fallback (last resort)
#
# Per plan.md 1.2 the SKOS property is one of
# {closeMatch | broadMatch | narrowMatch | relatedMatch}.
# RB/RN map directly to broad/narrow per the plan; SY/RQ are tighter than
# relatedMatch but looser than exactMatch, so closeMatch fits SKOS semantics
# (interchangeable in some applications, not in general).
METHOD_TO_SKOS_PROPERTY: Dict[str, str] = {
    "shared_cui_strict": SKOS_CLOSE_MATCH,
    "shared_cui":        SKOS_CLOSE_MATCH,
    "mrrel_sy":          SKOS_CLOSE_MATCH,
    "mrrel_rq":          SKOS_CLOSE_MATCH,
    "mrrel_rb":          SKOS_BROAD_MATCH,
    "mrrel_rn":          SKOS_NARROW_MATCH,
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
# Relation-scheme URIs (Phase 1.4)
# ---------------------------------------------------------------------------
# The unified relation vocabulary is a SKOS concept scheme, canon:RelationScheme.
# Each unified relation (causative-agent, treats, ...) is a skos:Concept in it
# carrying a canon:tier. Each source corpus's relation vocabulary is its own
# scheme (biored:RelationScheme, bc5cdr:RelationScheme) whose concepts link into
# canon:RelationScheme via skos:closeMatch with a canon:probability annotation.
# The compact canon relation URI (canon:causative-agent) matches the notation
# the plan uses for canon:relation values in the Phase 2.1 annotation format.
CANON_RELATION_SCHEME_URI = f"{PREFIXES['canon']}RelationScheme"

_CORPUS_SCHEME_PREFIX: Dict[str, str] = {
    "BioRED": "biored",
    "BC5CDR": "bc5cdr",
}


def source_scheme_prefix(source_corpus: str) -> str:
    """Return the registered SKOS prefix key for a source corpus."""
    try:
        return _CORPUS_SCHEME_PREFIX[source_corpus]
    except KeyError as exc:
        raise ValueError(
            f"No SKOS relation-scheme prefix registered for corpus {source_corpus!r}"
        ) from exc


def mint_source_relation_scheme_uri(source_corpus: str) -> str:
    """Return the per-corpus relation ConceptScheme URI (e.g. biored:RelationScheme)."""
    return f"{PREFIXES[source_scheme_prefix(source_corpus)]}RelationScheme"


def mint_source_relation_uri(source_corpus: str, source_relation_type: str) -> str:
    """Return the skos:Concept URI for a source-corpus relation label."""
    return f"{PREFIXES[source_scheme_prefix(source_corpus)]}{source_relation_type}"


def mint_canon_relation_uri(target_relation: str) -> str:
    """Return the skos:Concept URI for a unified (canon) relation label."""
    return f"{PREFIXES['canon']}{target_relation}"


def mint_document_uri(corpus: str, document_id: str) -> str:
    """Return the stable project IRI for a corpus document.

    The W3ID redirect is a publication concern, not a runtime dependency.
    CANON JSON-LD embeds its context and therefore remains usable offline.
    """
    return f"{CANON_BASE_URI}document/{corpus}/{document_id}"


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


# ---------------------------------------------------------------------------
# JSON-LD @context (Phase 2.0 Deliverable 1)
# ---------------------------------------------------------------------------
# The published @context is generated mechanically from PREFIXES plus the term
# coercions below, so the prefix table and the context cannot drift. Only terms
# whose default (plain string literal) behaviour is wrong need an entry here:
#   * properties whose value is an IRI  -> {"@type": "@id"}
#   * numeric annotations               -> {"@type": "xsd:<type>"}
# Plain-literal SKOS labels (skos:prefLabel / altLabel / hiddenLabel /
# notation) need no coercion: the skos: prefix binding already expands them and
# a bare JSON string is the correct value form.

XSD = "http://www.w3.org/2001/XMLSchema#"

TERM_DEFINITIONS: Dict[str, Dict[str, str]] = {
    # SKOS hierarchical, associative, and mapping properties all point at IRIs.
    "skos:broader":      {"@type": "@id"},
    "skos:narrower":     {"@type": "@id"},
    "skos:related":      {"@type": "@id"},
    "skos:closeMatch":   {"@type": "@id"},
    "skos:broadMatch":   {"@type": "@id"},
    "skos:narrowMatch":  {"@type": "@id"},
    "skos:relatedMatch": {"@type": "@id"},
    "skos:inScheme":     {"@type": "@id"},
    # canon annotation properties whose value is an IRI (Phase 2.1 / 2.2 / 2.3).
    "canon:normalizedConcept": {"@type": "@id"},
    "canon:sourceConcept":     {"@type": "@id"},
    "canon:relation":          {"@type": "@id"},
    "canon:subject":           {"@type": "@id"},
    "canon:object":            {"@type": "@id"},
    "canon:replacedBy":        {"@type": "@id"},
    # canon numeric annotations (Phase 1.2 / 1.4 / 2.x).
    "canon:tier":        {"@type": "xsd:integer"},
    "canon:probability": {"@type": "xsd:double"},
    "canon:confidence":  {"@type": "xsd:double"},
    # canon plain-literal annotations, listed with explicit type for clarity.
    "canon:sourceCorpus":     {"@type": "xsd:string"},
    "canon:split":            {"@type": "xsd:string"},
    "canon:mappingProperty":  {"@type": "xsd:string"},
    "canon:sourceNotation":   {"@type": "xsd:string"},
    "canon:schemaVersion":    {"@type": "xsd:string"},
}

#: Default on-disk location of the generated @context (scripts/contexts/).
CONTEXTS_DIR = Path(__file__).resolve().parent / "contexts"
CANON_CONTEXT_PATH = CONTEXTS_DIR / "canon.jsonld"

#: Relative reference embedded by write_jsonld so a payload sitting under
#: scripts/ resolves the context; overridable per call.
DEFAULT_CONTEXT_REF = "contexts/canon.jsonld"


def build_context() -> Dict[str, Any]:
    """Return the JSON-LD document ``{"@context": {...}}`` for CANON.

    Prefixes come straight from PREFIXES (plus xsd); term coercions from
    TERM_DEFINITIONS. Prefixes are emitted first so the ``xsd:``-typed term
    definitions resolve.
    """
    context: Dict[str, Any] = dict(PREFIXES)
    context["xsd"] = XSD
    context.update(TERM_DEFINITIONS)
    return {"@context": context}


def write_context(path: Optional[Path] = None) -> Path:
    """Write the generated @context to *path* (default: scripts/contexts/canon.jsonld)."""
    if path is None:
        path = CANON_CONTEXT_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(build_context(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# CURIE <-> IRI boundary helpers
# ---------------------------------------------------------------------------
# Longest-prefix map, built once. PREFIXES plus xsd covers every namespace the
# context declares. canon: and biored:/bc5cdr: share the W3ID CANON base
# stem but have distinct full namespaces, so compaction must prefer the longest
# matching namespace.
_ALL_NAMESPACES: Dict[str, str] = {**PREFIXES, "xsd": XSD}


def expand_curie(curie: str) -> str:
    """Expand a ``prefix:local`` CURIE to a full IRI using the declared namespaces.

    A value that is already an absolute IRI (starts with ``http://`` or
    ``https://``) is returned unchanged.
    """
    if curie.startswith(("http://", "https://")):
        return curie
    if ":" not in curie:
        raise ValueError(f"Not a CURIE (no prefix separator): {curie!r}")
    prefix, local = curie.split(":", 1)
    try:
        namespace = _ALL_NAMESPACES[prefix]
    except KeyError as exc:
        raise ValueError(f"Unknown prefix in CURIE {curie!r}: {prefix!r}") from exc
    return f"{namespace}{local}"


def compact_iri(iri: str) -> str:
    """Compact a full IRI to ``prefix:local`` using the longest matching namespace.

    An IRI under no declared namespace is returned unchanged.
    """
    best_prefix: Optional[str] = None
    best_namespace = ""
    for prefix, namespace in _ALL_NAMESPACES.items():
        if iri.startswith(namespace) and len(namespace) > len(best_namespace):
            best_prefix, best_namespace = prefix, namespace
    if best_prefix is None:
        return iri
    return f"{best_prefix}:{iri[len(best_namespace):]}"


# ---------------------------------------------------------------------------
# Thin SKOS JSON-LD reader / writer
# ---------------------------------------------------------------------------
# Deliberately minimal: attach or strip the relative @context reference so an
# already-SKOS-shaped payload (e.g. outputs/phase1/relation_scheme_skos.json)
# becomes a valid linked-data document, and read it back as a plain dict. No
# graph materialization or JSON-LD expansion happens here -- that is what the
# @context + a real processor (pyoxigraph) are for at the RDF boundary.


def write_jsonld(
    obj: Dict[str, Any],
    path: Path,
    context_ref: str = DEFAULT_CONTEXT_REF,
) -> Path:
    """Write *obj* to *path* with *context_ref* attached as its ``@context``.

    Any pre-existing ``@context`` on *obj* is replaced by *context_ref*.
    """
    path = Path(path)
    payload: Dict[str, Any] = {"@context": context_ref}
    payload.update({k: v for k, v in obj.items() if k != "@context"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def read_jsonld(path: Path) -> Dict[str, Any]:
    """Read a JSON-LD document from *path*, returning the payload without ``@context``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if isinstance(doc, dict):
        doc.pop("@context", None)
    return doc


# ---------------------------------------------------------------------------
# NetworkX -> RDF exporter (Phase 2.0 Deliverable 2)
# ---------------------------------------------------------------------------
# Walks the Phase 1.6 SNOMED hierarchy DiGraph and streams SKOS triples. The
# graph stores is-a as child -> parent edges (snomed_hierarchy.build_graph), so
# a directed edge (child, parent) is exactly `child skos:broader parent` -- no
# inversion. pyoxigraph is imported lazily so the many low-level producers that
# import this module for the mint helpers do not pull in the RDF dependency.

RDF_TYPE_IRI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SKOS_CONCEPT_IRI = f"{PREFIXES['skos']}Concept"
SKOS_BROADER_IRI = f"{PREFIXES['skos']}broader"
SKOS_NOTATION_IRI = f"{PREFIXES['skos']}notation"


def iter_hierarchy_triples(graph: Any = None) -> Iterator[Any]:
    """Yield pyoxigraph ``Triple`` objects for the SNOMED hierarchy.

    Emits, per concept node: ``rdf:type skos:Concept`` and ``skos:notation``
    (the SCTID); per is-a edge: ``skos:broader``. If *graph* is None the Phase
    1.6 graph is loaded via ``snomed_hierarchy.load_or_build``.
    """
    import pyoxigraph as ox  # lazy: keeps the base module RDF-dependency-free

    if graph is None:
        try:
            import snomed_hierarchy
        except ImportError:  # pragma: no cover - support package-relative import
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import snomed_hierarchy
        graph, _ = snomed_hierarchy.load_or_build(verbose=False)

    named = ox.NamedNode
    literal = ox.Literal
    triple = ox.Triple
    type_pred = named(RDF_TYPE_IRI)
    concept_obj = named(SKOS_CONCEPT_IRI)
    broader_pred = named(SKOS_BROADER_IRI)
    notation_pred = named(SKOS_NOTATION_IRI)

    for node in graph.nodes:
        subject = named(mint_snomed_uri(str(node)))
        yield triple(subject, type_pred, concept_obj)
        yield triple(subject, notation_pred, literal(str(node)))

    # DiGraph.edges yields (source, target) == (child, parent).
    for child, parent in graph.edges:
        yield triple(
            named(mint_snomed_uri(str(child))),
            broader_pred,
            named(mint_snomed_uri(str(parent))),
        )


def export_hierarchy_rdf(
    out_path: Optional[Path] = None,
    fmt: Any = None,
    graph: Any = None,
) -> Path:
    """Stream the SNOMED hierarchy to an RDF file and return its path.

    *fmt* is a ``pyoxigraph.RdfFormat`` (default N-Triples) or a media-type /
    file-extension string it can resolve. Default output is
    ``outputs/phase2/rdf/snomed_hierarchy.nt``.
    """
    import pyoxigraph as ox

    if fmt is None:
        fmt = ox.RdfFormat.N_TRIPLES
    elif isinstance(fmt, str):
        fmt = ox.RdfFormat.from_media_type(fmt) or ox.RdfFormat.from_extension(fmt)
        if fmt is None:
            raise ValueError("Unrecognized RDF format string")

    if out_path is None:
        try:
            import config
        except ImportError:  # pragma: no cover
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import config
        out_path = config.PHASE2_RDF_DIR / "snomed_hierarchy.nt"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as fh:
        ox.serialize(iter_hierarchy_triples(graph), output=fh, format=fmt)
    return out_path


# ---------------------------------------------------------------------------
# Self-check / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CANON Phase 2.0 SKOS foundation")
    parser.add_argument(
        "--export-rdf",
        action="store_true",
        help="also stream the Phase 1.6 hierarchy to outputs/phase2/rdf/snomed_hierarchy.nt",
    )
    args = parser.parse_args()

    ctx_path = write_context()
    context = build_context()["@context"]
    assert all(context.get(p) == ns for p, ns in PREFIXES.items()), \
        "context prefixes drifted from PREFIXES"
    for term in TERM_DEFINITIONS:
        prefix = term.split(":", 1)[0]
        assert prefix in _ALL_NAMESPACES, f"coerced term {term!r} has unknown prefix"
    print(f"context  -> {ctx_path}  "
          f"({len(PREFIXES)} prefixes + xsd, {len(TERM_DEFINITIONS)} coerced terms)")

    assert expand_curie("snomed:73211009") == "http://snomed.info/id/73211009"
    assert compact_iri("http://snomed.info/id/73211009") == "snomed:73211009"
    assert expand_curie(compact_iri(f"{PREFIXES['biored']}Association")) == \
        f"{PREFIXES['biored']}Association"
    print("curie roundtrip OK")

    if args.export_rdf:
        rdf_path = export_hierarchy_rdf()
        print(f"rdf      -> {rdf_path}")
