"""Per-corpus entity-type scoping registry (CANON Phase 1.3).

Single source of truth for which corpus entity types are normalized to SNOMED
versus retained as NER+RE-only. Phase 2.2 (concept mapping application),
Phase 3.2 (concept-normalization head), and Phase 3.5 (CSP solver) all
consult this registry to decide whether an entity participates in SNOMED-
backed reasoning.

Scoping decision (per CANON_Plan.txt 1.3):

    * MeSH-coded entities (chemicals, diseases) -> normalized to SNOMED.
    * Genes (NCBI Gene) and variants (dbSNP)    -> excluded from SNOMED
      normalization. SNOMED has limited genomic content; the CSP solver's
      constraints apply only to entities with SNOMED mappings.
    * Species (NCBI Taxonomy) and cell lines (Cellosaurus) -> excluded
      from SNOMED normalization to keep the CSP solver's domain coherent
      with the gene/variant decision; they remain in NER and RE.

All entities, in-scope or not, retain their original codes and continue to
participate in NER training and relation extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class EntityTypeSpec:
    corpus: str
    entity_type: str          # exact label as written in the source corpus
    vocabulary: str           # native normalization vocabulary
    semantic_class: str       # canonical class: chemical, disease, gene, variant, species, cell_line
    snomed_normalized: bool   # True iff Phase 2.2 maps the code to SNOMED
    rationale: str


_REASON_MESH = "MeSH descriptor; mapped to SNOMED via Phase 1.2 pipeline."
_REASON_GENOMIC = "SNOMED has limited genomic content; excluded from concept normalization (Phase 1.3 decision). NER + RE only."
_REASON_OTHER_NON_MESH = "Non-MeSH vocabulary held out of SNOMED normalization to keep the CSP solver's domain consistent with the gene/variant decision. NER + RE only."


ENTITY_TYPES: List[EntityTypeSpec] = [
    # ------------------------------------------------------------------ BioRED
    EntityTypeSpec("BioRED", "ChemicalEntity",             "MeSH",          "chemical",  True,  _REASON_MESH),
    EntityTypeSpec("BioRED", "DiseaseOrPhenotypicFeature", "MeSH",          "disease",   True,  _REASON_MESH),
    EntityTypeSpec("BioRED", "GeneOrGeneProduct",          "NCBI Gene",     "gene",      False, _REASON_GENOMIC),
    EntityTypeSpec("BioRED", "SequenceVariant",            "dbSNP",         "variant",   False, _REASON_GENOMIC),
    EntityTypeSpec("BioRED", "OrganismTaxon",              "NCBI Taxonomy", "species",   False, _REASON_OTHER_NON_MESH),
    EntityTypeSpec("BioRED", "CellLine",                   "Cellosaurus",   "cell_line", False, _REASON_OTHER_NON_MESH),

    # ------------------------------------------------------------------ BC5CDR
    EntityTypeSpec("BC5CDR", "Chemical", "MeSH", "chemical", True, _REASON_MESH),
    EntityTypeSpec("BC5CDR", "Disease",  "MeSH", "disease",  True, _REASON_MESH),

    # ------------------------------------------------------------- NCBI Disease
    EntityTypeSpec("NCBI_Disease", "DiseaseClass",     "MeSH", "disease", True, _REASON_MESH),
    EntityTypeSpec("NCBI_Disease", "SpecificDisease",  "MeSH", "disease", True, _REASON_MESH),
    EntityTypeSpec("NCBI_Disease", "Modifier",         "MeSH", "disease", True, _REASON_MESH),
    EntityTypeSpec("NCBI_Disease", "CompositeMention", "MeSH", "disease", True, _REASON_MESH),

    # ----------------------------------------------------------------- NLM-Chem
    EntityTypeSpec("NLM-Chem", "Chemical", "MeSH", "chemical", True, _REASON_MESH),
]


# Top-level NER tag classes the model emits for non-SNOMED-normalized types.
# The NER head still produces BIO tags for these; only the concept-normalization
# head and the CSP solver skip them.
NON_SNOMED_NER_CLASSES = ("gene", "variant", "species", "cell_line")
SNOMED_NER_CLASSES = ("chemical", "disease")


def iter_specs() -> Iterator[EntityTypeSpec]:
    return iter(ENTITY_TYPES)


def specs_for_corpus(corpus: str) -> List[EntityTypeSpec]:
    return [s for s in ENTITY_TYPES if s.corpus == corpus]


def lookup(corpus: str, entity_type: str) -> Optional[EntityTypeSpec]:
    for s in ENTITY_TYPES:
        if s.corpus == corpus and s.entity_type == entity_type:
            return s
    return None


def is_snomed_normalized(corpus: str, entity_type: str) -> bool:
    spec = lookup(corpus, entity_type)
    return bool(spec and spec.snomed_normalized)


def snomed_normalized_type_map(corpus: str) -> dict:
    """{entity_type: semantic_class} for SNOMED-normalized types in a corpus."""
    return {
        s.entity_type: s.semantic_class
        for s in specs_for_corpus(corpus)
        if s.snomed_normalized
    }


def out_of_scope_type_map(corpus: str) -> dict:
    """{entity_type: semantic_class} for non-SNOMED-normalized types in a corpus."""
    return {
        s.entity_type: s.semantic_class
        for s in specs_for_corpus(corpus)
        if not s.snomed_normalized
    }


if __name__ == "__main__":
    width = max(len(s.entity_type) for s in ENTITY_TYPES)
    for s in ENTITY_TYPES:
        flag = "SNOMED" if s.snomed_normalized else "  NER "
        print(f"{s.corpus:<14s} {s.entity_type:<{width}s}  {flag}  {s.vocabulary:<14s}  {s.semantic_class}")
