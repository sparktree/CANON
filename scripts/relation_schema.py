"""Relation schema alignment table for CANON Phase 1.4.

Maps each source-corpus relation type to the unified two-tier schema consumed
by Phase 2.3 (Apply Relation Mappings) and the CSP solver (Phase 3.5).

Tier 1 — SNOMED-native attribute relations the CSP solver can enforce via MRCM
          constraints (finding-site, causative-agent, etc.).
Tier 2 — Empirical / clinical relations that are valid annotation outputs but
          are not part of SNOMED's formal attribute model (treats, causes, etc.).

Mappings are many-to-many and entity-pair-dependent.  Where one source label
maps to multiple targets, probabilities sum to 1.0 within each group keyed by
(source_corpus, source_relation_type, subject_semantic_class,
 object_semantic_class).

Subject/object semantic_class values must match those defined in entity_scope.py:
  chemical, disease, gene, variant, species, cell_line
"""

from __future__ import annotations

import csv
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Iterator, List, Optional

# ---------------------------------------------------------------------------
# Unified relation label inventories
# ---------------------------------------------------------------------------

#: SNOMED CT attribute relation types the CSP solver can enforce via MRCM.
TIER1_RELATIONS: frozenset[str] = frozenset(
    {
        "causative-agent",        # SNOMED 246075003
        "finding-site",           # SNOMED 363698007
        "associated-morphology",  # SNOMED 116676008
        "due-to",                 # SNOMED 42752001
        "after",                  # SNOMED 255234002
    }
)

#: Empirical / clinical relations outside SNOMED's formal attribute model.
TIER2_RELATIONS: frozenset[str] = frozenset(
    {
        "treats",
        "causes",
        "associated-with",
        "interacts-with",
        "co-treats",
        "converts-to",
        "compared-with",
    }
)

ALL_TARGET_RELATIONS: frozenset[str] = TIER1_RELATIONS | TIER2_RELATIONS

# ---------------------------------------------------------------------------
# Mapping dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationMapping:
    source_corpus: str           # 'BioRED' | 'BC5CDR'
    source_relation_type: str    # exact label from PubTator (e.g. 'Negative_Correlation')
    subject_semantic_class: str  # entity_scope semantic_class of the subject
    object_semantic_class: str   # entity_scope semantic_class of the object
    target_relation: str         # unified relation label
    tier: int                    # 1 = SNOMED-native, 2 = empirical
    probability: float           # within-group probability; groups sum to 1.0
    notes: str                   # brief rationale for this row


# ---------------------------------------------------------------------------
# Mapping table
# Organised by source corpus and relation type.
# Probabilities within each (corpus, relation, subject, object) group sum to 1.0.
# ---------------------------------------------------------------------------

_N = ""  # empty notes placeholder used for clear-cut rows


RELATION_MAPPINGS: List[RelationMapping] = [

    # =========================================================================
    # BC5CDR — Chemical-Induced Disease (CID)
    # CID asserts that a chemical causes or is associated with a disease.
    # The annotation guidelines require a documented causal or correlative link;
    # most instances are side effects or adverse drug reactions (causes), with a
    # smaller fraction being general associations.
    # =========================================================================
    RelationMapping("BC5CDR", "CID", "chemical", "disease",
                    "causes",          2, 0.60,
                    "CID most commonly denotes a chemical causing a disease "
                    "(adverse effect / side effect)."),
    RelationMapping("BC5CDR", "CID", "chemical", "disease",
                    "causative-agent", 1, 0.30,
                    "When the causal mechanism is explicit the SNOMED attribute "
                    "causative-agent applies and the CSP solver can verify it."),
    RelationMapping("BC5CDR", "CID", "chemical", "disease",
                    "associated-with", 2, 0.10,
                    "A minority of CID instances state correlation only "
                    "without a documented causal mechanism."),

    # =========================================================================
    # BioRED — Association
    # Non-directional, non-specific relation.  BioRED guidelines use Association
    # as a catch-all when no more specific relation type applies.
    # =========================================================================
    RelationMapping("BioRED", "Association", "chemical", "disease",
                    "associated-with", 2, 0.80, _N),
    RelationMapping("BioRED", "Association", "chemical", "disease",
                    "causative-agent", 1, 0.20,
                    "Subset where context implies a causal mechanism."),

    RelationMapping("BioRED", "Association", "chemical", "gene",
                    "associated-with", 2, 1.00, _N),

    RelationMapping("BioRED", "Association", "gene", "disease",
                    "associated-with", 2, 0.70, _N),
    RelationMapping("BioRED", "Association", "gene", "disease",
                    "causative-agent", 1, 0.30,
                    "Gene-disease pairs often imply a causal role."),

    RelationMapping("BioRED", "Association", "variant", "disease",
                    "associated-with", 2, 0.70, _N),
    RelationMapping("BioRED", "Association", "variant", "disease",
                    "causative-agent", 1, 0.30,
                    "Pathogenic variants are causal by definition."),

    RelationMapping("BioRED", "Association", "variant", "gene",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "gene",     "gene",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "chemical", "chemical",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "chemical", "species",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "chemical", "cell_line",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "gene",     "cell_line",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "disease",  "cell_line",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "gene",     "species",
                    "associated-with", 2, 1.00, _N),

    # =========================================================================
    # BioRED — Positive_Correlation
    # Entity A increases, induces, or upregulates entity B.  For chemical-disease
    # pairs this almost always means the chemical exacerbates or causes the
    # disease.  For gene-disease pairs it implies the gene promotes the disease.
    # =========================================================================
    RelationMapping("BioRED", "Positive_Correlation", "chemical", "disease",
                    "causes",          2, 0.60, _N),
    RelationMapping("BioRED", "Positive_Correlation", "chemical", "disease",
                    "causative-agent", 1, 0.30,
                    "SNOMED causative-agent applies when the mechanism is stated."),
    RelationMapping("BioRED", "Positive_Correlation", "chemical", "disease",
                    "associated-with", 2, 0.10,
                    "Residual when no directional mechanism is explicit."),

    RelationMapping("BioRED", "Positive_Correlation", "chemical", "gene",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Positive_Correlation", "chemical", "gene",
                    "causes",          2, 0.40,
                    "Chemical upregulates / induces gene expression."),

    RelationMapping("BioRED", "Positive_Correlation", "gene", "disease",
                    "causes",          2, 0.50, _N),
    RelationMapping("BioRED", "Positive_Correlation", "gene", "disease",
                    "causative-agent", 1, 0.40,
                    "Gene gain-of-function variants causing disease."),
    RelationMapping("BioRED", "Positive_Correlation", "gene", "disease",
                    "associated-with", 2, 0.10, _N),

    RelationMapping("BioRED", "Positive_Correlation", "variant", "disease",
                    "causes",          2, 0.50, _N),
    RelationMapping("BioRED", "Positive_Correlation", "variant", "disease",
                    "causative-agent", 1, 0.40, _N),
    RelationMapping("BioRED", "Positive_Correlation", "variant", "disease",
                    "associated-with", 2, 0.10, _N),

    RelationMapping("BioRED", "Positive_Correlation", "variant", "gene",
                    "causes",          2, 0.50,
                    "Variant increases gene expression or activity."),
    RelationMapping("BioRED", "Positive_Correlation", "variant", "gene",
                    "associated-with", 2, 0.50, _N),

    RelationMapping("BioRED", "Positive_Correlation", "gene", "gene",
                    "causes",          2, 0.50,
                    "Gene A promotes expression of gene B."),
    RelationMapping("BioRED", "Positive_Correlation", "gene", "gene",
                    "associated-with", 2, 0.50, _N),

    RelationMapping("BioRED", "Positive_Correlation", "chemical", "chemical",
                    "causes",          2, 0.50,
                    "Chemical A increases levels / activity of chemical B."),
    RelationMapping("BioRED", "Positive_Correlation", "chemical", "chemical",
                    "associated-with", 2, 0.50, _N),

    RelationMapping("BioRED", "Positive_Correlation", "disease", "disease",
                    "associated-with", 2, 0.70, _N),
    RelationMapping("BioRED", "Positive_Correlation", "disease", "disease",
                    "causes",          2, 0.30,
                    "Comorbidity where disease A predisposes to disease B."),

    # =========================================================================
    # BioRED — Negative_Correlation
    # Entity A decreases, inhibits, or reverses entity B.  For chemical-disease
    # pairs the most common reading is that the chemical treats / alleviates the
    # disease, making 'treats' the primary Tier 2 label.  For gene-disease pairs
    # a negative correlation more often implies the gene is protective.
    # =========================================================================
    RelationMapping("BioRED", "Negative_Correlation", "chemical", "disease",
                    "treats",          2, 0.70,
                    "Chemical decreasing disease severity is the canonical "
                    "'treats' relationship."),
    RelationMapping("BioRED", "Negative_Correlation", "chemical", "disease",
                    "associated-with", 2, 0.20,
                    "Cases where inhibition is indirect or context unclear."),
    RelationMapping("BioRED", "Negative_Correlation", "chemical", "disease",
                    "due-to",          1, 0.10,
                    "Rare: disease symptom reduced due to chemical intervention; "
                    "SNOMED due-to captures this formally."),

    RelationMapping("BioRED", "Negative_Correlation", "chemical", "gene",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Negative_Correlation", "chemical", "gene",
                    "interacts-with",  2, 0.40,
                    "Chemical inhibiting a gene product implies physical "
                    "or functional interaction."),

    RelationMapping("BioRED", "Negative_Correlation", "gene", "disease",
                    "associated-with", 2, 0.70,
                    "Protective gene-disease association without explicit mechanism."),
    RelationMapping("BioRED", "Negative_Correlation", "gene", "disease",
                    "causes",          2, 0.30,
                    "Loss-of-function of protective gene causes disease."),

    RelationMapping("BioRED", "Negative_Correlation", "variant", "disease",
                    "associated-with", 2, 0.80, _N),
    RelationMapping("BioRED", "Negative_Correlation", "variant", "disease",
                    "causes",          2, 0.20,
                    "Protective loss-of-function variant."),

    RelationMapping("BioRED", "Negative_Correlation", "variant", "gene",
                    "associated-with", 2, 0.70, _N),
    RelationMapping("BioRED", "Negative_Correlation", "variant", "gene",
                    "interacts-with",  2, 0.30, _N),

    RelationMapping("BioRED", "Negative_Correlation", "gene", "gene",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Negative_Correlation", "gene", "gene",
                    "interacts-with",  2, 0.40,
                    "Gene A downregulates gene B via direct interaction."),

    RelationMapping("BioRED", "Negative_Correlation", "chemical", "chemical",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Negative_Correlation", "chemical", "chemical",
                    "interacts-with",  2, 0.40,
                    "Chemical A inhibits or antagonises chemical B."),

    RelationMapping("BioRED", "Negative_Correlation", "disease", "disease",
                    "associated-with", 2, 1.00,
                    "Disease A being inversely associated with disease B; "
                    "no standard Tier 1 SNOMED attribute covers this."),

    # =========================================================================
    # BioRED — Bind
    # Physical molecular binding.  Always maps to interacts-with (Tier 2); there
    # is no SNOMED attribute for direct molecular binding between two entities
    # that are not finding-site or causative-agent candidates.
    # =========================================================================
    RelationMapping("BioRED", "Bind", "chemical", "gene",
                    "interacts-with", 2, 1.00,
                    "Small molecule binding to a protein / gene product."),
    RelationMapping("BioRED", "Bind", "chemical", "chemical",
                    "interacts-with", 2, 1.00,
                    "Ligand-receptor or chelation-style binding."),
    RelationMapping("BioRED", "Bind", "gene",     "gene",
                    "interacts-with", 2, 1.00,
                    "Protein-protein interaction."),
    RelationMapping("BioRED", "Bind", "variant",  "gene",
                    "interacts-with", 2, 1.00,
                    "Mutant protein binding its wild-type partner."),
    RelationMapping("BioRED", "Bind", "chemical", "variant",
                    "interacts-with", 2, 1.00, _N),

    # =========================================================================
    # BioRED — Cotreatment
    # Two chemicals are co-administered as part of a treatment regimen.
    # Always chemical-chemical; co-treats is the dedicated Tier 2 label.
    # =========================================================================
    RelationMapping("BioRED", "Cotreatment", "chemical", "chemical",
                    "co-treats", 2, 1.00,
                    "Two drugs given together; co-treats distinguishes this "
                    "from a general interaction."),

    # =========================================================================
    # BioRED — Drug_Interaction
    # Pharmacokinetic or pharmacodynamic interaction between two chemicals.
    # Distinct from Cotreatment in that it does not imply co-administration
    # for therapeutic purpose.
    # =========================================================================
    RelationMapping("BioRED", "Drug_Interaction", "chemical", "chemical",
                    "interacts-with", 2, 1.00,
                    "PK/PD drug-drug interaction; interacts-with is the "
                    "appropriate Tier 2 label."),

    # =========================================================================
    # BioRED — Conversion
    # Entity A is biochemically converted to entity B (metabolite, product).
    # Primarily chemical-chemical; occasionally chemical-gene when a substrate
    # is processed by an enzyme (the gene product).
    # =========================================================================
    RelationMapping("BioRED", "Conversion", "chemical", "chemical",
                    "converts-to", 2, 1.00,
                    "Metabolic or chemical transformation product."),
    RelationMapping("BioRED", "Conversion", "chemical", "gene",
                    "converts-to", 2, 1.00,
                    "Chemical converted by an enzyme (gene product)."),

    # =========================================================================
    # BioRED — Comparison
    # Experimental or clinical comparison between two entities of the same or
    # different types.  No mechanistic or causal claim; compared-with (Tier 2).
    # =========================================================================
    RelationMapping("BioRED", "Comparison", "chemical", "chemical",
                    "compared-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Comparison", "gene",     "gene",
                    "compared-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Comparison", "chemical", "disease",
                    "compared-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Comparison", "gene",     "disease",
                    "compared-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Comparison", "variant",  "disease",
                    "compared-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Comparison", "chemical", "gene",
                    "compared-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Comparison", "variant",  "gene",
                    "compared-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Comparison", "disease",  "disease",
                    "compared-with", 2, 1.00, _N),

    # =========================================================================
    # BioRED — reversed-order pairs
    # BioRED does not enforce canonical subject-object ordering.  The rows below
    # cover observed pairs where subject and object are swapped relative to the
    # groups above.  Semantics are symmetric for all BioRED relation types
    # (correlation, binding, comparison, co-treatment) so the probability
    # distributions mirror the forward direction.
    # =========================================================================

    # --- Association (reversed) ----------------------------------------------
    RelationMapping("BioRED", "Association", "chemical", "variant",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "disease",  "chemical",
                    "associated-with", 2, 0.80, _N),
    RelationMapping("BioRED", "Association", "disease",  "chemical",
                    "causative-agent", 1, 0.20,
                    "Reversed ordering of chemical-disease Association; "
                    "same semantics, same probability split."),
    RelationMapping("BioRED", "Association", "disease",  "gene",
                    "associated-with", 2, 0.70, _N),
    RelationMapping("BioRED", "Association", "disease",  "gene",
                    "causative-agent", 1, 0.30, _N),
    RelationMapping("BioRED", "Association", "disease",  "variant",
                    "associated-with", 2, 0.70, _N),
    RelationMapping("BioRED", "Association", "disease",  "variant",
                    "causative-agent", 1, 0.30, _N),
    RelationMapping("BioRED", "Association", "gene",     "chemical",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "variant",  "chemical",
                    "associated-with", 2, 1.00, _N),
    RelationMapping("BioRED", "Association", "variant",  "variant",
                    "associated-with", 2, 1.00, _N),

    # --- Positive_Correlation (reversed) -------------------------------------
    RelationMapping("BioRED", "Positive_Correlation", "chemical", "variant",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Positive_Correlation", "chemical", "variant",
                    "causes",          2, 0.40,
                    "Chemical upregulates variant expression / activity."),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "chemical",
                    "causes",          2, 0.60,
                    "Reversed ordering: same semantics as chemical-disease."),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "chemical",
                    "causative-agent", 1, 0.30, _N),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "chemical",
                    "associated-with", 2, 0.10, _N),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "gene",
                    "causes",          2, 0.50,
                    "Reversed ordering: same semantics as gene-disease."),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "gene",
                    "causative-agent", 1, 0.40, _N),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "gene",
                    "associated-with", 2, 0.10, _N),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "variant",
                    "causes",          2, 0.50,
                    "Reversed ordering: same semantics as variant-disease."),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "variant",
                    "causative-agent", 1, 0.40, _N),
    RelationMapping("BioRED", "Positive_Correlation", "disease",  "variant",
                    "associated-with", 2, 0.10, _N),
    RelationMapping("BioRED", "Positive_Correlation", "gene",     "chemical",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Positive_Correlation", "gene",     "chemical",
                    "causes",          2, 0.40,
                    "Reversed ordering: same semantics as chemical-gene."),
    RelationMapping("BioRED", "Positive_Correlation", "variant",  "chemical",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Positive_Correlation", "variant",  "chemical",
                    "causes",          2, 0.40, _N),

    # --- Negative_Correlation (reversed) -------------------------------------
    RelationMapping("BioRED", "Negative_Correlation", "chemical", "variant",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Negative_Correlation", "chemical", "variant",
                    "interacts-with",  2, 0.40,
                    "Chemical inhibiting a variant/mutant protein."),
    RelationMapping("BioRED", "Negative_Correlation", "disease",  "chemical",
                    "treats",          2, 0.70,
                    "Reversed ordering of chemical-disease: chemical still "
                    "treats disease regardless of annotation direction."),
    RelationMapping("BioRED", "Negative_Correlation", "disease",  "chemical",
                    "associated-with", 2, 0.20, _N),
    RelationMapping("BioRED", "Negative_Correlation", "disease",  "chemical",
                    "due-to",          1, 0.10, _N),
    RelationMapping("BioRED", "Negative_Correlation", "disease",  "gene",
                    "associated-with", 2, 0.70,
                    "Protective gene reduces disease severity; association "
                    "reading preferred when mechanism is unstated."),
    RelationMapping("BioRED", "Negative_Correlation", "disease",  "gene",
                    "causes",          2, 0.30,
                    "Loss-of-function of protective gene causes disease."),
    RelationMapping("BioRED", "Negative_Correlation", "gene",     "chemical",
                    "associated-with", 2, 0.60, _N),
    RelationMapping("BioRED", "Negative_Correlation", "gene",     "chemical",
                    "interacts-with",  2, 0.40,
                    "Reversed ordering: gene product inhibited by chemical."),
    RelationMapping("BioRED", "Negative_Correlation", "variant",  "chemical",
                    "associated-with", 2, 0.70, _N),
    RelationMapping("BioRED", "Negative_Correlation", "variant",  "chemical",
                    "interacts-with",  2, 0.30, _N),

    # --- Bind (reversed) -----------------------------------------------------
    RelationMapping("BioRED", "Bind", "gene", "chemical",
                    "interacts-with", 2, 1.00,
                    "Reversed ordering of chemical-gene Bind; "
                    "binding is symmetric."),

    # --- Cotreatment (unusual gene-chemical pairing) -------------------------
    RelationMapping("BioRED", "Cotreatment", "gene", "chemical",
                    "co-treats", 2, 1.00,
                    "5 observed instances; likely a protein therapeutic "
                    "(e.g. monoclonal antibody) co-administered with a small "
                    "molecule.  Annotated as gene product but functions as "
                    "a biologic drug in the treatment context."),
]

# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------


def get_mappings(
    source_corpus: str,
    source_relation_type: str,
    subject_semantic_class: str,
    object_semantic_class: str,
) -> List[RelationMapping]:
    """Return all mapping rows for the given (corpus, relation, subject, object) group."""
    return [
        m for m in RELATION_MAPPINGS
        if (
            m.source_corpus == source_corpus
            and m.source_relation_type == source_relation_type
            and m.subject_semantic_class == subject_semantic_class
            and m.object_semantic_class == object_semantic_class
        )
    ]


def iter_rows() -> Iterator[RelationMapping]:
    return iter(RELATION_MAPPINGS)


def get_all_target_relations() -> frozenset[str]:
    return ALL_TARGET_RELATIONS


def get_tier1_relations() -> frozenset[str]:
    return TIER1_RELATIONS


def get_tier2_relations() -> frozenset[str]:
    return TIER2_RELATIONS


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase1"
_FIELDNAMES = [f.name for f in fields(RelationMapping)]


def dump_csv(path: Optional[Path] = None) -> Path:
    """Write the full mapping table to *path* (default: OUTPUT_DIR/relation_schema_alignment.csv)."""
    if path is None:
        path = OUTPUT_DIR / "relation_schema_alignment.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_FIELDNAMES)
        for row in RELATION_MAPPINGS:
            writer.writerow(astuple(row))
    return path


# ---------------------------------------------------------------------------
# Sanity check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import defaultdict

    groups: dict = defaultdict(float)
    for m in RELATION_MAPPINGS:
        key = (m.source_corpus, m.source_relation_type,
               m.subject_semantic_class, m.object_semantic_class)
        groups[key] += m.probability

    errors = [(k, v) for k, v in groups.items() if abs(v - 1.0) > 1e-9]
    if errors:
        print("PROBABILITY SUM ERRORS:")
        for k, v in errors:
            print(f"  {k} -> sum={v:.4f}")
    else:
        print(f"OK — {len(RELATION_MAPPINGS)} rows, "
              f"{len(groups)} groups, all probabilities sum to 1.0")

    out = dump_csv()
    print(f"CSV written to {out}")
