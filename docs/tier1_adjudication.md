# Tier-1 Relation Adjudication

Each instance is reviewed independently by two reviewers using only the cited
document context and the displayed entity pair. Reviewers choose exactly one:

- `tier1-correct`: the SNOMED-native attribute is explicitly supported.
- `tier2-correct`: the empirical Tier-2 reading is supported but the promoted
  Tier-1 attribute is too specific.
- `both-plausible`: both readings are defensible from the text.
- `neither`: neither proposed reading is supported.
- `insufficient-context`: the available abstract does not permit adjudication.

Reviewers must not use MRCM validity as evidence that the relation occurred;
MRCM is a necessary type constraint only. CTD, MED-RT, and SNOMED stated
relationships may be recorded as external attestation but do not replace the
textual judgment. Agreement automatically becomes consensus. Disagreements are
resolved by a third adjudicator and recorded in `consensus` with a short note.
