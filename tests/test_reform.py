from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config
import mrcm_validity
from canon_dataset import NO_RELATION_ID, RELATION_LABEL_TO_ID, enumerate_pairs
from csp_solver import load_constraint_tables, solve_document
from ctd_attestation import load_direct_evidence
from error_analysis import sample_errors
from phase4_evaluate import evaluate as evaluate_phase4
from contextualize_synthetic import contextualize
from relation_schema import TIER1_RELATIONS, TIER2_RELATIONS
from unified_format import (
    Document,
    EntityMention,
    Relation,
    derive_jsonl_cache,
    write_jsonl,
    write_jsonld_documents,
)


class JsonLdContractTest(unittest.TestCase):
    def test_round_trip_preserves_provenance_and_candidates(self):
        doc = Document(
            pmid="1", corpus="BioRED", split="test", title="Drug disease", abstract="",
            text="Drug disease",
            entities=[
                EntityMention("T1", 0, 4, "Drug", "ChemicalEntity", "chemical",
                              ner_type="substance", original_code="D000001",
                              mapped_snomed_id="105590001", mapping_confidence=0.9,
                              snomed_active=True, source_concept_uri="http://id.nlm.nih.gov/mesh/D000001",
                              normalized_concept_uri="http://snomed.info/id/105590001",
                              mapping_property="skos:closeMatch", umls_stys=["Chemical"]),
                EntityMention("T2", 5, 12, "disease", "DiseaseOrPhenotypicFeature", "disease",
                              ner_type="clinical_finding", original_code="D000002"),
            ],
            relations=[Relation(0, 1, "Association", "associated-with", 2, 0.8,
                                confidence=0.7,
                                target_candidates=[{"target_relation": "associated-with", "tier": 2,
                                                    "probability": 0.8}])],
        )
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "canonical"
            cache = Path(tmp) / "cache.jsonl"
            self.assertEqual(write_jsonld_documents(iter([doc]), canonical), 1)
            self.assertEqual(derive_jsonl_cache(canonical, cache), 1)
            loaded = json.loads(cache.read_text().strip())
            self.assertEqual(loaded["schema_version"], "3.0.0")
            self.assertEqual(loaded["entities"][0]["original_code"], "D000001")
            self.assertEqual(loaded["entities"][0]["ner_type"], "substance")
            self.assertEqual(loaded["relations"][0]["target_candidates"][0]["probability"], 0.8)

    def test_contextual_synthetic_is_canonical_first(self):
        entities = [
            EntityMention("T1", 0, 4, "Drug", "Chemical", "chemical",
                          mapped_snomed_id="100"),
            EntityMention("T2", 5, 12, "disease", "Disease", "disease",
                          mapped_snomed_id="200"),
        ]
        synthetic = Document("s1", "SNOMED_synthetic", "train", "", "", "Drug disease",
                             entities, [Relation(0, 1, "causative-agent", "causative-agent")])
        context = Document("p1", "PubTator_silver", "train", "Drug disease", "",
                           "Drug disease", entities, [])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synth_path, context_path, output_path = (
                tmp / "synthetic.jsonl", tmp / "context.jsonl", tmp / "contextual.jsonl")
            write_jsonl(iter([synthetic]), synth_path)
            write_jsonl(iter([context]), context_path)
            summary = contextualize(synth_path, [context_path], output_path,
                                    min_causative=1, canonical_dir=tmp / "canonical")
            cached = json.loads(output_path.read_text().strip())
        self.assertEqual(summary["documents"], 1)
        self.assertEqual(cached["schema_version"], "3.0.0")
        self.assertEqual(cached["relations"][0]["target_relation"], "causative-agent")


class RelationBagTest(unittest.TestCase):
    def test_all_repeated_mentions_share_positive_bag(self):
        entities = [
            {"original_code": "C1", "mapping_confidence": 1.0},
            {"original_code": "D1", "mapping_confidence": 1.0},
            {"original_code": "C1", "mapping_confidence": 1.0},
            {"original_code": "D1", "mapping_confidence": 1.0},
        ]
        relations = [{
            "subject_idx": 0, "object_idx": 1, "target_relation": "causes",
            "target_candidates": [
                {"target_relation": "causes", "probability": 0.7},
                {"target_relation": "causative-agent", "probability": 0.3},
            ],
        }]
        indices, labels, _, targets, bags = enumerate_pairs(
            entities, relations, survivor_index={i: i for i in range(4)}, neg_ratio=0)
        self.assertEqual(set(indices), {(0, 1), (0, 3), (2, 1), (2, 3)})
        self.assertEqual(len(set(bags)), 1)
        self.assertTrue(all(label == RELATION_LABEL_TO_ID["causes"] for label in labels))
        self.assertAlmostEqual(targets[0][RELATION_LABEL_TO_ID["causative-agent"]], 0.3)

    def test_hard_label_and_unweighted_ablation_contract(self):
        entities = [
            {"original_code": "C1", "mapping_confidence": 0.2},
            {"original_code": "D1", "mapping_confidence": 0.3},
        ]
        relations = [{
            "subject_idx": 0, "object_idx": 1, "target_relation": "causes",
            "confidence": 0.4,
            "target_candidates": [
                {"target_relation": "causes", "probability": 0.6},
                {"target_relation": "causative-agent", "probability": 0.4},
            ],
        }]
        _, labels, weights, targets, _ = enumerate_pairs(
            entities, relations, survivor_index={0: 0, 1: 1}, neg_ratio=0,
            hard_targets=True, confidence_weighting=False)
        causes = RELATION_LABEL_TO_ID["causes"]
        self.assertEqual(labels, [causes])
        self.assertEqual(weights, [1.0])
        self.assertEqual(targets[0][causes], 1.0)
        self.assertEqual(sum(targets[0]), 1.0)


class ConstraintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_constraint_tables(
            config.MRCM_CONSTRAINTS_JSON, config.SNOMED_ANCESTORS_PKL,
            semantic_network_path=config.UMLS_SEMANTIC_NETWORK_FILES["srstre1"])

    def test_tier1_epsilon_escalation(self):
        doc = {
            "entities": [
                {"type_candidates": [{"sem_class": "clinical_finding", "score": 0.9}],
                 "stys": ["Disease or Syndrome"]},
                {"type_candidates": [{"sem_class": "substance", "score": 0.9}],
                 "stys": ["Pharmacologic Substance"]},
            ],
            "pairs": [{"i": 0, "j": 1, "rel_candidates": [
                {"label": "causes", "score": 0.60},
                {"label": "causative-agent", "score": 0.58},
                {"label": "no-relation", "score": 0.01},
            ]}],
        }
        result = solve_document(doc, self.tables)
        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["assignment"]["pairs"][0]["relation"], "causative-agent")
        self.assertEqual(result["assignment"]["pairs"][0]["event"], "tier1-escalation")

    def test_invalid_tier2_abstains(self):
        doc = {
            "entities": [
                {"type_candidates": [{"sem_class": "gene", "score": 0.9}], "stys": ["Gene or Genome"]},
                {"type_candidates": [{"sem_class": "cell_line", "score": 0.9}], "stys": ["Cell"]},
            ],
            "pairs": [{"i": 0, "j": 1, "rel_candidates": [
                {"label": "treats", "score": 0.8}, {"label": "no-relation", "score": 0.2},
            ]}],
        }
        result = solve_document(doc, self.tables)
        self.assertEqual(result["assignment"]["pairs"][0]["relation"], "no-relation")
        self.assertEqual(result["assignment"]["pairs"][0]["event"], "tier2-sn-rejection")


class CtdTest(unittest.TestCase):
    def test_direct_evidence_only(self):
        payload = (
            "# comment\nChemicalName\tChemicalID\tCasRN\tDiseaseName\tDiseaseID\tDirectEvidence\n"
            "A\tMESH:C1\t\tD\tMESH:D1\tmarker/mechanism\n"
            "B\tMESH:C2\t\tD\tMESH:D2\ttherapeutic\n"
            "C\tMESH:C3\t\tD\tMESH:D3\t\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ctd.tsv"
            path.write_text(payload)
            lookup = load_direct_evidence(path)
        self.assertIn(("C1", "D1"), lookup["causes"])
        self.assertIn(("C2", "D2"), lookup["treats"])
        self.assertNotIn(("C3", "D3"), lookup["causes"])


class ErrorAnalysisTest(unittest.TestCase):
    def test_error_sample_contains_review_contract(self):
        gold = {
            "corpus": "BioRED", "pmid": "1", "title": "x", "abstract": "y",
            "entities": [{"span_start": 0, "span_end": 1, "ner_type": "clinical_finding",
                          "mapped_snomed_id": "1", "original_code": "D1"}],
            "relations": [],
        }
        prediction = {
            "corpus": "BioRED", "pmid": "1", "csp_status": "sat",
            "neural": {"entities": [], "pairs": []},
            "csp": {"entities": [], "pairs": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, prediction_path, output_path = (
                tmp / "gold.jsonl", tmp / "pred.jsonl", tmp / "review.jsonl")
            gold_path.write_text(json.dumps(gold) + "\n")
            prediction_path.write_text(json.dumps(prediction) + "\n")
            summary = sample_errors(prediction_path, gold_path, output_path, size=100)
            row = json.loads(output_path.read_text().strip())
        self.assertEqual(summary["sampled_documents"], 1)
        self.assertEqual(row["error_id"], "ERR-0001")
        self.assertIn("constraint-over-restriction", row["allowed_categories"])


class Phase4EvaluationTest(unittest.TestCase):
    def test_relation_override_effect_is_scored_against_gold(self):
        entities = [
            {"span_start": 0, "span_end": 4, "ner_type": "substance",
             "semantic_class": "chemical", "original_code": "C1"},
            {"span_start": 5, "span_end": 12, "ner_type": "clinical_finding",
             "semantic_class": "disease", "original_code": "D1"},
        ]
        gold = {"corpus": "BioRED", "pmid": "1", "entities": entities,
                "relations": [{"subject_idx": 0, "object_idx": 1,
                               "target_relation": "causes"}]}
        assigned_entities = [
            {"span_start": 0, "span_end": 4, "type": "substance", "concept": None,
             "stys": ["Pharmacologic Substance"]},
            {"span_start": 5, "span_end": 12, "type": "clinical_finding", "concept": None,
             "stys": ["Disease or Syndrome"]},
        ]
        prediction = {
            "corpus": "BioRED", "pmid": "1", "csp_status": "sat",
            "neural": {"entities": assigned_entities,
                       "pairs": [{"i": 0, "j": 1, "relation": "treats"}]},
            "csp": {"entities": assigned_entities,
                    "pairs": [{"i": 0, "j": 1, "relation": "causes",
                               "event": "tier2-sn-rejection"}]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold_path, prediction_path, output_path = (
                tmp / "gold.jsonl", tmp / "pred.jsonl", tmp / "metrics.json")
            gold_path.write_text(json.dumps(gold) + "\n")
            prediction_path.write_text(json.dumps(prediction) + "\n")
            metrics = evaluate_phase4(prediction_path, gold_path, output_path,
                                      use_ctd_attestation=False)
        self.assertEqual(metrics["relation"]["f1"], 1.0)
        self.assertEqual(
            metrics["csp_override_effects"]["tier2-sn-rejection"]["flip_to_correct"], 1)


if __name__ == "__main__":
    unittest.main()
