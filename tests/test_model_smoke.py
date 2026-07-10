from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from canon_dataset import BIO_LABEL_TO_ID
from csp_solver import model_predict
from heads import MultiTaskModel


class EndToEndInferenceSmokeTest(unittest.TestCase):
    def test_crf_predictions_feed_norm_relation_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp)
            BertModel(BertConfig(
                vocab_size=20, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32,
            )).save_pretrained(encoder)
            model = MultiTaskModel(str(encoder), num_concepts=2)
            model.norm_head._concept_ids = ["105590001", "373873005"]
            model.norm_head._concept_id_to_row = {"105590001": 0, "373873005": 1}
            with torch.no_grad():
                model.norm_head.concept_emb.copy_(torch.randn(2, 16))
                model.ner_head.classifier.weight.zero_()
                model.ner_head.classifier.bias.zero_()
                model.ner_head.classifier.bias[BIO_LABEL_TO_ID["B-substance"]] = 10
                model.ner_head.crf.start_transitions.zero_()
                model.ner_head.crf.end_transitions.zero_()
                model.ner_head.crf.transitions.zero_()
            batch = {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "offset_mapping": torch.tensor([[[0, 0], [0, 4], [5, 9], [0, 0]]]),
                "entity_token_spans": [[]], "entity_original": [[]],
                "pmids": ["smoke"], "corpora": ["fixture"],
                "raw_docs": [{"text": "drug drug"}],
            }
            rows = model_predict(
                model, [batch], torch.device("cpu"), top_k_types=2,
                top_k_concepts=2, top_k_relations=3,
                sty_lookup={"105590001": ["Chemical"],
                            "373873005": ["Pharmacologic Substance"]},
            )
        self.assertEqual(len(rows[0]["entities"]), 2)
        self.assertEqual(len(rows[0]["pairs"]), 2)
        self.assertTrue(all(
            "no-relation" in {candidate["label"] for candidate in pair["rel_candidates"]}
            for pair in rows[0]["pairs"]
        ))


if __name__ == "__main__":
    unittest.main()
