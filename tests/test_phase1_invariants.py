from __future__ import annotations

import csv
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import main  # noqa: E402
import mrcm  # noqa: E402
import relation_schema  # noqa: E402
from config import MRCM_FILES  # noqa: E402


class Phase1InvariantTests(unittest.TestCase):
    def test_main_registers_phase_1_steps(self) -> None:
        self.assertEqual(
            {"1.1", "1.2", "1.3", "1.4", "1.5", "1.6"},
            set(main.STEPS),
        )

    def test_relation_schema_csv_matches_memory(self) -> None:
        csv_path = OUTPUT_DIR / "relation_schema_alignment.csv"
        self.assertTrue(csv_path.exists(), f"Missing {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as fh:
            csv_rows = list(csv.DictReader(fh))

        memory_rows = [row.__dict__ for row in relation_schema.iter_rows()]
        normalized_csv_rows = []
        for row in csv_rows:
            item = dict(row)
            item["tier"] = int(item["tier"])
            item["probability"] = float(item["probability"])
            normalized_csv_rows.append(item)

        self.assertEqual(memory_rows, normalized_csv_rows)

    def test_relation_schema_probability_groups(self) -> None:
        groups: dict[tuple[str, str, str, str], float] = defaultdict(float)
        for row in relation_schema.iter_rows():
            key = (
                row.source_corpus,
                row.source_relation_type,
                row.subject_semantic_class,
                row.object_semantic_class,
            )
            groups[key] += row.probability

        bad = {key: total for key, total in groups.items() if abs(total - 1.0) > 1e-9}
        self.assertEqual({}, bad)

    def test_mrcm_json_matches_configured_release_files(self) -> None:
        path = OUTPUT_DIR / "mrcm_constraints.json"
        self.assertTrue(path.exists(), f"Missing {path}")
        data = json.loads(path.read_text(encoding="utf-8"))

        expected = {key: value.name for key, value in MRCM_FILES.items()}
        self.assertEqual(expected, data["metadata"]["release_files"])

    def test_mrcm_relation_constraints_are_complete(self) -> None:
        path = OUTPUT_DIR / "mrcm_constraints.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        constraints = data["relation_constraints"]

        self.assertEqual(set(relation_schema.TIER1_RELATIONS), set(constraints))
        self.assertFalse(data["metadata"]["tier1_relations_without_attribute_mapping"])
        self.assertFalse(data["metadata"]["attribute_mappings_outside_tier1"])

        for label, block in constraints.items():
            with self.subTest(label=label):
                self.assertEqual(mrcm.RELATION_TO_ATTRIBUTE_ID[label], block["snomed_attribute_id"])
                self.assertTrue(block["snomed_attribute_name"])
                self.assertGreater(len(block["domains"]), 0)
                self.assertGreater(len(block["ranges"]), 0)
                for domain in block["domains"]:
                    self.assertTrue(domain["domain_root_concept_ids"])
                for range_row in block["ranges"]:
                    self.assertTrue(range_row["range_root_concept_ids"])
                    self.assertEqual(
                        len(range_row["range_root_concept_ids"]),
                        len(range_row["range_root_concept_names"]),
                    )

    def test_snomed_hierarchy_artifacts_and_stats(self) -> None:
        graph_path = OUTPUT_DIR / "snomed_hierarchy.pkl"
        ancestors_path = OUTPUT_DIR / "snomed_ancestors.pkl"
        stats_path = OUTPUT_DIR / "snomed_hierarchy_stats.json"

        self.assertTrue(graph_path.exists(), f"Missing {graph_path}")
        self.assertTrue(ancestors_path.exists(), f"Missing {ancestors_path}")
        self.assertTrue(stats_path.exists(), f"Missing {stats_path}")

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.assertEqual("138875005", stats["snomed_root"])
        self.assertGreater(stats["active_concepts"], 100_000)
        self.assertGreater(stats["is_a_edges"], stats["active_concepts"])
        self.assertGreaterEqual(stats["top_level_hierarchy_count"], 10)
        self.assertGreater(stats["mapped_concepts_total"], 0)
        self.assertGreater(stats["mapped_concepts_in_graph"], 0)
        self.assertLess(
            len(stats["mapped_concepts_missing"]),
            stats["mapped_concepts_total"],
        )

        anchors = stats["mrcm_anchors_in_graph"]
        for required_anchor in ("404684003", "71388002", "105590001"):
            self.assertTrue(anchors.get(required_anchor), required_anchor)


if __name__ == "__main__":
    unittest.main()
