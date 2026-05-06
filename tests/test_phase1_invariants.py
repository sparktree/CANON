from __future__ import annotations

import csv
import json
import pickle
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
import snomed_hierarchy as sh  # noqa: E402
from config import MRCM_FILES  # noqa: E402

# Known fixtures (stable SNOMED concepts present in this release).
_PNEUMONIA        = "233604007"   # Pneumonia — Clinical finding
_CLINICAL_FINDING = "404684003"   # Top-level: Clinical finding
_SUBSTANCE        = "105590001"   # Top-level: Substance
_PROCEDURE        = "71388002"    # Top-level: Procedure
_DOXORUBICIN      = "372817009"   # Doxorubicin (mapped chemical in mesh_to_snomed.csv)
_SNOMED_ROOT      = sh.SNOMED_ROOT


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
        """Artifact files exist and stats JSON has expected shape + values."""
        self.assertTrue(sh.GRAPH_PKL.exists(),    f"Missing {sh.GRAPH_PKL}")
        self.assertTrue(sh.ANCESTORS_PKL.exists(), f"Missing {sh.ANCESTORS_PKL}")
        self.assertTrue(sh.STATS_JSON.exists(),    f"Missing {sh.STATS_JSON}")

        stats = json.loads(sh.STATS_JSON.read_text(encoding="utf-8"))

        # Basic shape.
        self.assertEqual(_SNOMED_ROOT, stats["snomed_root"])
        self.assertGreater(stats["active_concepts"], 100_000)
        self.assertGreater(stats["is_a_edges"], stats["active_concepts"])
        self.assertGreaterEqual(stats["top_level_hierarchy_count"], 10)
        self.assertGreater(stats["mapped_concepts_total"], 0)
        self.assertGreater(stats["mapped_concepts_in_graph"], 0)
        self.assertLess(
            len(stats["mapped_concepts_missing"]),
            stats["mapped_concepts_total"],
        )

        # multi_inheritance_concepts key is present (added in Dbport).
        self.assertIn("multi_inheritance_concepts", stats)
        self.assertGreaterEqual(stats["multi_inheritance_concepts"], 0)

        # mrcm_anchors_in_graph is now a list of dicts, not a plain dict.
        anchor_list = stats["mrcm_anchors_in_graph"]
        self.assertIsInstance(anchor_list, list)
        anchor_ids_in_graph = {
            entry["concept_id"]
            for entry in anchor_list
            if entry.get("in_graph")
        }
        for required in (_CLINICAL_FINDING, _PROCEDURE, _SUBSTANCE):
            self.assertIn(required, anchor_ids_in_graph, required)

        # top_level_hierarchies list has names resolved.
        tl_list = stats.get("top_level_hierarchies", [])
        self.assertIsInstance(tl_list, list)
        self.assertGreater(len(tl_list), 0)
        for entry in tl_list:
            self.assertIn("concept_id", entry)
            self.assertIn("name", entry)
            self.assertIn("primary_count", entry)

    def test_snomed_hierarchy_pickle_format(self) -> None:
        """Pickles use the versioned envelope format introduced in Dbport."""
        with sh.GRAPH_PKL.open("rb") as fh:
            payload = pickle.load(fh)
        self.assertIsInstance(payload, dict, "Graph pickle must be a dict envelope")
        self.assertIn("signature", payload)
        self.assertIn("graph", payload)

        with sh.ANCESTORS_PKL.open("rb") as fh:
            anc_payload = pickle.load(fh)
        self.assertIsInstance(anc_payload, dict, "Ancestors pickle must be a dict envelope")
        self.assertIn("signature", anc_payload)
        self.assertIn("ancestors", anc_payload)

        self.assertEqual(payload["signature"], anc_payload["signature"],
                         "Graph and ancestors pickles must share the same signature")

    def test_snomed_hierarchy_graph_semantics(self) -> None:
        """Graph structure, depths, semantic types, and multi-inheritance."""
        G, anc = sh.load_or_build(force=False, verbose=False)

        depths    = dict(G.nodes(data="depth",              default=-1))
        sem_types = dict(G.nodes(data="semantic_type",       default=None))
        tl_mem    = dict(G.nodes(data="top_level_hierarchies", default=()))

        # Root.
        self.assertEqual(0, depths[_SNOMED_ROOT])
        self.assertEqual(0, len(list(G.successors(_SNOMED_ROOT))),
                         "Root must have no parents in the child->parent graph")

        # Clinical finding is a direct child of root.
        self.assertEqual(1, depths[_CLINICAL_FINDING])

        # Pneumonia.
        self.assertIn(_PNEUMONIA, G, "Pneumonia must be in the graph")
        self.assertGreater(depths[_PNEUMONIA], 1)
        self.assertEqual(_CLINICAL_FINDING, sem_types[_PNEUMONIA])

        # Doxorubicin is under Substance, not Clinical finding.
        self.assertEqual(_SUBSTANCE, sem_types[_DOXORUBICIN])

        # top_level_hierarchies attribute is present and non-empty for most nodes.
        self.assertIn(_CLINICAL_FINDING, tl_mem[_PNEUMONIA],
                      "Clinical finding must be in Pneumonia's top_level_hierarchies")
        self.assertIn(_SUBSTANCE, tl_mem[_DOXORUBICIN])
        self.assertNotIn(_SUBSTANCE, tl_mem[_PNEUMONIA],
                         "Pneumonia must NOT be under Substance")

    def test_snomed_hierarchy_ancestor_and_descendant_sets(self) -> None:
        """Ancestor/descendant structures are correct and API helpers work."""
        G, anc = sh.load_or_build(force=False, verbose=False)

        # Ancestor sets for mapped concepts.
        pneu_anc = anc.get(_PNEUMONIA, frozenset())
        self.assertIn(_CLINICAL_FINDING, pneu_anc)
        self.assertIn(_SNOMED_ROOT, pneu_anc)
        self.assertNotIn(_SUBSTANCE, pneu_anc)

        dox_anc = anc.get(_DOXORUBICIN, frozenset())
        self.assertIn(_SUBSTANCE, dox_anc)
        self.assertNotIn(_CLINICAL_FINDING, dox_anc)

        # Descendant sets for MRCM anchors.
        cf_desc = anc.get(f"descendant:{_CLINICAL_FINDING}", frozenset())
        self.assertGreater(len(cf_desc), 100_000)
        self.assertIn(_PNEUMONIA, cf_desc)
        self.assertNotIn(_PNEUMONIA, anc.get(f"descendant:{_SUBSTANCE}", frozenset()))

        root_desc = anc.get(f"descendant:{_SNOMED_ROOT}", frozenset())
        self.assertEqual(G.number_of_nodes() - 1, len(root_desc))

        # is_descendant_of helper.
        self.assertTrue(sh.is_descendant_of(_PNEUMONIA,        _CLINICAL_FINDING, anc))
        self.assertFalse(sh.is_descendant_of(_CLINICAL_FINDING, _PNEUMONIA,        anc))
        self.assertTrue(sh.is_descendant_of(_DOXORUBICIN,       _SUBSTANCE,        anc))
        self.assertFalse(sh.is_descendant_of(_PNEUMONIA,         _SUBSTANCE,        anc))

        # get_ancestors: precomputed path.
        got = sh.get_ancestors(_PNEUMONIA, anc)
        self.assertIsInstance(got, frozenset)
        self.assertIn(_CLINICAL_FINDING, got)

        # get_ancestors: runtime BFS fallback for concept not in mapping table.
        # Use Clinical finding itself — it's not a mapped concept but is in the graph.
        fallback = sh.get_ancestors(_CLINICAL_FINDING, anc, G=G)
        self.assertIsInstance(fallback, frozenset)
        self.assertIn(_SNOMED_ROOT, fallback)

        # get_ancestors: unknown concept, no G → empty frozenset.
        self.assertEqual(frozenset(), sh.get_ancestors("NONEXISTENT", anc))

    def test_snomed_hierarchy_new_api_functions(self) -> None:
        """get_top_level_hierarchies() and get_depth() introduced in Dbport."""
        G, _ = sh.load_or_build(force=False, verbose=False)

        # get_top_level_hierarchies.
        pneu_tl = sh.get_top_level_hierarchies(G, _PNEUMONIA)
        self.assertIsInstance(pneu_tl, tuple)
        self.assertIn(_CLINICAL_FINDING, pneu_tl)

        dox_tl = sh.get_top_level_hierarchies(G, _DOXORUBICIN)
        self.assertIn(_SUBSTANCE, dox_tl)
        self.assertNotIn(_CLINICAL_FINDING, dox_tl)

        # Unknown concept returns empty tuple.
        self.assertEqual((), sh.get_top_level_hierarchies(G, "NONEXISTENT"))

        # get_depth.
        self.assertEqual(0, sh.get_depth(G, _SNOMED_ROOT))
        self.assertEqual(1, sh.get_depth(G, _CLINICAL_FINDING))
        self.assertIsNone(sh.get_depth(G, "NONEXISTENT"))
        pneu_depth = sh.get_depth(G, _PNEUMONIA)
        self.assertIsNotNone(pneu_depth)
        self.assertGreater(pneu_depth, 1)


if __name__ == "__main__":
    unittest.main()
