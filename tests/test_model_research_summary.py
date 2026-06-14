import csv
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "data" / "processed" / "model_research_summary.csv"
SUMMARY_DOC = ROOT / "docs" / "model_research_summary.md"


class ModelResearchSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_id = {row["experiment_id"]: row for row in cls.rows}
        cls.doc = SUMMARY_DOC.read_text(encoding="utf-8")

    def test_summary_files_exist_and_cover_all_experiments(self):
        self.assertTrue(SUMMARY_CSV.is_file())
        self.assertTrue(SUMMARY_DOC.is_file())
        self.assertEqual(
            set(self.by_id),
            {
                "base_model",
                "series_context",
                "recent_form",
                "momentum_state",
                "momentum_subgroups",
                "upset_response",
                "nba_api_sources",
                "advanced_v3",
                "team_advanced",
                "player_rotation",
            },
        )
        self.assertEqual(len(self.rows), 10)

    def test_required_summary_columns_are_present(self):
        required = {
            "purpose",
            "data_used",
            "validation_design",
            "best_result",
            "base_roc_auc",
            "best_roc_auc",
            "base_brier_score",
            "best_brier_score",
            "base_log_loss",
            "best_log_loss",
            "leakage_protections",
            "production_recommendation",
            "should_ship",
            "rationale",
        }
        self.assertTrue(required.issubset(self.rows[0]))

    def test_production_recommendations_match_research_decision(self):
        self.assertEqual(
            self.by_id["base_model"]["production_recommendation"],
            "keep_production",
        )
        self.assertEqual(
            self.by_id["series_context"]["production_recommendation"],
            "contextual_only",
        )
        for experiment_id in (
            "recent_form",
            "momentum_state",
            "momentum_subgroups",
            "upset_response",
            "team_advanced",
            "player_rotation",
        ):
            self.assertEqual(
                self.by_id[experiment_id]["production_recommendation"],
                "research_only",
            )
            self.assertEqual(self.by_id[experiment_id]["should_ship"], "no")

    def test_small_advanced_and_rotation_gains_remain_research_only(self):
        advanced = self.by_id["team_advanced"]
        self.assertLess(float(advanced["best_brier_score"]), float(advanced["base_brier_score"]))
        self.assertLess(float(advanced["best_roc_auc"]), float(advanced["base_roc_auc"]))
        self.assertEqual(advanced["production_recommendation"], "research_only")

        rotation = self.by_id["player_rotation"]
        self.assertLess(float(rotation["best_brier_score"]), float(rotation["base_brier_score"]))
        self.assertEqual(rotation["production_recommendation"], "research_only")
        self.assertIn("expanding-window", rotation["rationale"].lower())

    def test_document_states_canonical_conclusion(self):
        self.assertIn(
            "Production should remain the calibrated base game model plus the app's current safe behavior.",
            self.doc,
        )
        self.assertIn("Series context", self.doc)
        self.assertIn("research-only", self.doc)
        self.assertIn("834/834 playoff games", self.doc)


if __name__ == "__main__":
    unittest.main()
