import inspect
import unittest

import numpy as np
import pandas as pd

import src.team_advanced_broader_validation as audit


class TeamAdvancedBroaderValidationTests(unittest.TestCase):
    def test_validation_folds_match_requested_designs(self):
        folds = audit.validation_folds(audit.ALL_SEASONS)
        expanding = [
            fold
            for fold in folds
            if fold["validation_design"] == "expanding_window"
        ]
        leave_one_out = [
            fold
            for fold in folds
            if fold["validation_design"] == "leave_one_season_out"
        ]
        self.assertEqual(len(expanding), 7)
        self.assertEqual(len(leave_one_out), 10)
        self.assertEqual(expanding[0]["train_seasons"][-1], "2017-18")
        self.assertEqual(expanding[0]["test_season"], "2018-19")
        self.assertEqual(expanding[-1]["test_season"], "2024-25")

    def test_playoff_round_is_derived_from_game_id(self):
        self.assertEqual(
            audit.playoff_round_from_game_id("0042400401"),
            "nba_finals",
        )
        self.assertEqual(
            audit.playoff_round_from_game_id("0042400201"),
            "conference_semifinals",
        )

    def test_metric_row_includes_requested_metrics(self):
        row = audit.metric_row(
            model="base_only",
            target=pd.Series([0, 0, 1, 1]),
            probability=np.array([0.2, 0.4, 0.6, 0.8]),
            average_probability_movement=0.03,
        )
        for column in (
            "roc_auc",
            "brier_score",
            "log_loss",
            "accuracy",
            "f1",
            "expected_calibration_error",
            "average_probability_movement",
        ):
            self.assertIn(column, row)
        self.assertAlmostEqual(row["average_probability_movement"], 0.03)

    def test_summary_counts_fold_improvements(self):
        rows = []
        for fold, base_brier, advanced_brier in (
            ("fold_a", 0.25, 0.20),
            ("fold_b", 0.20, 0.22),
        ):
            for model, brier, roc, loss in (
                ("base_only", base_brier, 0.65, 0.65),
                (
                    "base_plus_team_advanced",
                    advanced_brier,
                    0.66,
                    0.64 if advanced_brier < base_brier else 0.67,
                ),
            ):
                rows.append(
                    {
                        "record_type": "fold",
                        "validation_design": "expanding_window",
                        "fold": fold,
                        "test_season": "2024-25",
                        "model": model,
                        "roc_auc": roc,
                        "brier_score": brier,
                        "log_loss": loss,
                        "accuracy": 0.6,
                        "f1": 0.6,
                        "expected_calibration_error": 0.05,
                        "average_probability_movement": 0.03,
                    }
                )
        summary = audit.summarize_validation(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(summary["scope"], "all_games")
        self.assertEqual(summary["folds"], 2)
        self.assertEqual(summary["brier_improved_folds"], 1)
        self.assertEqual(summary["log_loss_improved_folds"], 1)
        self.assertEqual(summary["roc_auc_improved_folds"], 2)

    def test_subgroups_cover_requested_probability_contexts(self):
        frame = pd.DataFrame(
            {
                "GAME_ID": ["0042400401"] * 6,
                "playoff_round": ["nba_finals"] * 6,
                "game_number_audit": [1, 2, 3, 4, 5, 6],
                "base_probability": [0.48, 0.52, 0.7, 0.3, 0.6, 0.4],
                "advanced_probability_delta": [
                    0.03,
                    -0.03,
                    0.01,
                    -0.01,
                    0.04,
                    -0.04,
                ],
            }
        )
        masks = audit._subgroup_masks(frame)
        self.assertIn("playoff_round=nba_finals", masks)
        self.assertIn("game_number=6", masks)
        self.assertEqual(int(masks["close_prediction_45_55"].sum()), 2)
        self.assertEqual(
            int(masks["high_confidence_over_65"].sum()), 2
        )
        self.assertEqual(
            int(masks["advanced_movement_over_2_pct"].sum()), 4
        )

    def test_module_does_not_write_models_or_import_streamlit(self):
        source = inspect.getsource(audit)
        self.assertNotIn("streamlit", source)
        self.assertNotIn("joblib.dump", source)
        self.assertNotIn("train_production_models", source)


if __name__ == "__main__":
    unittest.main()
