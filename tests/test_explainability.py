import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import app
from app import (
    display_factor_table,
    filter_explanation_features_for_mode,
    filter_noninformative_factors,
    local_factor_table,
    save_prediction_explanation,
)
from src.model import extract_feature_importances


class ExplainabilityDirectionTests(unittest.TestCase):
    def test_pushes_toward_matches_model_probability_delta(self):
        class DirectionalPipeline:
            def predict_proba(self, frame):
                score = frame["good_for_a"].to_numpy(dtype=float) - frame["good_for_b"].to_numpy(dtype=float)
                probabilities = 1.0 / (1.0 + np.exp(-score))
                return np.column_stack([1.0 - probabilities, probabilities])

        pipeline = DirectionalPipeline()
        features = {"good_for_a": 1.0, "good_for_b": 1.0}
        feature_columns = ["good_for_a", "good_for_b"]
        full_probability = float(pipeline.predict_proba(pd.DataFrame([features], columns=feature_columns))[0, 1])
        importances = pd.DataFrame(
            [
                {"feature": "good_for_a", "importance": 1.0},
                {"feature": "good_for_b", "importance": 1.0},
            ]
        )

        factors = local_factor_table(
            features,
            importances,
            feature_columns,
            pipeline=pipeline,
            full_probability=full_probability,
        ).set_index("feature")

        self.assertGreater(factors.loc["good_for_a", "signed_contribution"], 0.0)
        self.assertEqual(factors.loc["good_for_a", "pushes_toward"], "Team A")
        self.assertLess(factors.loc["good_for_b", "signed_contribution"], 0.0)
        self.assertEqual(factors.loc["good_for_b", "pushes_toward"], "Team B")

    def test_higher_seed_a_displays_as_selected_team_even_when_model_delta_conflicts(self):
        class ConflictingSeedPipeline:
            def predict_proba(self, frame):
                score = -frame["higher_seed_A"].to_numpy(dtype=float)
                probabilities = 1.0 / (1.0 + np.exp(-score))
                return np.column_stack([1.0 - probabilities, probabilities])

        pipeline = ConflictingSeedPipeline()
        features = {"higher_seed_A": 1.0}
        feature_columns = ["higher_seed_A"]
        full_probability = float(pipeline.predict_proba(pd.DataFrame([features], columns=feature_columns))[0, 1])
        importances = pd.DataFrame([{"feature": "higher_seed_A", "importance": 0.0}])

        factors = local_factor_table(
            features,
            importances,
            feature_columns,
            pipeline=pipeline,
            full_probability=full_probability,
        )
        displayed = display_factor_table(factors, "NYK", "CLE")

        self.assertEqual(factors.loc[0, "pushes_toward"], "Team A")
        self.assertEqual(factors.loc[0, "model_delta_direction"], "Team B")
        self.assertEqual(displayed.loc[0, "favors"], "NYK")
        self.assertNotIn("model_delta_direction", displayed.columns)

    def test_main_factor_table_hides_empty_rows_and_keeps_source_data(self):
        factors = pd.DataFrame(
            [
                {
                    "feature": "OFF_RATING_DIFF",
                    "value": 1.0,
                    "importance": 0.2,
                    "signed_contribution": 0.05,
                    "model_delta_direction": "Team A",
                    "pushes_toward": "Team A",
                },
                {
                    "feature": "PACE_DIFF",
                    "value": 0.0,
                    "importance": 0.0,
                    "signed_contribution": 0.0,
                    "model_delta_direction": "Team A",
                    "pushes_toward": "Team A",
                },
            ]
        )

        displayed = display_factor_table(factors, "NYK", "CLE")

        self.assertEqual(displayed["feature"].tolist(), ["OFF_RATING_DIFF"])
        self.assertIn("PACE_DIFF", factors["feature"].tolist())

    def test_model_delta_direction_hidden_when_it_matches_pushes_toward(self):
        factors = pd.DataFrame(
            [
                {
                    "feature": "OFF_RATING_DIFF",
                    "value": 1.0,
                    "importance": 0.2,
                    "signed_contribution": 0.05,
                    "model_delta_direction": "Team A",
                    "pushes_toward": "Team A",
                }
            ]
        )

        displayed = display_factor_table(factors, "NYK", "CLE")

        self.assertNotIn("model_delta_direction", displayed.columns)

    def test_seed_difference_displays_semantic_direction_for_both_orders(self):
        importances = pd.DataFrame(
            [
                {"feature": "seed_difference", "importance": 0.0},
                {"feature": "higher_seed_A", "importance": 0.0},
            ]
        )
        positive = local_factor_table(
            {"seed_difference": 1.0, "higher_seed_A": 1.0},
            importances,
            ["seed_difference", "higher_seed_A"],
        ).set_index("feature")
        negative = local_factor_table(
            {"seed_difference": -1.0, "higher_seed_A": 0.0},
            importances,
            ["seed_difference", "higher_seed_A"],
        ).set_index("feature")

        self.assertEqual(positive.loc["seed_difference", "pushes_toward"], "Team A")
        self.assertEqual(positive.loc["higher_seed_A", "pushes_toward"], "Team A")
        self.assertEqual(negative.loc["seed_difference", "pushes_toward"], "Team B")
        self.assertEqual(negative.loc["higher_seed_A", "pushes_toward"], "Team B")

    def test_calibrated_model_local_importances_use_probability_delta(self):
        class CalibratedLikePipeline:
            def predict_proba(self, frame):
                score = (
                    frame["OFF_RATING_DIFF"].to_numpy(dtype=float) * 0.2
                    - frame["DEF_RATING_DIFF"].to_numpy(dtype=float) * 0.1
                )
                probabilities = 1.0 / (1.0 + np.exp(-score))
                return np.column_stack([1.0 - probabilities, probabilities])

        pipeline = CalibratedLikePipeline()
        features = {"OFF_RATING_DIFF": 5.0, "DEF_RATING_DIFF": 4.0}
        feature_columns = ["OFF_RATING_DIFF", "DEF_RATING_DIFF"]
        full_probability = float(pipeline.predict_proba(pd.DataFrame([features], columns=feature_columns))[0, 1])
        importances = pd.DataFrame(
            [
                {"feature": "OFF_RATING_DIFF", "importance": 0.0},
                {"feature": "DEF_RATING_DIFF", "importance": 0.0},
            ]
        )

        factors = local_factor_table(
            features,
            importances,
            feature_columns,
            pipeline=pipeline,
            full_probability=full_probability,
        )

        self.assertTrue((factors["signed_contribution"].abs() > 0).any())
        self.assertTrue((factors["local_effect"].abs() > 0).any())
        self.assertTrue(factors["global_importance"].eq(0.0).all())

    def test_current_hypothetical_excludes_series_context_from_explanations(self):
        factors = pd.DataFrame(
            [
                {"feature": "OFF_RATING_DIFF", "signed_contribution": 0.1},
                {"feature": "game_number", "signed_contribution": 0.9},
                {"feature": "series_score_diff", "signed_contribution": 0.8},
                {"feature": "elimination_game", "signed_contribution": 0.7},
            ]
        )
        importances = pd.DataFrame(
            [
                {"feature": "OFF_RATING_DIFF", "importance": 0.1},
                {"feature": "game_number", "importance": 0.9},
                {"feature": "series_score_diff", "importance": 0.8},
                {"feature": "elimination_game", "importance": 0.7},
            ]
        )

        filtered_factors, filtered_importances = filter_explanation_features_for_mode(
            factors,
            importances,
            "Current Hypothetical",
        )

        self.assertEqual(filtered_factors["feature"].tolist(), ["OFF_RATING_DIFF"])
        self.assertEqual(filtered_importances["feature"].tolist(), ["OFF_RATING_DIFF"])

    def test_playoff_explanations_exclude_series_context_from_game_factors(self):
        factors = pd.DataFrame(
            [
                {"feature": "OFF_RATING_DIFF", "signed_contribution": 0.1},
                {"feature": "game_number", "signed_contribution": 0.9},
                {"feature": "series_score_diff", "signed_contribution": 0.8},
                {"feature": "elimination_game", "signed_contribution": 0.7},
            ]
        )
        importances = pd.DataFrame(
            [
                {"feature": "OFF_RATING_DIFF", "importance": 0.1},
                {"feature": "game_number", "importance": 0.9},
                {"feature": "series_score_diff", "importance": 0.8},
                {"feature": "elimination_game", "importance": 0.7},
            ]
        )

        filtered_factors, filtered_importances = filter_explanation_features_for_mode(
            factors,
            importances,
            "Playoff Series Context",
        )

        self.assertNotIn("series_score_diff", filtered_factors["feature"].tolist())
        self.assertNotIn("series_score_diff", filtered_importances["feature"].tolist())
        self.assertNotIn("game_number", filtered_factors["feature"].tolist())
        self.assertNotIn("elimination_game", filtered_factors["feature"].tolist())
        self.assertEqual(filtered_factors["feature"].tolist(), ["OFF_RATING_DIFF"])


class CalibratedFeatureImportanceTests(unittest.TestCase):
    def test_calibrated_rf_explanation_has_multiple_global_important_rows(self):
        class FakeRandomForest:
            feature_importances_ = np.array([0.6, 0.3, 0.1])

        class FakePipeline:
            named_steps = {"classifier": FakeRandomForest()}

        class FakeCalibratedClassifier:
            estimator = FakePipeline()

        class FakeCalibratedRandomForest:
            calibrated_classifiers_ = [FakeCalibratedClassifier()]

        importances = extract_feature_importances(
            FakeCalibratedRandomForest(),
            "Random Forest",
            ["pace", "offense", "defense"],
        )

        self.assertEqual(importances["feature"].tolist(), ["pace", "offense", "defense"])
        self.assertTrue((importances["importance"] > 0).all())
        self.assertGreater(len(importances[importances["importance"] > 0]), 1)
        self.assertEqual(
            importances["importance_type"].iloc[0],
            "calibrated_underlying_feature_importances_or_abs_coef",
        )


class NoninformativeFactorFilterTests(unittest.TestCase):
    def test_filter_drops_missing_or_zero_importance_and_contribution(self):
        factors = pd.DataFrame(
            [
                {"feature": "missing_both", "importance": np.nan, "signed_contribution": np.nan},
                {"feature": "zero_both", "importance": 0.0, "signed_contribution": 0.0},
                {"feature": "important", "importance": 0.2, "signed_contribution": 0.0},
                {"feature": "contribution_only", "importance": 0.0, "signed_contribution": -0.1},
            ]
        )

        filtered = filter_noninformative_factors(factors)

        self.assertEqual(filtered["feature"].tolist(), ["important", "contribution_only"])

    def test_global_important_zero_local_rows_remain(self):
        factors = pd.DataFrame(
            [
                {
                    "feature": "important_no_local",
                    "value": 0.0,
                    "global_importance": 0.4,
                    "local_effect": 0.0,
                    "pushes_toward": "Team A",
                },
                {
                    "feature": "zero_both",
                    "value": 0.0,
                    "global_importance": 0.0,
                    "local_effect": 0.0,
                    "pushes_toward": "Team A",
                },
            ]
        )

        filtered = filter_noninformative_factors(factors)

        self.assertEqual(filtered["feature"].tolist(), ["important_no_local"])

    def test_ui_table_hides_true_zero_zero_rows(self):
        factors = pd.DataFrame(
            [
                {
                    "feature": "zero_both",
                    "value": 0.0,
                    "global_importance": 0.0,
                    "local_effect": 0.0,
                    "pushes_toward": "Team A",
                },
                {
                    "feature": "important_no_local",
                    "value": 0.0,
                    "global_importance": 0.4,
                    "local_effect": 0.0,
                    "pushes_toward": "Team B",
                },
            ]
        )

        displayed = display_factor_table(factors, "NYK", "CLE")

        self.assertEqual(displayed["feature"].tolist(), ["important_no_local"])
        self.assertEqual(
            displayed.columns.tolist(),
            ["feature", "value", "global_importance", "local_effect", "favors"],
        )

    def test_ui_table_filters_noninformative_rows(self):
        factors = pd.DataFrame(
            [
                {
                    "feature": "zero_both",
                    "value": 0.0,
                    "importance": 0.0,
                    "signed_contribution": 0.0,
                    "model_delta_direction": "Team A",
                    "pushes_toward": "Team A",
                },
                {
                    "feature": "real_factor",
                    "value": 1.0,
                    "importance": 0.3,
                    "signed_contribution": 0.0,
                    "model_delta_direction": "Team A",
                    "pushes_toward": "Team A",
                },
            ]
        )

        displayed = display_factor_table(factors, "NYK", "CLE")

        self.assertEqual(displayed["feature"].tolist(), ["real_factor"])

    def test_saved_payload_filters_noninformative_top_factors(self):
        factors = pd.DataFrame(
            [
                {"feature": "drop_me", "importance": 0.0, "signed_contribution": 0.0},
                {"feature": "keep_me", "importance": 0.0, "signed_contribution": 0.12},
            ]
        )
        importances = pd.DataFrame([{"feature": "keep_me", "importance": 0.0}])

        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir)
            output_path = processed_dir / "prediction_explanations.csv"
            with patch.object(app, "PROCESSED_DIR", processed_dir), patch.object(
                app, "PREDICTION_EXPLANATIONS_PATH", output_path
            ):
                save_prediction_explanation(
                    season="2025-26",
                    prediction_date=pd.to_datetime("2026-05-25").date(),
                    team_a_label="NYK",
                    team_b_label="CLE",
                    home_team="NYK",
                    team_a_probability=0.6,
                    features={"OFF_RATING_DIFF": 1.0},
                    factors=factors,
                    importances=importances,
                )

            saved = pd.read_csv(output_path)
            top_factors = json.loads(saved.loc[0, "top_factors"])

        self.assertEqual([row["feature"] for row in top_factors], ["keep_me"])


if __name__ == "__main__":
    unittest.main()
