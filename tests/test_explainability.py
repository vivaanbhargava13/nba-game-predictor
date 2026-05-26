import unittest

import numpy as np
import pandas as pd

from app import filter_explanation_features_for_mode, local_factor_table


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


if __name__ == "__main__":
    unittest.main()
