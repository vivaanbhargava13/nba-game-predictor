import unittest

import pandas as pd

from src.momentum_state_experiment import (
    MOMENTUM_FEATURE_COLUMNS,
    build_momentum_dataset,
    build_momentum_features,
    reverse_momentum_frame,
    series_win_probability,
)


class MomentumStateExperimentTests(unittest.TestCase):
    def test_series_probability_respects_score_without_fixed_boost(self):
        tied = series_win_probability(0.5, 2, 2)
        leading = series_win_probability(0.5, 3, 2)
        self.assertAlmostEqual(tied, 0.5)
        self.assertGreater(leading, tied)

    def test_momentum_uses_previous_game_residual(self):
        history = [
            {
                "game_id": "g1",
                "team_a_won": 1,
                "team_a_margin": 8.0,
                "expected_probability_a": 0.60,
                "expected_margin_a": 3.0,
                "residual_margin_a": 5.0,
                "residual_win_a": 0.40,
                "team_a_wins_before": 0,
                "team_b_wins_before": 0,
                "series_probability_before_a": 0.60,
                "series_probability_after_a": 0.78,
                "series_probability_swing_a": 0.18,
            }
        ]
        features = build_momentum_features(history)
        self.assertEqual(features["residual_margin_previous_game_diff"], 10.0)
        self.assertAlmostEqual(features["residual_win_previous_game_diff"], 0.8)
        self.assertAlmostEqual(
            features["series_win_prob_swing_previous_game_diff"], 0.36
        )

    def test_target_game_is_excluded_from_momentum_history(self):
        base = pd.DataFrame(
            [
                {
                    "SEASON": "2024-25",
                    "GAME_ID": "g1",
                    "GAME_DATE": "2025-04-20",
                    "TEAM_A_ID": 1,
                    "TEAM_A": "AAA",
                    "TEAM_B_ID": 2,
                    "TEAM_B": "BBB",
                    "TEAM_A_WON": 1,
                    "TEAM_A_MARGIN": 8.0,
                    "EXPECTED_WIN_PROBABILITY_A": 0.6,
                    "EXPECTED_MARGIN_A": 3.0,
                },
                {
                    "SEASON": "2024-25",
                    "GAME_ID": "g2",
                    "GAME_DATE": "2025-04-22",
                    "TEAM_A_ID": 1,
                    "TEAM_A": "AAA",
                    "TEAM_B_ID": 2,
                    "TEAM_B": "BBB",
                    "TEAM_A_WON": 0,
                    "TEAM_A_MARGIN": -20.0,
                    "EXPECTED_WIN_PROBABILITY_A": 0.7,
                    "EXPECTED_MARGIN_A": 5.0,
                },
            ]
        )
        base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
        rows = build_momentum_dataset(base)
        game_two = rows[rows["GAME_ID"].eq("g2")].iloc[0]
        self.assertEqual(game_two["games_used_to_compute_momentum"], "g1")
        self.assertEqual(game_two["latest_game_used"], "g1")
        self.assertTrue(game_two["target_game_excluded"])
        self.assertEqual(game_two["previous_feature_game_id"], "g1")
        self.assertEqual(game_two["residual_margin_previous_game_diff"], 10.0)

    def test_reverse_negates_all_directional_features(self):
        frame = pd.DataFrame(
            [{column: float(index + 1) for index, column in enumerate(MOMENTUM_FEATURE_COLUMNS)}]
        )
        reversed_frame = reverse_momentum_frame(frame)
        for column in MOMENTUM_FEATURE_COLUMNS:
            self.assertEqual(reversed_frame.iloc[0][column], -frame.iloc[0][column])


if __name__ == "__main__":
    unittest.main()
