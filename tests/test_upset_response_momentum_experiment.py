import unittest

import numpy as np
import pandas as pd

from src.upset_response_momentum_experiment import (
    _margin_bucket,
    build_upset_response_rows,
    previous_winner_was_lower_probability_team,
)


class UpsetResponseMomentumExperimentTests(unittest.TestCase):
    def test_gate_uses_previous_winner_pregame_probability(self):
        self.assertTrue(previous_winner_was_lower_probability_team(0.49))
        self.assertFalse(previous_winner_was_lower_probability_team(0.50))
        self.assertTrue(previous_winner_was_lower_probability_team(0.44, 0.45))
        self.assertFalse(previous_winner_was_lower_probability_team(0.45, 0.45))

    def test_response_row_uses_previous_game_not_target_result(self):
        frame = pd.DataFrame(
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
                    "TEAM_A_MARGIN": 8,
                    "TEAM_A_HOME": 0,
                    "seed_A": 6,
                    "seed_B": 2,
                    "EXPECTED_WIN_PROBABILITY_A": 0.4,
                    "EXPECTED_MARGIN_A": -2,
                    "SERIES_CONTEXT_PROBABILITY_A": 0.45,
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
                    "TEAM_A_MARGIN": -20,
                    "TEAM_A_HOME": 1,
                    "seed_A": 6,
                    "seed_B": 2,
                    "EXPECTED_WIN_PROBABILITY_A": 0.55,
                    "EXPECTED_MARGIN_A": 1,
                    "SERIES_CONTEXT_PROBABILITY_A": 0.6,
                },
            ]
        )
        frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
        row = build_upset_response_rows(frame).iloc[0]
        self.assertEqual(row["previous_game_id"], "g1")
        self.assertTrue(row["previous_game_result_excluded"])
        self.assertAlmostEqual(row["previous_winner_probability"], 0.4)
        self.assertAlmostEqual(row["previous_game_residual_win"], 0.6)
        self.assertAlmostEqual(row["previous_game_residual_margin"], 10)
        self.assertEqual(row["upset_winner_won_next_game"], 0)
        self.assertEqual(row["next_game_changes_venue"], 1)

    def test_margin_buckets(self):
        self.assertEqual(_margin_bucket(5), "close_win")
        self.assertEqual(_margin_bucket(6), "normal_win")
        self.assertEqual(_margin_bucket(14), "normal_win")
        self.assertEqual(_margin_bucket(15), "blowout_win")

    def test_gate_is_stricter_at_lower_thresholds(self):
        probabilities = pd.Series([0.39, 0.42, 0.47, 0.51])
        counts = [
            int(probabilities.lt(threshold).sum())
            for threshold in (0.50, 0.45, 0.40)
        ]
        self.assertEqual(counts, [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
