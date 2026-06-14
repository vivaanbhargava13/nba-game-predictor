import inspect
import unittest

import pandas as pd

import src.team_advanced_recent_deltas_experiment as experiment
from src.team_advanced_recent_deltas_experiment import (
    ADVANCED_FEATURE_COLUMNS,
    build_advanced_dataset,
    reverse_advanced_frame,
)


def _base_rows():
    return pd.DataFrame(
        [
            {
                "SEASON": "2024-25",
                "GAME_ID": "1",
                "GAME_DATE": "2025-04-20",
                "TEAM_A_ID": 1,
                "TEAM_A": "AAA",
                "TEAM_B_ID": 2,
                "TEAM_B": "BBB",
                "TEAM_A_WON": 1,
            },
            {
                "SEASON": "2024-25",
                "GAME_ID": "2",
                "GAME_DATE": "2025-04-22",
                "TEAM_A_ID": 1,
                "TEAM_A": "AAA",
                "TEAM_B_ID": 2,
                "TEAM_B": "BBB",
                "TEAM_A_WON": 0,
            },
            {
                "SEASON": "2024-25",
                "GAME_ID": "3",
                "GAME_DATE": "2025-04-24",
                "TEAM_A_ID": 1,
                "TEAM_A": "AAA",
                "TEAM_B_ID": 2,
                "TEAM_B": "BBB",
                "TEAM_A_WON": 1,
            },
            {
                "SEASON": "2024-25",
                "GAME_ID": "4",
                "GAME_DATE": "2025-04-26",
                "TEAM_A_ID": 1,
                "TEAM_A": "AAA",
                "TEAM_B_ID": 2,
                "TEAM_B": "BBB",
                "TEAM_A_WON": 1,
            },
        ]
    )


def _team_rows():
    rows = []
    for game_index in range(1, 5):
        for team_id, sign in ((1, 1), (2, -1)):
            rows.append(
                {
                    "season": "2024-25",
                    "gameDate": f"2025-04-{18 + game_index * 2:02d}",
                    "gameId": str(game_index).zfill(10),
                    "teamId": team_id,
                    "offensiveRating": 110 + sign * game_index,
                    "defensiveRating": 110 - sign * game_index,
                    "netRating": sign * game_index * 2,
                    "pace": 95 + sign * game_index,
                    "possessions": 95 + sign * game_index,
                    "trueShootingPercentage": 0.55 + sign * game_index / 100,
                    "effectiveFieldGoalPercentage": 0.52 + sign * game_index / 100,
                    "turnoverRatio": 12 - sign * game_index / 10,
                    "reboundPercentage": 0.5 + sign * game_index / 100,
                    "assistRatio": 16 + sign * game_index,
                    "PIE": 0.5 + sign * game_index / 100,
                }
            )
    return pd.DataFrame(rows)


class TeamAdvancedRecentDeltasTests(unittest.TestCase):
    def test_game_four_uses_only_games_one_through_three(self):
        features = build_advanced_dataset(_base_rows(), _team_rows())
        game_four = features[features["GAME_ID"].eq("0000000004")].iloc[0]
        used = set(
            game_four["v3_games_used_in_rolling_window"].split(",")
        )
        self.assertEqual(
            used,
            {"0000000001", "0000000002", "0000000003"},
        )
        self.assertTrue(game_four["target_game_excluded"])
        self.assertTrue(game_four["postgame_state_excluded"])
        self.assertEqual(game_four["game_number_audit"], 4)

    def test_last_one_and_series_features_are_team_a_minus_team_b(self):
        features = build_advanced_dataset(_base_rows(), _team_rows())
        game_two = features[features["GAME_ID"].eq("0000000002")].iloc[0]
        self.assertEqual(
            game_two["last_1_netRating_diff"], 4.0
        )
        self.assertEqual(
            game_two["series_to_date_netRating_diff"], 4.0
        )

    def test_target_boxscore_change_does_not_change_target_features(self):
        base = _base_rows()
        original = build_advanced_dataset(base, _team_rows())
        changed_rows = _team_rows()
        mask = changed_rows["gameId"].eq("0000000004")
        changed_rows.loc[mask, "netRating"] = [999, -999]
        changed = build_advanced_dataset(base, changed_rows)
        original_row = original[
            original["GAME_ID"].eq("0000000004")
        ][ADVANCED_FEATURE_COLUMNS].iloc[0]
        changed_row = changed[
            changed["GAME_ID"].eq("0000000004")
        ][ADVANCED_FEATURE_COLUMNS].iloc[0]
        pd.testing.assert_series_equal(original_row, changed_row)

    def test_reverse_negates_all_directional_features(self):
        frame = pd.DataFrame(
            [{column: 1.0 for column in ADVANCED_FEATURE_COLUMNS}]
        )
        reversed_frame = reverse_advanced_frame(frame)
        self.assertTrue(
            reversed_frame[ADVANCED_FEATURE_COLUMNS].eq(-1.0).all().all()
        )

    def test_module_isolated_from_streamlit_and_artifact_writes(self):
        source = inspect.getsource(experiment)
        self.assertNotIn("streamlit", source)
        self.assertNotIn("joblib.dump", source)
        self.assertNotIn("playoff_predictor.joblib\",", source)


if __name__ == "__main__":
    unittest.main()
