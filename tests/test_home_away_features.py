import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.nba_data import FEATURE_COLUMNS, _home_court_features, _star_power_features
from src.predictor import _predict_probability


class HomeAwayFeatureTests(unittest.TestCase):
    def test_home_court_features_are_model_features(self):
        self.assertIn("home_team_A", FEATURE_COLUMNS)
        self.assertIn("home_win_pct_diff", FEATURE_COLUMNS)
        self.assertIn("away_win_pct_diff", FEATURE_COLUMNS)
        self.assertIn("home_advantage_diff", FEATURE_COLUMNS)
        self.assertIn("clipped_home_win_pct_diff", FEATURE_COLUMNS)

    def test_home_team_flip_changes_location_adjusted_features(self):
        team_a_games = pd.DataFrame(
            [
                {"IS_HOME": 1, "WON": 1},
                {"IS_HOME": 1, "WON": 1},
                {"IS_HOME": 0, "WON": 0},
                {"IS_HOME": 0, "WON": 0},
            ]
        )
        team_b_games = pd.DataFrame(
            [
                {"IS_HOME": 1, "WON": 1},
                {"IS_HOME": 1, "WON": 0},
                {"IS_HOME": 0, "WON": 1},
                {"IS_HOME": 0, "WON": 0},
            ]
        )

        team_a_home = _home_court_features(team_a_games, team_b_games, team_a_id=1, home_team_id=1)
        team_b_home = _home_court_features(team_a_games, team_b_games, team_a_id=1, home_team_id=2)

        self.assertEqual(team_a_home["home_team_A"], 1.0)
        self.assertEqual(team_b_home["home_team_A"], 0.0)
        self.assertEqual(team_a_home["home_win_pct_diff"], 0.5)
        self.assertEqual(team_a_home["away_win_pct_diff"], -0.5)
        self.assertEqual(team_b_home["home_win_pct_diff"], -0.5)
        self.assertEqual(team_b_home["away_win_pct_diff"], 0.5)
        self.assertEqual(team_a_home["home_advantage_diff"], 0.25)
        self.assertEqual(team_b_home["home_advantage_diff"], -0.0)
        self.assertEqual(team_a_home["clipped_home_win_pct_diff"], 0.25)
        self.assertEqual(team_b_home["clipped_home_win_pct_diff"], -0.25)

    def test_home_team_switch_increases_selected_home_team_probability(self):
        class HomeCourtOnlyPipeline:
            def predict_proba(self, frame):
                home_flag = frame["home_team_A"].to_numpy(dtype=float)
                home_advantage = frame["home_advantage_diff"].to_numpy(dtype=float)
                score = 0.2 * ((home_flag * 2) - 1) + 0.2 * home_advantage
                probabilities = 1.0 / (1.0 + np.exp(-score))
                return np.column_stack([1.0 - probabilities, probabilities])

        def feature_row(*, home_team_id, **_kwargs):
            games_a = pd.DataFrame([{"IS_HOME": 1, "WON": 1}, {"IS_HOME": 0, "WON": 0}])
            games_b = pd.DataFrame([{"IS_HOME": 1, "WON": 1}, {"IS_HOME": 0, "WON": 0}])
            return _home_court_features(games_a, games_b, team_a_id=1, home_team_id=home_team_id)

        team_stats = pd.DataFrame(
            [
                {"TEAM_ID": 1, "TEAM_NAME": "Team A", "TEAM_ABBREVIATION": "AAA"},
                {"TEAM_ID": 2, "TEAM_NAME": "Team B", "TEAM_ABBREVIATION": "BBB"},
            ]
        )
        model_bundle = {
            "pipeline": HomeCourtOnlyPipeline(),
            "feature_columns": ["home_team_A", "home_advantage_diff"],
        }

        with (
            patch("src.predictor.load_model", return_value=model_bundle),
            patch("src.predictor.resolve_team_id", side_effect=lambda value: {"AAA": 1, "BBB": 2}[value]),
            patch("src.predictor.load_team_stats", return_value=team_stats),
            patch("src.predictor.build_matchup_feature_row", side_effect=feature_row),
        ):
            team_a_home_probability, team_a_home_features, *_ = _predict_probability(
                team_a="AAA",
                team_b="BBB",
                season="2024-25",
                prediction_date="2025-04-19",
                home_team="team1",
                cache_dir=Path("data/raw"),
                feature_season_type="Regular Season",
                model_path=Path("models/playoff_predictor.joblib"),
            )
            team_b_home_probability, team_b_home_features, *_ = _predict_probability(
                team_a="AAA",
                team_b="BBB",
                season="2024-25",
                prediction_date="2025-04-19",
                home_team="team2",
                cache_dir=Path("data/raw"),
                feature_season_type="Regular Season",
                model_path=Path("models/playoff_predictor.joblib"),
            )

        self.assertEqual(team_a_home_features["home_team_A"], 1.0)
        self.assertEqual(team_b_home_features["home_team_A"], 0.0)
        self.assertGreater(team_a_home_probability, team_b_home_probability)
        self.assertGreater(1.0 - team_b_home_probability, 1.0 - team_a_home_probability)
        self.assertLess(abs(team_a_home_probability - team_b_home_probability), 0.15)

    def test_empty_player_stats_use_neutral_defaults(self):
        features = _star_power_features(pd.DataFrame(), pd.DataFrame())

        self.assertEqual(features["top_1_ppg_diff"], 0.0)
        self.assertEqual(features["top_1_mpg_diff"], 0.0)
        self.assertEqual(features["top_1_ts_pct_diff"], 0.0)
        self.assertEqual(features["top_3_ppg_diff"], 0.0)
        self.assertEqual(features["top_3_mpg_diff"], 0.0)
        self.assertEqual(features["top_3_ts_pct_diff"], 0.0)
        self.assertEqual(features["top_5_ppg_diff"], 0.0)
        self.assertEqual(features["top_5_mpg_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
