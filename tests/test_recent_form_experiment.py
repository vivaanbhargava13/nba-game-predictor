import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.recent_form_experiment import (
    TEAM_RECENT_FEATURE_COLUMNS,
    _calibration_rows,
    build_recent_form_row,
    reverse_recent_form_frame,
)


def _team_row(team_id, abbr, won, home, point_diff):
    return pd.Series(
        {
            "TEAM_ID": team_id,
            "TEAM_ABBREVIATION": abbr,
            "WON": won,
            "IS_HOME": home,
            "POINT_DIFF": point_diff,
        }
    )


class RecentFormExperimentTests(unittest.TestCase):
    def test_recent_form_excludes_current_game(self):
        target_date = pd.Timestamp("2025-04-20")
        games = pd.DataFrame(
            [
                {
                    "TEAM_ID": 1,
                    "GAME_ID": "prior-a",
                    "GAME_DATE": "2025-04-18",
                    "SEASON_TYPE": "Regular Season",
                    "WON": 1,
                    "POINT_DIFF": 10,
                    "NET_RATING_GAME": 10,
                    "OFF_RATING_GAME": 115,
                    "DEF_RATING_GAME": 105,
                    "PACE_GAME": 98,
                },
                {
                    "TEAM_ID": 2,
                    "GAME_ID": "prior-b",
                    "GAME_DATE": "2025-04-18",
                    "SEASON_TYPE": "Regular Season",
                    "WON": 0,
                    "POINT_DIFF": -10,
                    "NET_RATING_GAME": -10,
                    "OFF_RATING_GAME": 105,
                    "DEF_RATING_GAME": 115,
                    "PACE_GAME": 98,
                },
                {
                    "TEAM_ID": 1,
                    "GAME_ID": "target",
                    "GAME_DATE": target_date,
                    "SEASON_TYPE": "Playoffs",
                    "WON": 0,
                    "POINT_DIFF": -50,
                    "NET_RATING_GAME": -50,
                    "OFF_RATING_GAME": 80,
                    "DEF_RATING_GAME": 130,
                    "PACE_GAME": 90,
                },
                {
                    "TEAM_ID": 2,
                    "GAME_ID": "target",
                    "GAME_DATE": target_date,
                    "SEASON_TYPE": "Playoffs",
                    "WON": 1,
                    "POINT_DIFF": 50,
                    "NET_RATING_GAME": 50,
                    "OFF_RATING_GAME": 130,
                    "DEF_RATING_GAME": 80,
                    "PACE_GAME": 90,
                },
            ]
        )
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
        row = build_recent_form_row(
            season="2024-25",
            game_id="target",
            game_date=target_date,
            team_a=_team_row(1, "AAA", 0, 1, -50),
            team_b=_team_row(2, "BBB", 1, 0, 50),
            season_games=games,
            prior_series_games=[],
            team_a_series_wins=0,
            team_b_series_wins=0,
        )
        self.assertEqual(row["last_3_win_pct_diff"], 1.0)
        self.assertEqual(row["last_3_point_margin_diff"], 20.0)
        self.assertEqual(row["cumulative_series_margin_before_game"], 0.0)

    def test_series_features_use_prior_games_only(self):
        games = pd.DataFrame(
            [
                {
                    "TEAM_ID": team,
                    "GAME_ID": f"regular-{team}",
                    "GAME_DATE": "2025-04-10",
                    "SEASON_TYPE": "Regular Season",
                    "WON": int(team == 1),
                    "POINT_DIFF": 5 if team == 1 else -5,
                    "NET_RATING_GAME": 5 if team == 1 else -5,
                    "OFF_RATING_GAME": 110,
                    "DEF_RATING_GAME": 105,
                    "PACE_GAME": 97,
                }
                for team in (1, 2)
            ]
        )
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
        row = build_recent_form_row(
            season="2024-25",
            game_id="game-3",
            game_date=pd.Timestamp("2025-04-25"),
            team_a=_team_row(1, "AAA", 1, 1, 99),
            team_b=_team_row(2, "BBB", 0, 0, -99),
            season_games=games,
            prior_series_games=[
                {"team_a_home": 1, "team_a_won": 1, "team_a_margin": 8},
                {"team_a_home": 0, "team_a_won": 0, "team_a_margin": -3},
            ],
            team_a_series_wins=1,
            team_b_series_wins=1,
        )
        self.assertEqual(row["game_number"], 3.0)
        self.assertEqual(row["cumulative_series_margin_before_game"], 5.0)
        self.assertEqual(row["average_series_margin_before_game"], 2.5)
        self.assertEqual(row["previous_game_margin"], -3.0)

    def test_reverse_recent_features_negates_differences(self):
        values = {column: 1.0 for column in TEAM_RECENT_FEATURE_COLUMNS}
        values.update(
            {
                "game_number": 4.0,
                "series_score_diff": 1.0,
                "team_a_series_wins": 2.0,
                "team_b_series_wins": 1.0,
                "team_a_elimination_game": 0.0,
                "team_b_elimination_game": 0.0,
                "team_a_home": 1.0,
                "team_a_trailing_series": 0.0,
                "team_a_leading_series": 1.0,
                "previous_game_winner": 1.0,
                "previous_game_margin": 7.0,
                "team_a_recent_series_wins": 2.0,
                "team_b_recent_series_wins": 1.0,
                "team_a_road_wins_in_series": 1.0,
                "team_b_road_wins_in_series": 0.0,
                "team_a_home_losses_in_series": 0.0,
                "team_b_home_losses_in_series": 1.0,
                "team_a_closeout_game": 0.0,
                "team_b_closeout_game": 0.0,
                "cumulative_series_margin_before_game": 12.0,
                "average_series_margin_before_game": 4.0,
                "rest_days_diff": 1.0,
            }
        )
        reversed_row = reverse_recent_form_frame(pd.DataFrame([values])).iloc[0]
        self.assertEqual(reversed_row["last_3_win_pct_diff"], -1.0)
        self.assertEqual(reversed_row["series_score_diff"], -1.0)
        self.assertEqual(reversed_row["team_a_home"], 0.0)
        self.assertEqual(reversed_row["cumulative_series_margin_before_game"], -12.0)

    def test_calibration_rows_cover_all_predictions(self):
        rows = _calibration_rows(
            "test",
            pd.Series([0, 0, 1, 1]),
            pd.Series([0.1, 0.4, 0.6, 0.9]).to_numpy(),
        )
        self.assertEqual(sum(row["count"] for row in rows), 4)
        self.assertTrue(
            {"mean_predicted_probability", "observed_win_rate"}
            <= set(rows[0])
        )


if __name__ == "__main__":
    unittest.main()
