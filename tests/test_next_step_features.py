import unittest

import pandas as pd

from src.nba_data import (
    FEATURE_COLUMNS,
    build_matchup_feature_row,
    _games_before_date,
    _head_to_head_features,
    _playoff_context_features,
    compute_elo_snapshots,
    user_playoff_series_context_features,
)
from unittest.mock import patch


class NextStepFeatureTests(unittest.TestCase):
    def test_new_features_are_in_feature_columns(self):
        for feature in [
            "elo_diff",
            "efg_pct_diff",
            "ts_pct_diff",
            "turnover_pct_diff",
            "reb_pct_diff",
            "assist_turnover_ratio_diff",
            "ft_rate_diff",
            "season_h2h_win_pct_diff",
            "season_h2h_margin_diff",
            "h2h_win_pct_diff",
            "h2h_margin_diff",
            "h2h_net_rating_diff",
            "h2h_efg_pct_diff",
            "h2h_ts_pct_diff",
            "h2h_turnover_pct_diff",
            "h2h_reb_pct_diff",
            "seed_difference",
            "higher_seed_A",
            "game_number",
            "elimination_game",
            "series_score_diff",
            "weighted_recent_win_pct_diff",
            "weighted_recent_net_rating_diff",
            "weighted_recent_ts_pct_diff",
            "weighted_recent_def_rating_diff",
            "three_point_offense_vs_defense_diff",
            "paint_scoring_vs_paint_defense_diff",
            "offensive_rebound_vs_defensive_rebound_diff",
            "turnover_creation_vs_turnover_rate_diff",
            "ft_rate_vs_foul_rate_diff",
        ]:
            self.assertIn(feature, FEATURE_COLUMNS)

    def test_games_before_date_excludes_current_and_future_games(self):
        logs = pd.DataFrame(
            [
                {"TEAM_ID": 1, "GAME_ID": "prior", "GAME_DATE": "2024-04-01"},
                {"TEAM_ID": 1, "GAME_ID": "current", "GAME_DATE": "2024-04-10"},
                {"TEAM_ID": 1, "GAME_ID": "future", "GAME_DATE": "2024-04-20"},
            ]
        )
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

        result = _games_before_date(logs, team_id=1, prediction_date=pd.Timestamp("2024-04-10"))

        self.assertEqual(result["GAME_ID"].tolist(), ["prior"])

    def test_elo_snapshots_are_pre_game(self):
        logs = pd.DataFrame(
            [
                {"TEAM_ID": 1, "GAME_ID": "g1", "GAME_DATE": "2024-01-01", "WON": 1},
                {"TEAM_ID": 2, "GAME_ID": "g1", "GAME_DATE": "2024-01-01", "WON": 0},
                {"TEAM_ID": 1, "GAME_ID": "g2", "GAME_DATE": "2024-01-03", "WON": 0},
                {"TEAM_ID": 2, "GAME_ID": "g2", "GAME_DATE": "2024-01-03", "WON": 1},
            ]
        )
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

        snapshots = compute_elo_snapshots(logs)

        self.assertEqual(snapshots[("g1", 1)], 1500.0)
        self.assertEqual(snapshots[("g1", 2)], 1500.0)
        self.assertNotEqual(snapshots[("g2", 1)], 1500.0)
        self.assertNotEqual(snapshots[("g2", 2)], 1500.0)

    def test_playoff_context_uses_prior_series_games_only(self):
        playoff_games = pd.DataFrame(
            [
                {"TEAM_ID": 1, "GAME_ID": "g1", "GAME_DATE": "2024-04-01", "WL": "W"},
                {"TEAM_ID": 2, "GAME_ID": "g1", "GAME_DATE": "2024-04-01", "WL": "L"},
                {"TEAM_ID": 1, "GAME_ID": "g2", "GAME_DATE": "2024-04-03", "WL": "W"},
                {"TEAM_ID": 2, "GAME_ID": "g2", "GAME_DATE": "2024-04-03", "WL": "L"},
                {"TEAM_ID": 1, "GAME_ID": "g3", "GAME_DATE": "2024-04-05", "WL": "W"},
                {"TEAM_ID": 2, "GAME_ID": "g3", "GAME_DATE": "2024-04-05", "WL": "L"},
                {"TEAM_ID": 1, "GAME_ID": "current", "GAME_DATE": "2024-04-07", "WL": "L"},
                {"TEAM_ID": 2, "GAME_ID": "current", "GAME_DATE": "2024-04-07", "WL": "W"},
            ]
        )
        playoff_games["GAME_DATE"] = pd.to_datetime(playoff_games["GAME_DATE"])

        features = _playoff_context_features(
            team_a_id=1,
            team_b_id=2,
            prediction_date=pd.Timestamp("2024-04-07"),
            playoff_games=playoff_games,
            seeds={1: 2, 2: 3},
        )

        self.assertEqual(features["game_number"], 4.0)
        self.assertEqual(features["series_score_diff"], 3.0)
        self.assertEqual(features["elimination_game"], 1.0)
        self.assertEqual(features["seed_difference"], 1.0)
        self.assertEqual(features["higher_seed_A"], 1.0)

    def test_seed_direction_favors_better_seed(self):
        features = _playoff_context_features(
            team_a_id=1,
            team_b_id=2,
            prediction_date=pd.Timestamp("2024-04-07"),
            playoff_games=pd.DataFrame(),
            seeds={1: 6, 2: 1},
        )

        self.assertLess(features["seed_difference"], 0.0)
        self.assertEqual(features["higher_seed_A"], 0.0)

    def test_nyk_seed_3_vs_cle_seed_4_favors_team_a(self):
        features = _playoff_context_features(
            team_a_id=1,
            team_b_id=2,
            prediction_date=pd.Timestamp("2024-04-07"),
            playoff_games=pd.DataFrame(),
            seeds={1: 3, 2: 4},
        )

        self.assertEqual(features["seed_A"], 3.0)
        self.assertEqual(features["seed_B"], 4.0)
        self.assertEqual(features["seed_difference"], 1.0)
        self.assertEqual(features["higher_seed_A"], 1.0)

    def test_cle_seed_4_vs_nyk_seed_3_favors_opponent(self):
        features = _playoff_context_features(
            team_a_id=1,
            team_b_id=2,
            prediction_date=pd.Timestamp("2024-04-07"),
            playoff_games=pd.DataFrame(),
            seeds={1: 4, 2: 3},
        )

        self.assertEqual(features["seed_A"], 4.0)
        self.assertEqual(features["seed_B"], 3.0)
        self.assertEqual(features["seed_difference"], -1.0)
        self.assertEqual(features["higher_seed_A"], 0.0)

    def test_h2h_features_are_team_a_minus_team_b(self):
        team_a_games = pd.DataFrame(
            [
                {
                    "OPP_TEAM_ID": 2,
                    "WON": 1,
                    "POINT_DIFF": 10,
                    "NET_RATING_GAME": 12,
                    "FGM": 40,
                    "FGA": 80,
                    "FG3M": 10,
                    "FTA": 20,
                    "PTS": 110,
                    "TOV": 10,
                    "AST": 25,
                    "REB": 45,
                    "OPP_REB": 35,
                    "OPP_FGM": 35,
                    "OPP_FGA": 85,
                    "OPP_FG3M": 8,
                    "OPP_FTA": 15,
                    "OPP_PTS": 100,
                    "OPP_TOV": 14,
                },
                {
                    "OPP_TEAM_ID": 2,
                    "WON": 0,
                    "POINT_DIFF": -4,
                    "NET_RATING_GAME": -5,
                    "FGM": 38,
                    "FGA": 82,
                    "FG3M": 9,
                    "FTA": 18,
                    "PTS": 98,
                    "TOV": 11,
                    "AST": 22,
                    "REB": 42,
                    "OPP_REB": 40,
                    "OPP_FGM": 39,
                    "OPP_FGA": 84,
                    "OPP_FG3M": 11,
                    "OPP_FTA": 20,
                    "OPP_PTS": 102,
                    "OPP_TOV": 13,
                },
            ]
        )

        features = _head_to_head_features(team_a_games, team_b_id=2)

        self.assertEqual(features["h2h_win_pct_diff"], 0.0)
        self.assertEqual(features["h2h_margin_diff"], 6.0)
        self.assertEqual(features["season_h2h_margin_diff"], 6.0)
        self.assertAlmostEqual(features["h2h_net_rating_diff"], 7.0)

    def test_hypothetical_prediction_context_is_neutral(self):
        with patch("src.nba_data.load_team_stats", return_value=_team_stats("Regular Season")):
            features = build_matchup_feature_row(
                season="2024-25",
                team_a_id=1,
                team_b_id=2,
                prediction_date="2025-04-01",
                home_team_id=1,
                game_logs=_sample_game_logs(),
                player_stats=pd.DataFrame(),
                playoff_games=pd.DataFrame(),
                seeds={1: 4, 2: 5},
            )

        self.assertEqual(features["game_number"], 1.0)
        self.assertEqual(features["series_score_diff"], 0.0)
        self.assertEqual(features["elimination_game"], 0.0)

    def test_user_playoff_series_context_computes_context_fields(self):
        context = user_playoff_series_context_features(
            game_number=6,
            team_a_series_wins=3,
            team_b_series_wins=2,
        )

        self.assertEqual(context["game_number"], 6.0)
        self.assertEqual(context["series_score_diff"], 1.0)
        self.assertEqual(context["elimination_game"], 1.0)

    def test_stats_source_changes_feature_values(self):
        def team_stats_by_source(_season, _cache_dir, season_type):
            return _team_stats(season_type)

        with patch("src.nba_data.load_team_stats", side_effect=team_stats_by_source):
            regular = build_matchup_feature_row(
                season="2024-25",
                team_a_id=1,
                team_b_id=2,
                prediction_date="2025-04-01",
                home_team_id=1,
                game_logs=_sample_game_logs(),
                player_stats=pd.DataFrame(),
                playoff_games=pd.DataFrame(),
                seeds={1: 4, 2: 5},
                feature_season_type="Regular Season",
            )
            playoffs = build_matchup_feature_row(
                season="2024-25",
                team_a_id=1,
                team_b_id=2,
                prediction_date="2025-04-01",
                home_team_id=1,
                game_logs=_sample_game_logs(),
                player_stats=pd.DataFrame(),
                playoff_games=pd.DataFrame(),
                seeds={1: 4, 2: 5},
                feature_season_type="Playoffs",
            )

        self.assertNotEqual(regular["OFF_RATING_DIFF"], playoffs["OFF_RATING_DIFF"])


def _team_stats(season_type: str) -> pd.DataFrame:
    if season_type == "Regular Season":
        off_a, off_b = 112.0, 101.0
    else:
        off_a, off_b = 94.0, 101.0
    return pd.DataFrame(
        [
            {"TEAM_ID": 1, "TEAM_NAME": "Alpha", "TEAM_ABBREVIATION": "AAA", "OFF_RATING": off_a, "DEF_RATING": 105.0, "NET_RATING": 7.0, "W_PCT": 0.6, "PLUS_MINUS": 4.0, "PACE": 99.0},
            {"TEAM_ID": 2, "TEAM_NAME": "Beta", "TEAM_ABBREVIATION": "BBB", "OFF_RATING": off_b, "DEF_RATING": 110.0, "NET_RATING": -2.0, "W_PCT": 0.5, "PLUS_MINUS": -1.0, "PACE": 98.0},
        ]
    )


def _sample_game_logs() -> pd.DataFrame:
    rows = []
    for team_id, opp_id, won, point_diff, is_home, date in [
        (1, 2, 1, 8, 1, "2025-03-01"),
        (1, 2, 0, -2, 0, "2025-03-10"),
        (2, 1, 0, -8, 0, "2025-03-01"),
        (2, 1, 1, 2, 1, "2025-03-10"),
    ]:
        pts = 108 + point_diff
        opp_pts = 108
        rows.append(
            {
                "TEAM_ID": team_id,
                "OPP_TEAM_ID": opp_id,
                "GAME_ID": f"{date}-{team_id}",
                "GAME_DATE": pd.Timestamp(date),
                "WON": won,
                "IS_HOME": is_home,
                "POINT_DIFF": point_diff,
                "OFF_RATING_GAME": 110 + point_diff,
                "DEF_RATING_GAME": 108,
                "NET_RATING_GAME": point_diff,
                "FGM": 40,
                "FGA": 86,
                "FG3M": 11,
                "FG3A": 32,
                "FTM": 16,
                "FTA": 20,
                "PTS": pts,
                "TOV": 11,
                "AST": 24,
                "REB": 44,
                "OREB": 10,
                "DREB": 34,
                "OPP_REB": 41,
                "OPP_FGM": 39,
                "OPP_FGA": 84,
                "OPP_FG3M": 10,
                "OPP_FG3A": 30,
                "OPP_FTM": 15,
                "OPP_FTA": 19,
                "OPP_PTS": opp_pts,
                "OPP_TOV": 13,
                "OPP_OREB": 9,
                "OPP_DREB": 32,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
