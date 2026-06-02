import unittest

import pandas as pd

from src.nba_data import (
    FEATURE_COLUMNS,
    build_matchup_feature_row,
    _games_before_date,
    _head_to_head_features,
    _playoff_form_features,
    _playoff_context_features,
    _recent_form_features,
    _rest_features,
    _weighted_recent_features,
    compute_elo_snapshots,
    user_playoff_series_context_features,
)
from unittest.mock import patch


class NextStepFeatureTests(unittest.TestCase):
    def test_new_features_are_in_feature_columns(self):
        for feature in [
            "elo_diff",
            "team_A_season_elo",
            "team_B_season_elo",
            "season_elo_diff",
            "season_elo_diff_carryover_0_25",
            "season_elo_diff_carryover_0_5",
            "rest_days_A",
            "rest_days_B",
            "rest_diff",
            "is_back_to_back_A",
            "is_back_to_back_B",
            "last_3_win_pct_diff",
            "weighted_recent_point_diff",
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

    def test_elo_resets_to_1500_at_each_season_start(self):
        logs = pd.DataFrame(
            [
                {"SEASON": "2021-22", "TEAM_ID": 1, "GAME_ID": "s1g1", "GAME_DATE": "2021-10-01", "WON": 1},
                {"SEASON": "2021-22", "TEAM_ID": 2, "GAME_ID": "s1g1", "GAME_DATE": "2021-10-01", "WON": 0},
                {"SEASON": "2022-23", "TEAM_ID": 1, "GAME_ID": "s2g1", "GAME_DATE": "2022-10-01", "WON": 0},
                {"SEASON": "2022-23", "TEAM_ID": 2, "GAME_ID": "s2g1", "GAME_DATE": "2022-10-01", "WON": 1},
            ]
        )
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

        snapshots = compute_elo_snapshots(logs, carryover_weight=0.0)

        self.assertEqual(snapshots[("s1g1", 1)], 1500.0)
        self.assertEqual(snapshots[("s1g1", 2)], 1500.0)
        self.assertEqual(snapshots[("s2g1", 1)], 1500.0)
        self.assertEqual(snapshots[("s2g1", 2)], 1500.0)

    def test_offseason_carryover_weights_create_different_elo_features(self):
        logs = pd.DataFrame(
            [
                {"SEASON": "2021-22", "TEAM_ID": 1, "GAME_ID": "s1g1", "GAME_DATE": "2021-10-01", "WON": 1},
                {"SEASON": "2021-22", "TEAM_ID": 2, "GAME_ID": "s1g1", "GAME_DATE": "2021-10-01", "WON": 0},
                {"SEASON": "2022-23", "TEAM_ID": 1, "GAME_ID": "s2g1", "GAME_DATE": "2022-10-01", "WON": 0},
                {"SEASON": "2022-23", "TEAM_ID": 2, "GAME_ID": "s2g1", "GAME_DATE": "2022-10-01", "WON": 1},
            ]
        )
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

        snapshots_025 = compute_elo_snapshots(logs, carryover_weight=0.25)
        snapshots_05 = compute_elo_snapshots(logs, carryover_weight=0.5)

        diff_025 = snapshots_025[("s2g1", 1)] - snapshots_025[("s2g1", 2)]
        diff_05 = snapshots_05[("s2g1", 1)] - snapshots_05[("s2g1", 2)]
        self.assertNotEqual(diff_025, diff_05)
        self.assertGreater(abs(diff_05), abs(diff_025))

    def test_elo_uses_only_previous_games_from_same_season(self):
        logs = pd.DataFrame(
            [
                {"SEASON": "2021-22", "TEAM_ID": 1, "GAME_ID": "old", "GAME_DATE": "2021-10-01", "WON": 1},
                {"SEASON": "2021-22", "TEAM_ID": 2, "GAME_ID": "old", "GAME_DATE": "2021-10-01", "WON": 0},
                {"SEASON": "2022-23", "TEAM_ID": 1, "GAME_ID": "first", "GAME_DATE": "2022-10-01", "WON": 1},
                {"SEASON": "2022-23", "TEAM_ID": 2, "GAME_ID": "first", "GAME_DATE": "2022-10-01", "WON": 0},
                {"SEASON": "2022-23", "TEAM_ID": 1, "GAME_ID": "second", "GAME_DATE": "2022-10-03", "WON": 0},
                {"SEASON": "2022-23", "TEAM_ID": 2, "GAME_ID": "second", "GAME_DATE": "2022-10-03", "WON": 1},
            ]
        )
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

        snapshots = compute_elo_snapshots(logs, carryover_weight=0.0)

        self.assertEqual(snapshots[("first", 1)], 1500.0)
        self.assertEqual(snapshots[("first", 2)], 1500.0)
        self.assertNotEqual(snapshots[("second", 1)], 1500.0)
        self.assertNotEqual(snapshots[("second", 2)], 1500.0)

    def test_okc_prior_seasons_do_not_affect_reset_elo(self):
        okc_id = 1610612760
        logs = pd.DataFrame(
            [
                {"SEASON": "2015-16", "TEAM_ID": okc_id, "GAME_ID": "old", "GAME_DATE": "2015-10-01", "WON": 1},
                {"SEASON": "2015-16", "TEAM_ID": 2, "GAME_ID": "old", "GAME_DATE": "2015-10-01", "WON": 0},
                {"SEASON": "2024-25", "TEAM_ID": okc_id, "GAME_ID": "new", "GAME_DATE": "2024-10-01", "WON": 0},
                {"SEASON": "2024-25", "TEAM_ID": 3, "GAME_ID": "new", "GAME_DATE": "2024-10-01", "WON": 1},
            ]
        )
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

        snapshots = compute_elo_snapshots(logs, carryover_weight=0.0)

        self.assertEqual(snapshots[("new", okc_id)], 1500.0)

    def test_rest_days_and_back_to_back_flags_are_correct(self):
        team_a_games = pd.DataFrame([{"GAME_DATE": "2024-04-09"}])
        team_b_games = pd.DataFrame([{"GAME_DATE": "2024-04-07"}])
        team_a_games["GAME_DATE"] = pd.to_datetime(team_a_games["GAME_DATE"])
        team_b_games["GAME_DATE"] = pd.to_datetime(team_b_games["GAME_DATE"])

        features = _rest_features(team_a_games, team_b_games, pd.Timestamp("2024-04-10"))

        self.assertEqual(features["rest_days_A"], 1.0)
        self.assertEqual(features["rest_days_B"], 3.0)
        self.assertEqual(features["rest_diff"], -2.0)
        self.assertEqual(features["is_back_to_back_A"], 1.0)
        self.assertEqual(features["is_back_to_back_B"], 0.0)

    def test_rolling_features_exclude_current_game(self):
        logs = pd.DataFrame(
            [
                {"TEAM_ID": 1, "GAME_ID": "a_prior", "GAME_DATE": "2024-04-01", "WON": 0, "NET_RATING_GAME": -5, "POINT_DIFF": -4, "DEF_RATING_GAME": 105, "PTS": 90, "FGA": 80, "FTA": 10},
                {"TEAM_ID": 1, "GAME_ID": "a_current", "GAME_DATE": "2024-04-10", "WON": 1, "NET_RATING_GAME": 20, "POINT_DIFF": 20, "DEF_RATING_GAME": 90, "PTS": 120, "FGA": 80, "FTA": 20},
                {"TEAM_ID": 2, "GAME_ID": "b_prior", "GAME_DATE": "2024-04-01", "WON": 1, "NET_RATING_GAME": 5, "POINT_DIFF": 4, "DEF_RATING_GAME": 95, "PTS": 100, "FGA": 80, "FTA": 10},
                {"TEAM_ID": 2, "GAME_ID": "b_current", "GAME_DATE": "2024-04-10", "WON": 0, "NET_RATING_GAME": -20, "POINT_DIFF": -20, "DEF_RATING_GAME": 120, "PTS": 80, "FGA": 80, "FTA": 20},
            ]
        )
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        team_a_games = _games_before_date(logs, 1, pd.Timestamp("2024-04-10"))
        team_b_games = _games_before_date(logs, 2, pd.Timestamp("2024-04-10"))

        recent = _recent_form_features(team_a_games, team_b_games)
        weighted = _weighted_recent_features(team_a_games, team_b_games)

        self.assertEqual(recent["last_3_win_pct_diff"], -1.0)
        self.assertEqual(recent["last_5_net_rating_diff"], -10.0)
        self.assertEqual(weighted["weighted_recent_point_diff"], -8.0)

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

    def test_playoff_form_features_use_only_games_before_target(self):
        logs = _playoff_game_logs_with_future_game()
        target_date = pd.Timestamp("2025-04-25")
        team_a_prior = _games_before_date(logs, 1, target_date)
        team_b_prior = _games_before_date(logs, 2, target_date)

        features = _playoff_form_features(team_a_prior, team_b_prior)

        self.assertEqual(features["playoff_net_rating_diff"], 14.0)
        self.assertEqual(features["playoff_off_rating_diff"], 10.0)
        self.assertEqual(features["playoff_def_rating_diff"], 4.0)
        self.assertEqual(features["last_5_playoff_net_rating_diff"], 14.0)
        self.assertEqual(features["last_5_playoff_point_diff"], 14.0)

    def test_playoff_form_features_game_one_handles_no_prior_playoff_games(self):
        features = _playoff_form_features(pd.DataFrame(), pd.DataFrame())

        self.assertEqual(features["playoff_net_rating_diff"], 0.0)
        self.assertEqual(features["playoff_off_rating_diff"], 0.0)
        self.assertEqual(features["playoff_def_rating_diff"], 0.0)
        self.assertEqual(features["last_5_playoff_net_rating_diff"], 0.0)
        self.assertEqual(features["last_5_playoff_point_diff"], 0.0)

    def test_build_matchup_playoff_form_has_no_future_leakage(self):
        regular_logs = _sample_game_logs()
        playoff_logs = _playoff_game_logs_with_future_game()

        with (
            patch("src.nba_data.load_team_stats", return_value=_team_stats("Regular Season")),
            patch("src.nba_data.load_all_team_game_logs", return_value=regular_logs),
            patch("src.nba_data.load_player_stats_before_date", return_value=pd.DataFrame()),
            patch("src.nba_data.load_playoff_games", return_value=pd.DataFrame()),
            patch("src.nba_data.load_team_seeds", return_value={}),
        ):
            features = build_matchup_feature_row(
                season="2024-25",
                team_a_id=1,
                team_b_id=2,
                prediction_date="2025-04-25",
                home_team_id=1,
                game_logs=regular_logs,
                playoff_game_logs=playoff_logs,
            )

        self.assertEqual(features["playoff_net_rating_diff"], 14.0)
        self.assertEqual(features["last_5_playoff_point_diff"], 14.0)
        self.assertNotEqual(features["playoff_net_rating_diff"], -36.0)


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


def _playoff_game_logs_with_future_game() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TEAM_ID": 1,
                "GAME_ID": "prior",
                "GAME_DATE": pd.Timestamp("2025-04-20"),
                "POINT_DIFF": 7,
                "OFF_RATING_GAME": 115.0,
                "DEF_RATING_GAME": 101.0,
                "NET_RATING_GAME": 14.0,
            },
            {
                "TEAM_ID": 2,
                "GAME_ID": "prior",
                "GAME_DATE": pd.Timestamp("2025-04-20"),
                "POINT_DIFF": -7,
                "OFF_RATING_GAME": 105.0,
                "DEF_RATING_GAME": 105.0,
                "NET_RATING_GAME": 0.0,
            },
            {
                "TEAM_ID": 1,
                "GAME_ID": "future",
                "GAME_DATE": pd.Timestamp("2025-04-30"),
                "POINT_DIFF": -50,
                "OFF_RATING_GAME": 80.0,
                "DEF_RATING_GAME": 130.0,
                "NET_RATING_GAME": -50.0,
            },
            {
                "TEAM_ID": 2,
                "GAME_ID": "future",
                "GAME_DATE": pd.Timestamp("2025-04-30"),
                "POINT_DIFF": 50,
                "OFF_RATING_GAME": 130.0,
                "DEF_RATING_GAME": 80.0,
                "NET_RATING_GAME": 50.0,
            },
        ]
    )


if __name__ == "__main__":
    unittest.main()
