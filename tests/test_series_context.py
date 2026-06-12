import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.series_context import (
    BLEND_FEATURE_COLUMNS,
    DEFAULT_SERIES_CONTEXT_MODEL_PATH,
    SERIES_CONTEXT_FEATURE_COLUMNS,
    apply_series_context_probability,
    blended_probability,
    build_series_context_dataset,
    build_runtime_series_context,
    build_series_state_features,
    load_series_context_model,
    predict_series_context_probability,
    resolve_completed_game_pregame_state_for_card,
    resolve_pregame_series_state_for_card,
    reverse_series_context_frame,
)


class SeriesContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_series_context_model(DEFAULT_SERIES_CONTEXT_MODEL_PATH)

    def test_historical_dataset_uses_pregame_series_state(self):
        games = pd.DataFrame(
            [
                _game_row("1", "2024-04-20", 1, "AAA", "vs. BBB", "W", 110),
                _game_row("1", "2024-04-20", 2, "BBB", "@ AAA", "L", 100),
                _game_row("2", "2024-04-22", 1, "AAA", "vs. BBB", "L", 101),
                _game_row("2", "2024-04-22", 2, "BBB", "@ AAA", "W", 105),
            ]
        )

        with patch("src.series_context.load_playoff_games", return_value=games):
            frame = build_series_context_dataset(["2023-24"])

        first = frame.iloc[0]
        second = frame.iloc[1]
        self.assertEqual(first["game_number"], 1)
        self.assertEqual(first["team_a_series_wins"], 0)
        self.assertEqual(first["team_b_series_wins"], 0)
        self.assertEqual(first["previous_game_winner"], 0)
        self.assertEqual(second["game_number"], 2)
        self.assertEqual(second["team_a_series_wins"], 1)
        self.assertEqual(second["team_b_series_wins"], 0)
        self.assertEqual(second["previous_game_winner"], 1)
        self.assertEqual(second["previous_game_margin"], 10)

    def test_series_context_probability_changes_with_series_score(self):
        down = _state(1, 3)
        up = _state(3, 1)

        down_probability = predict_series_context_probability(self.artifact, down)
        up_probability = predict_series_context_probability(self.artifact, up)

        self.assertNotEqual(down_probability, up_probability)
        self.assertGreater(up_probability, down_probability)

    def test_down_three_one_and_up_three_one_use_learned_artifact(self):
        down_context = predict_series_context_probability(self.artifact, _state(1, 3))
        up_context = predict_series_context_probability(self.artifact, _state(3, 1))
        down_blended = blended_probability(self.artifact["blend_model"], np.array([0.6]), np.array([down_context]))[0]
        up_blended = blended_probability(self.artifact["blend_model"], np.array([0.6]), np.array([up_context]))[0]

        self.assertLess(down_blended, up_blended)
        self.assertIn("context_model", self.artifact)
        self.assertIn("blend_model", self.artifact)

    def test_blending_weights_are_loaded_from_artifact(self):
        self.assertEqual(self.artifact["blend_feature_columns"], BLEND_FEATURE_COLUMNS)
        loaded = np.array([self.artifact["blend_coefficients"][name] for name in BLEND_FEATURE_COLUMNS])

        np.testing.assert_allclose(loaded, self.artifact["blend_model"].coef_[0])

    def test_missing_series_context_artifact_falls_back_to_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = apply_series_context_probability(
                base_probability=0.63,
                season="2025-26",
                team_a_id=1,
                team_b_id=2,
                prediction_date="2026-05-25",
                home_team_id=2,
                game_number=5,
                team_a_series_wins=1,
                team_b_series_wins=3,
                model_path=Path(tmpdir) / "missing.joblib",
            )

        self.assertFalse(result["applied"])
        self.assertEqual(result["probability"], 0.63)
        self.assertIn("using base game probability", result["note"])

    def test_context_prediction_is_order_symmetric(self):
        forward = pd.DataFrame([_state(3, 1)], columns=SERIES_CONTEXT_FEATURE_COLUMNS)
        reverse = reverse_series_context_frame(forward)
        direct = self.artifact["context_model"].predict_proba(forward)[0, 1]
        reverse_probability = self.artifact["context_model"].predict_proba(reverse)[0, 1]
        symmetric = (direct + 1 - reverse_probability) / 2
        reversed_symmetric = (reverse_probability + 1 - direct) / 2

        self.assertAlmostEqual(symmetric, 1 - reversed_symmetric)

    def test_saved_metrics_cover_base_context_and_blended(self):
        metrics = pd.read_csv("data/processed/series_context_model_comparison.csv")

        self.assertEqual(set(metrics["model"]), {"base_only", "series_context_only", "blended"})
        self.assertTrue({"roc_auc", "brier_score", "log_loss", "accuracy", "f1"}.issubset(metrics.columns))

    def test_three_one_series_resolves_games_five_six_and_seven(self):
        expected = {
            5: (3, 1, [], False, "NYK leads 3-1"),
            6: (2, 3, ["SAS"], True, "NYK leads 3-2"),
            7: (3, 3, ["SAS", "SAS"], True, "Series tied 3-3"),
        }
        matchups = {
            5: ("NYK", "SAS"),
            6: ("SAS", "NYK"),
            7: ("NYK", "SAS"),
        }

        for game_number, (away, home) in matchups.items():
            with self.subTest(game_number=game_number):
                result = resolve_pregame_series_state_for_card(
                    current_series_wins={"NYK": 3, "SAS": 1},
                    current_completed_games=[],
                    requested_game_number=game_number,
                    requested_away_team=away,
                    requested_home_team=home,
                    if_necessary=game_number > 5,
                    series_teams=("NYK", "SAS"),
                )
                away_wins, home_wins, assumed, conditional, status = expected[game_number]
                self.assertEqual(result["resolved_game_number"], game_number)
                self.assertEqual(result["resolved_team_a_series_wins"], away_wins)
                self.assertEqual(result["resolved_team_b_series_wins"], home_wins)
                self.assertEqual(result["assumed_prior_winners"], assumed)
                self.assertEqual(result["conditional_context"], conditional)
                self.assertEqual(result["resolved_series_score_text"], status)
                self.assertTrue(result["context_available"])

    def test_assumed_future_result_never_invents_previous_margin(self):
        result = resolve_pregame_series_state_for_card(
            current_series_wins={"NYK": 3, "SAS": 1},
            current_completed_games=[],
            requested_game_number=6,
            requested_away_team="SAS",
            requested_home_team="NYK",
            if_necessary=True,
            series_teams=("NYK", "SAS"),
        )

        self.assertEqual(result["previous_game_winner"], "SAS")
        self.assertIsNone(result["previous_game_margin"])

    def test_only_conditional_context_hides_historical_previous_margin(self):
        games = pd.DataFrame(
            [
                _game_row("1", "2026-06-12", 1, "NYK", "vs. SAS", "W", 110),
                _game_row("1", "2026-06-12", 2, "SAS", "@ NYK", "L", 100),
            ]
        )
        common = {
            "season": "2025-26",
            "team_a_id": 1,
            "team_b_id": 2,
            "prediction_date": "2026-06-14",
            "home_team_id": 2,
            "game_number": 2,
            "team_a_series_wins": 1,
            "team_b_series_wins": 0,
            "team_a_abbr": "NYK",
            "team_b_abbr": "SAS",
        }

        with patch("src.series_context.load_playoff_games", return_value=games):
            actual = build_runtime_series_context(
                **common,
                resolved_context={"conditional_context": False, "previous_game_margin": None},
            )
            conditional = build_runtime_series_context(
                **common,
                resolved_context={
                    "conditional_context": True,
                    "previous_game_winner": "SAS",
                    "previous_game_margin": None,
                    "assumed_prior_winners": ["SAS"],
                },
            )

        self.assertEqual(actual["previous_game_margin"], 10)
        self.assertTrue(np.isnan(conditional["previous_game_margin"]))

    def test_nondeterministic_future_context_is_not_fabricated(self):
        result = resolve_pregame_series_state_for_card(
            current_series_wins={"NYK": 2, "SAS": 2},
            current_completed_games=[],
            requested_game_number=6,
            requested_away_team="SAS",
            requested_home_team="NYK",
            if_necessary=True,
            series_teams=("NYK", "SAS"),
        )

        self.assertFalse(result["context_available"])
        self.assertFalse(result["conditional_context"])
        self.assertEqual(result["context_note"], "Playoff context pending prior result.")

    def test_completed_game_four_reconstructs_pregame_state_and_previous_game(self):
        result = resolve_completed_game_pregame_state_for_card(
            postgame_series_wins={"SAS": 1, "NYK": 3},
            completed_games=[
                {
                    "game_id": "game-3",
                    "game_number": 3,
                    "away_abbr": "NYK",
                    "home_abbr": "SAS",
                    "away_score": 101,
                    "home_score": 106,
                },
                {
                    "game_id": "game-4",
                    "game_number": 4,
                    "away_abbr": "SAS",
                    "home_abbr": "NYK",
                    "away_score": 106,
                    "home_score": 107,
                },
            ],
            requested_game_number=4,
            requested_away_team="SAS",
            requested_home_team="NYK",
            game_id="game-4",
            away_score=106,
            home_score=107,
        )

        self.assertEqual(result["resolved_game_number"], 4)
        self.assertEqual(result["resolved_team_a_series_wins"], 1)
        self.assertEqual(result["resolved_team_b_series_wins"], 2)
        self.assertEqual(result["resolved_series_score_text"], "NYK leads 2-1")
        self.assertEqual(result["postgame_series_state"], "NYK leads 3-1")
        self.assertEqual(result["previous_game_winner"], "SAS")
        self.assertEqual(result["previous_game_margin"], 5)
        self.assertNotEqual(result["previous_game_winner"], result["loaded_game_winner"])


def _state(team_a_wins: int, team_b_wins: int) -> dict[str, float]:
    return build_series_state_features(
        game_number=team_a_wins + team_b_wins + 1,
        team_a_series_wins=team_a_wins,
        team_b_series_wins=team_b_wins,
        team_a_home=True,
        prior_games=[],
    )


def _game_row(game_id, game_date, team_id, abbreviation, matchup, result, points):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": game_date,
        "TEAM_ID": team_id,
        "TEAM_ABBREVIATION": abbreviation,
        "MATCHUP": matchup,
        "WL": result,
        "PTS": points,
    }


if __name__ == "__main__":
    unittest.main()
