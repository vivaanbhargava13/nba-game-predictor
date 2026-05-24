import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.model import (
    PLAYOFF_CONTEXT_FEATURE_COLUMNS,
    PREDICTION_MODE_CURRENT,
    PREDICTION_MODE_PLAYOFF,
    PRODUCTION_FEATURE_COLUMNS,
)
from src.predictor import _predict_probability, validate_series_score
from app import valid_team_a_series_win_options


class PredictionModeTests(unittest.TestCase):
    def test_invalid_series_score_blocks_prediction(self):
        with self.assertRaisesRegex(ValueError, "For Game 7, the series score must add up to 6."):
            validate_series_score(game_number=7, team_a_series_wins=3, team_b_series_wins=2)

    def test_game_7_requires_total_series_wins_of_6(self):
        validate_series_score(game_number=7, team_a_series_wins=3, team_b_series_wins=3)
        with self.assertRaises(ValueError):
            validate_series_score(game_number=7, team_a_series_wins=4, team_b_series_wins=2)

    def test_series_score_options_auto_balance_without_four_wins(self):
        self.assertEqual(valid_team_a_series_win_options(1), [0])
        self.assertEqual(valid_team_a_series_win_options(5), [1, 2, 3])
        self.assertEqual(valid_team_a_series_win_options(7), [3])

    def test_current_hypothetical_ignores_series_features(self):
        probabilities, captured_contexts = _run_prediction_pair(PREDICTION_MODE_CURRENT)

        self.assertEqual(captured_contexts, [None, None])
        self.assertEqual(probabilities[0], probabilities[1])

    def test_playoff_context_includes_series_features_and_changes_row(self):
        probabilities, captured_contexts = _run_prediction_pair(PREDICTION_MODE_PLAYOFF)

        self.assertNotEqual(captured_contexts[0]["series_score_diff"], captured_contexts[1]["series_score_diff"])
        self.assertNotEqual(probabilities[0], probabilities[1])


class CurrentPipeline:
    def predict_proba(self, frame):
        score = frame["home_team_A"].to_numpy(dtype=float) * 0.1
        probabilities = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probabilities, probabilities])


class PlayoffPipeline:
    def predict_proba(self, frame):
        score = frame["series_score_diff"].to_numpy(dtype=float) * 0.25
        probabilities = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probabilities, probabilities])


def _run_prediction_pair(mode: str):
    team_stats = pd.DataFrame(
        [
            {"TEAM_ID": 1, "TEAM_NAME": "Team A", "TEAM_ABBREVIATION": "AAA"},
            {"TEAM_ID": 2, "TEAM_NAME": "Team B", "TEAM_ABBREVIATION": "BBB"},
        ]
    )
    model_bundle = {
        "pipeline": CurrentPipeline(),
        "feature_columns": PRODUCTION_FEATURE_COLUMNS,
        "production_models": {
            PREDICTION_MODE_CURRENT: {
                "pipeline": CurrentPipeline(),
                "feature_columns": PRODUCTION_FEATURE_COLUMNS,
                "metrics": {"model": "Current"},
            },
            PREDICTION_MODE_PLAYOFF: {
                "pipeline": PlayoffPipeline(),
                "feature_columns": PLAYOFF_CONTEXT_FEATURE_COLUMNS,
                "metrics": {"model": "Playoff"},
            },
        },
    }
    captured_contexts = []

    def feature_row(*, user_series_context=None, home_team_id, **_kwargs):
        captured_contexts.append(user_series_context)
        row = {feature: 0.0 for feature in PLAYOFF_CONTEXT_FEATURE_COLUMNS}
        row["home_team_A"] = 1.0 if home_team_id == 1 else 0.0
        row["game_number"] = 1.0
        row["series_score_diff"] = 0.0
        row["elimination_game"] = 0.0
        if user_series_context:
            row.update(user_series_context)
        return row

    with (
        patch("src.predictor.load_model", return_value=model_bundle),
        patch("src.predictor.resolve_team_id", side_effect=lambda value: {"AAA": 1, "BBB": 2}[value]),
        patch("src.predictor.load_team_stats", return_value=team_stats),
        patch("src.predictor.build_matchup_feature_row", side_effect=feature_row),
    ):
        probability_a, *_ = _predict_probability(
            team_a="AAA",
            team_b="BBB",
            season="2024-25",
            prediction_date="2025-04-19",
            home_team="team1",
            cache_dir=Path("data/raw"),
            feature_season_type="Regular Season",
            model_path=Path("models/playoff_predictor.joblib"),
            prediction_context_mode=mode,
            game_number=6,
            team_a_series_wins=3,
            team_b_series_wins=2,
        )
        probability_b, *_ = _predict_probability(
            team_a="AAA",
            team_b="BBB",
            season="2024-25",
            prediction_date="2025-04-19",
            home_team="team1",
            cache_dir=Path("data/raw"),
            feature_season_type="Regular Season",
            model_path=Path("models/playoff_predictor.joblib"),
            prediction_context_mode=mode,
            game_number=6,
            team_a_series_wins=2,
            team_b_series_wins=3,
        )

    return (probability_a, probability_b), captured_contexts


if __name__ == "__main__":
    unittest.main()
