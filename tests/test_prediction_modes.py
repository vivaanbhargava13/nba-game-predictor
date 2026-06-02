import unittest
from pathlib import Path
from typing import Optional
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
from app import (
    apply_semantic_series_factor_direction,
    build_prediction_context,
    compose_initial_explanation,
    context_system_prompt,
    deterministic_initial_explanation,
    factor_basketball_sentence,
    game_win_prediction_features,
    simulate_best_of_seven_series_probability,
    text_contradicts_series_status,
    validated_llm_response_or_fallback,
    series_status_text,
    valid_team_a_series_win_options,
)


class PredictionModeTests(unittest.TestCase):
    def test_playoff_context_game_model_uses_current_production_features(self):
        self.assertEqual(PLAYOFF_CONTEXT_FEATURE_COLUMNS, PRODUCTION_FEATURE_COLUMNS)
        self.assertNotIn("game_number", PLAYOFF_CONTEXT_FEATURE_COLUMNS)
        self.assertNotIn("elimination_game", PLAYOFF_CONTEXT_FEATURE_COLUMNS)

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

    def test_playoff_context_keeps_series_score_semantic_for_game_prediction(self):
        probabilities, captured_contexts = _run_prediction_pair(PREDICTION_MODE_PLAYOFF)

        self.assertNotEqual(captured_contexts[0]["series_score_diff"], captured_contexts[1]["series_score_diff"])
        self.assertEqual(probabilities[0], probabilities[1])

    def test_series_status_text_selected_team_leads(self):
        status, leader = series_status_text("NYK", "CLE", 3, 0)

        self.assertEqual(status, "NYK leads 3-0")
        self.assertEqual(leader, "NYK")

    def test_series_status_text_opponent_leads(self):
        status, leader = series_status_text("CLE", "NYK", 0, 3)

        self.assertEqual(status, "NYK leads 3-0")
        self.assertEqual(leader, "NYK")

    def test_series_status_text_tied(self):
        status, leader = series_status_text("SAS", "OKC", 2, 2)

        self.assertEqual(status, "Series tied 2-2")
        self.assertEqual(leader, "Tied")

    def test_chat_context_nyk_selected_leads_cle(self):
        context = _prediction_context_for_series("NYK", "CLE", 3, 0)

        self.assertEqual(context["selected_team_series_wins"], 3)
        self.assertEqual(context["opponent_series_wins"], 0)
        self.assertEqual(context["series_score_diff"], 3)
        self.assertEqual(context["series_status_text"], "NYK leads 3-0")
        self.assertEqual(context["series_leader"], "NYK")
        self.assertIn('"series_status_text": "NYK leads 3-0"', context_system_prompt(context))
        self.assertNotIn("CLE leads 3-0", context_system_prompt(context))

    def test_chat_context_nyk_opponent_leads_cle_selected(self):
        context = _prediction_context_for_series("CLE", "NYK", 0, 3)

        self.assertEqual(context["selected_team_series_wins"], 0)
        self.assertEqual(context["opponent_series_wins"], 3)
        self.assertEqual(context["series_score_diff"], -3)
        self.assertEqual(context["series_status_text"], "NYK leads 3-0")
        self.assertEqual(context["series_leader"], "NYK")

    def test_chat_context_tied_series(self):
        context = _prediction_context_for_series("SAS", "OKC", 2, 2)

        self.assertEqual(context["series_score_diff"], 0)
        self.assertEqual(context["series_status_text"], "Series tied 2-2")
        self.assertEqual(context["series_leader"], "Tied")

    def test_series_score_factor_explanation_uses_actual_status(self):
        context = _prediction_context_for_series("NYK", "CLE", 3, 0)
        row = {"feature": "series_score_diff", "value": 3.0, "pushes_toward": "Team B", "signed_contribution": -0.2}

        sentence = factor_basketball_sentence(row, context)

        self.assertIn("points toward New York Knicks (NYK)", sentence)
        self.assertIn("actual series status is NYK leads 3-0", sentence)
        self.assertNotIn("CLE leads", sentence)

    def test_series_score_diff_direction_is_semantic_not_learned(self):
        factors = pd.DataFrame(
            [
                {"feature": "series_score_diff", "value": 3.0, "pushes_toward": "Team B"},
                {"feature": "series_score_diff", "value": -3.0, "pushes_toward": "Team A"},
                {"feature": "series_score_diff", "value": 0.0, "pushes_toward": "Team A"},
            ]
        )

        adjusted = apply_semantic_series_factor_direction(factors)

        self.assertEqual(adjusted["pushes_toward"].tolist(), ["Team A", "Team B", "Tied"])

    def test_game_win_features_neutralize_series_score_diff(self):
        features = {"series_score_diff": 3.0, "game_number": 4.0, "elimination_game": 1.0}

        game_features = game_win_prediction_features(features, PREDICTION_MODE_PLAYOFF)

        self.assertEqual(game_features["series_score_diff"], 0.0)
        self.assertEqual(game_features["game_number"], 1.0)
        self.assertEqual(game_features["elimination_game"], 0.0)

    def test_nyk_leads_chat_cannot_say_cleveland_leads(self):
        context = _prediction_context_for_series("NYK", "CLE", 3, 0)

        self.assertTrue(text_contradicts_series_status("CLE leads 3-0.", context))
        self.assertTrue(text_contradicts_series_status("Cleveland leads 3-0.", context))
        self.assertTrue(text_contradicts_series_status("Cavaliers leading this series.", context))

        fallback = deterministic_initial_explanation(context)
        response = validated_llm_response_or_fallback(
            "Cleveland leads 3-0 and is holding a series lead.",
            fallback,
            context,
        )

        self.assertEqual(response, fallback)
        self.assertNotIn("CLE leads", response)
        self.assertNotIn("Cleveland leads", response)
        self.assertNotIn("Cavaliers leading", response)

    def test_cle_favored_in_game_while_nyk_leads_series(self):
        context = _prediction_context_for_series(
            "NYK",
            "CLE",
            3,
            0,
            team_a_probability=0.42,
            team_a_series_probability=0.8868,
        )

        response = deterministic_initial_explanation(context)

        self.assertIn("Series context: NYK leads 3-0.", response)
        self.assertIn("Cleveland Cavaliers (CLE)", response)
        self.assertIn("favored to win the series", response)
        self.assertIn("NYK leads 3-0", response)
        self.assertNotIn("CLE leads 3-0", response)

    def test_nyk_3_0_cle_gives_nyk_high_series_probability(self):
        probability = simulate_best_of_seven_series_probability(0.42, 3, 0)

        self.assertGreater(probability, 0.85)

    def test_game_probability_and_series_probability_are_separate_outputs(self):
        raw_game_probability = 0.42
        context = _prediction_context_for_series(
            "NYK",
            "CLE",
            3,
            0,
            team_a_probability=raw_game_probability,
            team_a_series_probability=simulate_best_of_seven_series_probability(raw_game_probability, 3, 0),
        )
        tied_series_probability = simulate_best_of_seven_series_probability(raw_game_probability, 2, 2)

        self.assertEqual(context["team_a"]["win_probability"], raw_game_probability)
        self.assertNotEqual(context["team_a"]["win_probability"], context["team_a"]["series_win_probability"])
        self.assertGreater(context["team_a"]["series_win_probability"], context["team_a"]["win_probability"])
        self.assertNotEqual(context["team_a"]["series_win_probability"], tied_series_probability)

    def test_series_tied_chat_does_not_invent_leader(self):
        context = _prediction_context_for_series("SAS", "OKC", 2, 2)

        response = compose_initial_explanation(context, "The model factors are close.")

        self.assertIn("Series context: Series tied 2-2.", response)
        self.assertFalse(text_contradicts_series_status(response, context))
        self.assertNotIn("SAS leads", response)
        self.assertNotIn("OKC leads", response)


class CurrentPipeline:
    def predict_proba(self, frame):
        score = frame["home_team_A"].to_numpy(dtype=float) * 0.1
        probabilities = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probabilities, probabilities])


class PlayoffPipeline:
    def predict_proba(self, frame):
        score = frame["home_team_A"].to_numpy(dtype=float) * 0.1
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


def _prediction_context_for_series(
    selected_abbr: str,
    opponent_abbr: str,
    selected_wins: int,
    opponent_wins: int,
    team_a_probability: float = 0.6,
    team_a_series_probability: Optional[float] = None,
) -> dict:
    names = {
        "NYK": "New York Knicks",
        "CLE": "Cleveland Cavaliers",
        "SAS": "San Antonio Spurs",
        "OKC": "Oklahoma City Thunder",
    }
    selected_name = names.get(selected_abbr, f"{selected_abbr} Team")
    opponent_name = names.get(opponent_abbr, f"{opponent_abbr} Team")
    team_a = pd.Series({"TEAM_NAME": selected_name, "TEAM_ABBREVIATION": selected_abbr})
    team_b = pd.Series({"TEAM_NAME": opponent_name, "TEAM_ABBREVIATION": opponent_abbr})
    features = {"series_score_diff": float(selected_wins - opponent_wins)}
    factors = pd.DataFrame(
        [
            {
                "feature": "series_score_diff",
                "value": float(selected_wins - opponent_wins),
                "importance": 1.0,
                "signed_contribution": 0.2,
                "pushes_toward": "Team A",
                "abs_contribution": 0.2,
            }
        ]
    )
    importances = pd.DataFrame([{"feature": "series_score_diff", "importance": 1.0}])
    return build_prediction_context(
        season="2025-26",
        season_type="Playoffs",
        prediction_date=pd.to_datetime("2026-05-25").date(),
        home_team=f"{opponent_name} ({opponent_abbr})",
        team_a=team_a,
        team_b=team_b,
        team_a_label=f"{selected_name} ({selected_abbr})",
        team_b_label=f"{opponent_name} ({opponent_abbr})",
        team_a_display=selected_abbr,
        team_b_display=opponent_abbr,
        team_a_probability=team_a_probability,
        team_a_series_probability=team_a_series_probability,
        features=features,
        factors=factors,
        importances=importances,
        model_bundle={"metrics": {"model": "Test"}},
        feature_columns=["series_score_diff"],
        prediction_context_mode=PREDICTION_MODE_PLAYOFF,
        game_number=selected_wins + opponent_wins + 1,
        team_a_series_wins=selected_wins,
        team_b_series_wins=opponent_wins,
    )


class PredictionChatContextFactorTests(unittest.TestCase):
    def test_chat_context_filters_noninformative_top_factors(self):
        team_a = pd.Series({"TEAM_NAME": "New York Knicks", "TEAM_ABBREVIATION": "NYK"})
        team_b = pd.Series({"TEAM_NAME": "Cleveland Cavaliers", "TEAM_ABBREVIATION": "CLE"})
        factors = pd.DataFrame(
            [
                {
                    "feature": "drop_me",
                    "value": 0.0,
                    "importance": 0.0,
                    "signed_contribution": 0.0,
                    "model_delta_direction": "Team A",
                    "pushes_toward": "Team A",
                },
                {
                    "feature": "keep_contribution",
                    "value": 1.0,
                    "importance": 0.0,
                    "signed_contribution": -0.08,
                    "model_delta_direction": "Team B",
                    "pushes_toward": "Team B",
                },
            ]
        )
        importances = pd.DataFrame([{"feature": "keep_contribution", "importance": 0.0}])

        context = build_prediction_context(
            season="2025-26",
            season_type="Playoffs",
            prediction_date=pd.to_datetime("2026-05-25").date(),
            home_team="New York Knicks (NYK)",
            team_a=team_a,
            team_b=team_b,
            team_a_label="New York Knicks (NYK)",
            team_b_label="Cleveland Cavaliers (CLE)",
            team_a_display="NYK",
            team_b_display="CLE",
            team_a_probability=0.6,
            features={"keep_contribution": 1.0},
            factors=factors,
            importances=importances,
            model_bundle={"metrics": {"model": "Test"}},
            feature_columns=["keep_contribution"],
            team_a_series_probability=None,
            prediction_context_mode=PREDICTION_MODE_PLAYOFF,
        )

        self.assertEqual([row["feature"] for row in context["top_factors"]], ["keep_contribution"])


if __name__ == "__main__":
    unittest.main()
