import inspect
import unittest

import numpy as np
import pandas as pd

import src.player_rotation_strength_experiment as experiment
from src.player_rotation_strength_experiment import (
    ROTATION_FEATURE_COLUMNS,
    build_rotation_dataset,
    build_team_game_snapshot,
    parse_minutes,
    reverse_rotation_frame,
)


def _base_rows():
    return pd.DataFrame(
        [
            {
                "SEASON": "2024-25",
                "GAME_ID": str(game),
                "GAME_DATE": f"2025-04-{18 + game * 2:02d}",
                "TEAM_A_ID": 1,
                "TEAM_A": "AAA",
                "TEAM_B_ID": 2,
                "TEAM_B": "BBB",
                "TEAM_A_WON": int(game != 2),
            }
            for game in range(1, 5)
        ]
    )


def _player_rows():
    rows = []
    for game in range(1, 5):
        for team_id, sign in ((1, 1), (2, -1)):
            for player in range(1, 10):
                minutes = 42 - player * 2 + sign * game / 4
                rows.append(
                    {
                        "season": "2024-25",
                        "gameDate": f"2025-04-{18 + game * 2:02d}",
                        "gameId": str(game).zfill(10),
                        "teamId": team_id,
                        "personId": team_id * 100 + player,
                        "minutes": (
                            f"{int(minutes)}:"
                            f"{int((minutes % 1) * 60):02d}"
                        ),
                        "usagePercentage": 0.30 - player / 100,
                        "netRating": sign * game + player / 10,
                        "offensiveRating": 110 + sign * game,
                        "defensiveRating": 110 - sign * game,
                        "PIE": 0.1 + sign * game / 100,
                        "trueShootingPercentage": (
                            0.55 + sign * game / 100
                        ),
                    }
                )
    return pd.DataFrame(rows)


class PlayerRotationStrengthTests(unittest.TestCase):
    def test_parse_minutes(self):
        self.assertAlmostEqual(parse_minutes("37:30"), 37.5)
        self.assertEqual(parse_minutes(None), 0.0)

    def test_snapshot_rotation_shares_and_loads_are_valid(self):
        snapshot = build_team_game_snapshot(
            _player_rows()[
                (_player_rows()["gameId"].eq("0000000001"))
                & (_player_rows()["teamId"].eq(1))
            ]
        )
        self.assertGreater(snapshot["top_5_minutes_share"], 0)
        self.assertLess(snapshot["top_5_minutes_share"], 1)
        self.assertGreaterEqual(
            snapshot["top_7_minutes_share"],
            snapshot["top_5_minutes_share"],
        )
        self.assertGreater(snapshot["top_3_minutes_load"], 0)

    def test_game_four_uses_only_games_one_through_three(self):
        features = build_rotation_dataset(_base_rows(), _player_rows())
        row = features[features["GAME_ID"].eq("0000000004")].iloc[0]
        used = set(
            row["player_games_used_in_rolling_window"].split(",")
        )
        self.assertEqual(
            used,
            {"0000000001", "0000000002", "0000000003"},
        )
        self.assertTrue(row["target_game_excluded"])
        self.assertTrue(row["postgame_state_excluded"])
        self.assertEqual(row["game_number_audit"], 4)

    def test_target_player_rows_do_not_change_target_features(self):
        original = build_rotation_dataset(_base_rows(), _player_rows())
        changed_rows = _player_rows()
        mask = changed_rows["gameId"].eq("0000000004")
        changed_rows.loc[mask, "netRating"] = [999] * int(mask.sum())
        changed = build_rotation_dataset(_base_rows(), changed_rows)
        original_row = original[
            original["GAME_ID"].eq("0000000004")
        ][ROTATION_FEATURE_COLUMNS].iloc[0]
        changed_row = changed[
            changed["GAME_ID"].eq("0000000004")
        ][ROTATION_FEATURE_COLUMNS].iloc[0]
        pd.testing.assert_series_equal(original_row, changed_row)

    def test_directional_features_reverse_sign(self):
        frame = pd.DataFrame(
            [{column: 1.0 for column in ROTATION_FEATURE_COLUMNS}]
        )
        reversed_frame = reverse_rotation_frame(frame)
        self.assertTrue(
            reversed_frame[ROTATION_FEATURE_COLUMNS].eq(-1).all().all()
        )

    def test_broader_summary_counts_fold_wins(self):
        rows = []
        for fold, base_brier, rotation_brier in (
            ("a", 0.25, 0.23),
            ("b", 0.20, 0.21),
        ):
            for model, brier in (
                ("base_only", base_brier),
                ("base_plus_rotation", rotation_brier),
            ):
                rows.append(
                    {
                        "evaluation_scope": "broader_validation",
                        "validation_design": "expanding_window",
                        "fold": fold,
                        "test_seasons": "2024-25",
                        "model": model,
                        "roc_auc": 0.65,
                        "brier_score": brier,
                        "log_loss": brier + 0.4,
                        "accuracy": 0.6,
                        "f1": 0.6,
                        "expected_calibration_error": 0.1,
                        "average_probability_movement": 0.02,
                    }
                )
        summary = experiment.summarize_broader_validation(
            pd.DataFrame(rows)
        ).iloc[0]
        self.assertEqual(summary["folds"], 2)
        self.assertEqual(summary["brier_improved_folds"], 1)

    def test_module_is_offline_and_does_not_write_models(self):
        source = inspect.getsource(experiment)
        self.assertNotIn("streamlit", source)
        self.assertNotIn("joblib.dump", source)
        self.assertNotIn("train_production_models", source)


if __name__ == "__main__":
    unittest.main()
