import unittest

import pandas as pd

from src.momentum_subgroup_audit import (
    MODEL_COLUMNS,
    audit_subgroups,
    orient_predictions,
    subgroup_masks,
)


def _prediction_row():
    return {
        "SEASON": "2024-25",
        "GAME_ID": "g6",
        "GAME_DATE": "2025-05-20",
        "TEAM_A": "AAA",
        "TEAM_B": "BBB",
        "TEAM_A_WON": 1,
        "seed_A": 6,
        "seed_B": 2,
        "game_number_audit": 6,
        "team_a_series_wins_audit": 2,
        "team_b_series_wins_audit": 3,
        "team_a_home_audit": 1,
        "previous_game_winner_audit": 1,
        "team_a_won_previous_two_audit": 1,
        "team_b_won_previous_two_audit": 0,
        "previous_margin_audit": 12,
        "previous_expected_probability_a_audit": 0.4,
        "previous_residual_margin_a_audit": 8,
        "team_a_last_two_positive_residuals_audit": 1,
        "team_b_last_two_positive_residuals_audit": 0,
        "team_a_momentum_state_audit": 5,
        "team_b_momentum_state_audit": -5,
        "home_court_swing_audit": 1,
        "base_probability": 0.4,
        "base_plus_series_probability": 0.42,
        "base_plus_momentum_probability": 0.55,
        "base_plus_series_plus_momentum_probability": 0.58,
    }


class MomentumSubgroupAuditTests(unittest.TestCase):
    def test_orientation_creates_both_team_views(self):
        oriented = orient_predictions(pd.DataFrame([_prediction_row()]))
        self.assertEqual(len(oriented), 2)
        self.assertEqual(oriented.iloc[0]["focal_team"], "AAA")
        self.assertEqual(oriented.iloc[1]["focal_team"], "BBB")
        self.assertAlmostEqual(oriented.iloc[1]["base_only"], 0.6)

    def test_requested_subgroups_use_pregame_state(self):
        oriented = orient_predictions(pd.DataFrame([_prediction_row()]))
        masks = subgroup_masks(oriented)
        team_a = oriented["focal_team"].eq("AAA")
        self.assertTrue((masks["game_6_or_7"] & team_a).any())
        self.assertTrue((masks["elimination_game"] & team_a).any())
        self.assertTrue((masks["team_down_3_1_or_3_2"] & team_a).any())
        self.assertTrue(
            (masks["won_previous_as_lower_base_probability_team"] & team_a).any()
        )
        self.assertTrue((masks["lower_seed_with_positive_momentum"] & team_a).any())

    def test_audit_contains_all_models_and_metrics(self):
        rows = []
        for index in range(40):
            row = _prediction_row()
            row["GAME_ID"] = f"g{index}"
            row["TEAM_A_WON"] = index % 2
            rows.append(row)
        audit, examples, recommendation = audit_subgroups(pd.DataFrame(rows))
        game_six = audit[audit["subgroup"].eq("game_6_or_7")]
        self.assertEqual(set(game_six["model"]), set(MODEL_COLUMNS))
        self.assertTrue(
            {
                "brier_score",
                "log_loss",
                "roc_auc",
                "n_games",
                "average_probability_movement_vs_base",
            }
            <= set(audit.columns)
        )
        self.assertFalse(examples.empty)
        self.assertRegex(recommendation, r"^[1-4]\.")


if __name__ == "__main__":
    unittest.main()
