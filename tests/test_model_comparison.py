import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.model import (
    PRODUCTION_MODEL_DEFAULTS,
    PRODUCTION_FEATURE_COLUMNS,
    PRODUCTION_FEATURE_SET_NAME,
    PRODUCTION_HOME_FEATURE_SET_NAME,
    PLAYOFF_CONTEXT_FEATURE_COLUMNS,
    PREDICTION_MODE_CURRENT,
    PREDICTION_MODE_PLAYOFF,
    SERIES_CONTEXT_FEATURES,
    compare_models_by_season,
    evaluate_calibrated_feature_audit,
    evaluate_home_feature_ablation,
    evaluate_feature_group_ablation,
    get_model_entry_for_mode,
    load_model,
    select_features_with_extra_trees,
    train_production_models,
)
from src.nba_data import FEATURE_COLUMNS


class ModelComparisonTests(unittest.TestCase):
    def test_compares_models_and_saves_outputs(self):
        rows = []
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
        for season_index, season in enumerate(seasons):
            for game_index in range(8):
                target = int((game_index + season_index) % 2 == 0)
                row = {
                    "SEASON": season,
                    "TEAM_A_WON": target,
                }
                for feature_index, feature in enumerate(FEATURE_COLUMNS):
                    row[feature] = float(target * 2 - 1 + feature_index * 0.01)
                rows.append(row)

        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            comparison_path = Path(tmpdir) / "model_comparison.csv"
            importance_path = Path(tmpdir) / "feature_importances.csv"
            selection_path = Path(tmpdir) / "feature_selection.csv"
            feature_selection, selected_features = select_features_with_extra_trees(
                frame,
                train_seasons=["2021-22", "2022-23"],
                test_seasons=["2023-24", "2024-25"],
                feature_selection_path=selection_path,
            )
            comparison = compare_models_by_season(
                frame,
                train_seasons=["2021-22", "2022-23"],
                test_seasons=["2023-24", "2024-25"],
                model_path=model_path,
                comparison_path=comparison_path,
                feature_importance_path=importance_path,
                feature_columns=selected_features,
            )

            self.assertEqual(
                set(comparison["model"]),
                {
                    "Logistic Regression",
                    "Random Forest",
                    "Gradient Boosting",
                    "Extra Trees",
                    "SVC",
                    "AdaBoost",
                    "KNN",
                },
            )
            self.assertTrue({"accuracy", "roc_auc", "precision", "recall", "f1"}.issubset(comparison.columns))
            self.assertTrue({"brier_score", "log_loss"}.issubset(comparison.columns))
            self.assertEqual(comparison["roc_auc"].tolist(), sorted(comparison["roc_auc"].tolist(), reverse=True))
            self.assertTrue(model_path.exists())
            self.assertTrue(comparison_path.exists())
            self.assertTrue(importance_path.exists())
            self.assertTrue(selection_path.exists())
            self.assertEqual(feature_selection["roc_auc"].tolist(), sorted(feature_selection["roc_auc"].tolist(), reverse=True))
            self.assertEqual(load_model(model_path)["feature_columns"], selected_features)

    def test_feature_group_ablation_saves_best_model(self):
        rows = []
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
        for season_index, season in enumerate(seasons):
            for game_index in range(10):
                target = int((game_index + season_index) % 2 == 0)
                row = {
                    "SEASON": season,
                    "TEAM_A_WON": target,
                }
                for feature_index, feature in enumerate(FEATURE_COLUMNS):
                    signal = float(target * 2 - 1)
                    row[feature] = signal + feature_index * 0.001
                rows.append(row)

        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            ablation_path = Path(tmpdir) / "feature_ablation.csv"
            importance_path = Path(tmpdir) / "feature_importances.csv"
            ablation = evaluate_feature_group_ablation(
                frame,
                train_seasons=["2021-22", "2022-23"],
                test_seasons=["2023-24", "2024-25"],
                ablation_path=ablation_path,
                model_path=model_path,
                feature_importance_path=importance_path,
            )

            expected_sets = {
                "baseline_original_features",
                "baseline_plus_corrected_signs",
                "baseline_plus_h2h",
                "baseline_plus_style_matchups",
                "baseline_plus_weighted_recent_form",
                "baseline_plus_star_features",
                "all_features",
                "top_10_by_importance",
                "top_15_by_importance",
                "top_20_by_importance",
            }
            self.assertTrue(expected_sets.issubset(set(ablation["feature_set"])))
            self.assertEqual(set(ablation["model"]), {"Random Forest", "Extra Trees"})
            self.assertTrue({"accuracy", "roc_auc", "precision", "recall", "f1", "n_features"}.issubset(ablation.columns))
            self.assertTrue({"brier_score", "log_loss"}.issubset(ablation.columns))
            self.assertTrue(ablation_path.exists())
            self.assertTrue(model_path.exists())
            self.assertTrue(importance_path.exists())
            saved = load_model(model_path)
            self.assertEqual(saved["feature_set"], PRODUCTION_FEATURE_SET_NAME)
            production_rows = ablation[ablation["feature_set"].eq(PRODUCTION_FEATURE_SET_NAME)]
            production_winner = production_rows.sort_values(["roc_auc", "accuracy", "f1"], ascending=False).iloc[0]
            self.assertEqual(saved["feature_columns"], str(production_winner["features"]).split(","))

    def test_production_feature_set_is_corrected_baseline(self):
        self.assertEqual(
            PRODUCTION_FEATURE_COLUMNS,
            [
                "OFF_RATING_DIFF",
                "DEF_RATING_DIFF",
                "NET_RATING_DIFF",
                "W_PCT_DIFF",
                "PLUS_MINUS_DIFF",
                "PACE_DIFF",
                "home_team_A",
                "clipped_home_win_pct_diff",
                "clipped_away_win_pct_diff",
                "seed_difference",
                "higher_seed_A",
            ],
        )

    def test_home_feature_ablation_saves_results(self):
        rows = []
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
        for season_index, season in enumerate(seasons):
            for game_index in range(10):
                target = int((game_index + season_index) % 2 == 0)
                row = {"SEASON": season, "TEAM_A_WON": target}
                for feature in FEATURE_COLUMNS:
                    row[feature] = float(target)
                rows.append(row)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "home_feature_ablation.csv"
            results = evaluate_home_feature_ablation(
                pd.DataFrame(rows),
                train_seasons=["2021-22", "2022-23"],
                test_seasons=["2023-24", "2024-25"],
                home_ablation_path=path,
            )

            self.assertTrue(path.exists())
            self.assertIn("home_advantage_diff", set(results["home_feature_set"]))
            self.assertIn("is_selected_production_home_design", results.columns)
            selected_rows = results[results["is_selected_production_home_design"]]
            self.assertFalse(selected_rows.empty)
            self.assertEqual(set(selected_rows["home_feature_set"]), set(selected_rows["selected_home_feature_design"]))

    def test_train_production_models_saves_two_prediction_modes(self):
        rows = []
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
        for season_index, season in enumerate(seasons):
            for game_index in range(10):
                target = int((game_index + season_index) % 2 == 0)
                row = {"SEASON": season, "TEAM_A_WON": target}
                for feature in FEATURE_COLUMNS:
                    row[feature] = float(target)
                row["series_score_diff"] = float(game_index % 3 - 1)
                row["game_number"] = float((game_index % 7) + 1)
                row["elimination_game"] = float(game_index % 2)
                rows.append(row)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            calibration_path = Path(tmpdir) / "model_calibration.csv"
            artifact = train_production_models(
                pd.DataFrame(rows),
                train_seasons=["2021-22", "2022-23"],
                test_seasons=["2023-24", "2024-25"],
                model_path=model_path,
                feature_importance_path=Path(tmpdir) / "feature_importances.csv",
                calibration_path=calibration_path,
            )

            saved = load_model(model_path)
            calibration = pd.read_csv(calibration_path)
            self.assertEqual(saved["feature_columns"], PRODUCTION_FEATURE_COLUMNS)
            self.assertIn("current_hypothetical_model", saved)
            self.assertIn("playoff_context_model", saved)
            self.assertIn("current_hypothetical_features", saved)
            self.assertIn("playoff_context_features", saved)
            self.assertTrue(calibration_path.exists())
            self.assertTrue(
                {
                    "model",
                    "calibration_method",
                    "brier_score",
                    "log_loss",
                    "expected_calibration_error",
                    "bin_index",
                    "mean_predicted_probability",
                    "observed_win_rate",
                }.issubset(calibration.columns)
            )
            self.assertTrue({"raw", "sigmoid", "isotonic"}.issubset(set(calibration["calibration_method"])))
            self.assertIn("selected_models", saved["metadata"])
            self.assertIn("calibration_method", saved["metadata"]["selected_models"][PREDICTION_MODE_CURRENT])
            self.assertIn("calibration_method", saved["metadata"]["selected_models"][PREDICTION_MODE_PLAYOFF])
            self.assertIn("calibration_method", get_model_entry_for_mode(artifact, PREDICTION_MODE_CURRENT)["metrics"])
            self.assertEqual(
                saved["metadata"]["selected_models"][PREDICTION_MODE_CURRENT]["model_type"],
                PRODUCTION_MODEL_DEFAULTS[PREDICTION_MODE_CURRENT][0],
            )
            self.assertEqual(
                saved["metadata"]["selected_models"][PREDICTION_MODE_CURRENT]["calibration_method"],
                PRODUCTION_MODEL_DEFAULTS[PREDICTION_MODE_CURRENT][1],
            )
            self.assertEqual(
                saved["metadata"]["selected_models"][PREDICTION_MODE_PLAYOFF]["model_type"],
                PRODUCTION_MODEL_DEFAULTS[PREDICTION_MODE_PLAYOFF][0],
            )
            self.assertEqual(
                saved["metadata"]["selected_models"][PREDICTION_MODE_PLAYOFF]["calibration_method"],
                PRODUCTION_MODEL_DEFAULTS[PREDICTION_MODE_PLAYOFF][1],
            )
            self.assertIn("validation_rank", saved["metadata"]["selected_models"][PREDICTION_MODE_CURRENT]["metrics"])
            self.assertIn("selected_by", saved["metadata"]["selected_models"][PREDICTION_MODE_PLAYOFF]["metrics"])
            self.assertEqual(saved["metadata"]["selected_home_feature_design"], saved["selected_home_feature_design"])
            self.assertEqual(saved["metadata"]["selected_home_feature_design"], PRODUCTION_HOME_FEATURE_SET_NAME)
            self.assertEqual(saved["home_feature_set"], PRODUCTION_HOME_FEATURE_SET_NAME)
            self.assertEqual(get_model_entry_for_mode(artifact, PREDICTION_MODE_CURRENT)["feature_columns"], PRODUCTION_FEATURE_COLUMNS)
            self.assertEqual(get_model_entry_for_mode(artifact, PREDICTION_MODE_PLAYOFF)["feature_columns"], PLAYOFF_CONTEXT_FEATURE_COLUMNS)
            self.assertEqual(PRODUCTION_HOME_FEATURE_SET_NAME, "clipped_home_split_features")
            self.assertIn("clipped_home_win_pct_diff", PRODUCTION_FEATURE_COLUMNS)
            self.assertIn("clipped_away_win_pct_diff", PRODUCTION_FEATURE_COLUMNS)
            self.assertNotIn("home_advantage_diff", PRODUCTION_FEATURE_COLUMNS)
            self.assertFalse(
                any(
                    market_feature in feature.lower()
                    for feature in PRODUCTION_FEATURE_COLUMNS
                    for market_feature in ("odds", "market", "spread", "moneyline")
                )
            )
            self.assertEqual(saved["current_hypothetical_features"], saved["playoff_context_features"])
            self.assertFalse(saved["metadata"]["series_context_used_for_game_prediction"])
            self.assertTrue(saved["metadata"]["series_context_used_for_series_probability"])
            self.assertIn("current production feature set", saved["metadata"]["game_prediction_feature_policy"])
            sample = pd.DataFrame([{feature: 0.0 for feature in saved["current_hypothetical_features"]}])
            probability = saved["current_hypothetical_model"].predict_proba(sample)[0, 1]
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
            for feature in SERIES_CONTEXT_FEATURES:
                self.assertNotIn(feature, PRODUCTION_FEATURE_COLUMNS)
                self.assertNotIn(feature, PLAYOFF_CONTEXT_FEATURE_COLUMNS)

    def test_calibrated_feature_audit_saves_required_columns(self):
        rows = []
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
        for season_index, season in enumerate(seasons):
            for game_index in range(10):
                target = int((game_index + season_index) % 2 == 0)
                row = {"SEASON": season, "TEAM_A_WON": target}
                for feature_index, feature in enumerate(FEATURE_COLUMNS):
                    row[feature] = float(target * 2 - 1 + feature_index * 0.001)
                row["game_number"] = float((game_index % 7) + 1)
                row["elimination_game"] = float(game_index % 2)
                rows.append(row)

        def tiny_random_forest():
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline

            return Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", RandomForestClassifier(n_estimators=10, random_state=42, min_samples_leaf=1)),
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "calibrated_feature_audit.csv"
            from src import model as model_module

            original_builder = model_module.CALIBRATION_MODEL_BUILDERS["Random Forest"]
            model_module.CALIBRATION_MODEL_BUILDERS["Random Forest"] = tiny_random_forest
            try:
                audit = evaluate_calibrated_feature_audit(
                    pd.DataFrame(rows),
                    train_seasons=["2021-22", "2022-23"],
                    test_seasons=["2023-24", "2024-25"],
                    audit_path=audit_path,
                )
            finally:
                model_module.CALIBRATION_MODEL_BUILDERS["Random Forest"] = original_builder

            saved = pd.read_csv(audit_path)
            focused_path = Path(tmpdir) / "elo_carryover_focused_audit.csv"
            summary_path = Path(tmpdir) / "elo_carryover_feature_summary.csv"
            correlation_path = Path(tmpdir) / "elo_carryover_feature_correlation.csv"
            self.assertTrue(audit_path.exists())
            self.assertTrue(focused_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(correlation_path.exists())

            required_columns = {
                "feature_set",
                "model",
                "calibration_method",
                "prediction_context_mode",
                "roc_auc",
                "brier_score",
                "log_loss",
                "accuracy",
                "f1",
                "expected_calibration_error",
                "audit_rank",
                "is_best_feature_set",
            }
            self.assertTrue(required_columns.issubset(saved.columns))
            self.assertTrue(required_columns.issubset(audit.columns))
            expected_sets = {
                "current_production_features",
                "prior_clipped_home_away_splits",
                "production_plus_clipped_home_away_splits",
                "production_plus_playoff_form",
                "production_plus_elo",
                "production_plus_rest",
                "production_plus_playoff_form_elo_rest",
            }
            self.assertEqual(set(saved["feature_set"]), expected_sets)
            combined_features = ",".join(saved["features"].astype(str))
            feature_text_by_set = dict(zip(saved["feature_set"], saved["features"]))
            self.assertNotIn("home_advantage_diff", feature_text_by_set["current_production_features"])
            self.assertIn("clipped_home_win_pct_diff", feature_text_by_set["current_production_features"])
            self.assertIn("clipped_away_win_pct_diff", feature_text_by_set["current_production_features"])
            self.assertNotIn("home_advantage_diff", feature_text_by_set["prior_clipped_home_away_splits"])
            self.assertIn("clipped_home_win_pct_diff", feature_text_by_set["prior_clipped_home_away_splits"])
            self.assertIn("clipped_away_win_pct_diff", feature_text_by_set["prior_clipped_home_away_splits"])
            self.assertIn("home_advantage_diff", feature_text_by_set["production_plus_clipped_home_away_splits"])
            self.assertIn("clipped_home_win_pct_diff", feature_text_by_set["production_plus_clipped_home_away_splits"])
            self.assertIn("clipped_away_win_pct_diff", feature_text_by_set["production_plus_clipped_home_away_splits"])
            self.assertIn("playoff_net_rating_diff", combined_features)
            self.assertIn("last_5_playoff_point_diff", combined_features)
            for market_feature in ("odds", "market", "spread", "moneyline"):
                self.assertNotIn(market_feature, combined_features.lower())
            self.assertTrue(saved["is_best_feature_set"].any())
            focused = pd.read_csv(focused_path)
            focused_required_columns = {
                "feature_set",
                "roc_auc",
                "brier_score",
                "log_loss",
                "expected_calibration_error",
            }
            self.assertTrue(focused_required_columns.issubset(focused.columns))
            self.assertEqual(
                set(focused["feature_set"]),
                {
                    "current_production_features",
                    "production_plus_carryover_0_25_only",
                    "production_plus_carryover_0_5_only",
                },
            )
            summary = pd.read_csv(summary_path)
            self.assertTrue(
                {
                    "season_elo_diff_carryover_0_25",
                    "season_elo_diff_carryover_0_5",
                }.issubset(set(summary["feature"]))
            )
            correlations = pd.read_csv(correlation_path)
            carryover_pair = correlations[
                correlations["feature_x"].eq("season_elo_diff_carryover_0_25")
                & correlations["feature_y"].eq("season_elo_diff_carryover_0_5")
            ]
            self.assertFalse(carryover_pair.empty)
            self.assertFalse(bool(carryover_pair.iloc[0]["identical_values"]))

    def test_top_n_feature_selection_excludes_series_context_by_default(self):
        rows = []
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
        for season_index, season in enumerate(seasons):
            for game_index in range(10):
                target = int((game_index + season_index) % 2 == 0)
                row = {"SEASON": season, "TEAM_A_WON": target}
                for feature in FEATURE_COLUMNS:
                    row[feature] = float(target)
                for feature in SERIES_CONTEXT_FEATURES:
                    row[feature] = float(target * 100)
                rows.append(row)

        with tempfile.TemporaryDirectory() as tmpdir:
            selection, _ = select_features_with_extra_trees(
                pd.DataFrame(rows),
                train_seasons=["2021-22", "2022-23"],
                test_seasons=["2023-24", "2024-25"],
                feature_selection_path=Path(tmpdir) / "feature_selection.csv",
            )

        selected_feature_text = ",".join(selection["features"].astype(str).tolist())
        for feature in SERIES_CONTEXT_FEATURES:
            self.assertNotIn(feature, selected_feature_text)


if __name__ == "__main__":
    unittest.main()
