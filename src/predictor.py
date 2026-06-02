from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import pandas as pd

from .model import (
    DIFF_COLUMNS,
    compare_models_by_season,
    evaluate_calibrated_feature_audit,
    evaluate_home_feature_ablation,
    evaluate_feature_group_ablation,
    get_model_entry_for_mode,
    HOME_FEATURE_SET_COLUMNS,
    load_model,
    PREDICTION_MODE_CURRENT,
    PREDICTION_MODE_PLAYOFF,
    PRODUCTION_FEATURE_SET_NAME,
    PRODUCTION_HOME_FEATURE_SET_NAME,
    select_features_with_extra_trees,
    train_model,
    train_production_models,
)
from .nba_data import (
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    _processed_training_path,
    build_matchup_feature_row,
    load_team_stats,
    load_training_frame,
    resolve_team_id,
    season_range,
    user_playoff_series_context_features,
)


DEFAULT_MODEL_PATH = Path("models/playoff_predictor.joblib")
DEFAULT_CACHE_DIR = Path("data/raw")
DEFAULT_FIGURE_DIR = Path("reports/figures")


def validate_series_score(game_number: int, team_a_series_wins: int, team_b_series_wins: int) -> None:
    expected_total = int(game_number) - 1
    actual_total = int(team_a_series_wins) + int(team_b_series_wins)
    if actual_total != expected_total:
        raise ValueError(f"For Game {int(game_number)}, the series score must add up to {expected_total}.")
    if int(team_a_series_wins) >= 4 or int(team_b_series_wins) >= 4:
        raise ValueError("Neither team can have 4 wins before the current game.")


def _json_safe_feature_value(value):
    if pd.isna(value):
        return None
    return float(value)


def _predict_probability(
    team_a: str,
    team_b: str,
    season: str,
    prediction_date: str,
    home_team: str,
    cache_dir: str | Path,
    feature_season_type: str,
    model_path: str | Path,
    debug: bool = False,
    prediction_context_mode: str = "Current Hypothetical",
    game_number: int = 1,
    team_a_series_wins: int = 0,
    team_b_series_wins: int = 0,
) -> tuple[float, dict[str, float], pd.Series, pd.Series, int]:
    model_bundle = load_model(model_path)
    if prediction_context_mode == PREDICTION_MODE_PLAYOFF:
        validate_series_score(game_number, team_a_series_wins, team_b_series_wins)
    elif prediction_context_mode == PREDICTION_MODE_CURRENT:
        game_number = 1
        team_a_series_wins = 0
        team_b_series_wins = 0

    model_entry = get_model_entry_for_mode(model_bundle, prediction_context_mode)
    pipeline = model_entry["pipeline"]
    feature_columns = model_entry.get("feature_columns", DIFF_COLUMNS)
    unknown_model_features = [column for column in feature_columns if column not in FEATURE_COLUMNS]
    if unknown_model_features:
        raise ValueError(
            "Loaded model has unknown feature columns. Retrain with "
            "`python -m src.predictor train ...` before predicting. "
            f"Unknown: {unknown_model_features}"
        )

    team_a_id = resolve_team_id(team_a)
    team_b_id = resolve_team_id(team_b)
    team_stats = load_team_stats(season, cache_dir=cache_dir, season_type=feature_season_type)
    stats_by_team = team_stats.set_index("TEAM_ID")

    missing = [team_id for team_id in [team_a_id, team_b_id] if team_id not in stats_by_team.index]
    if missing:
        raise ValueError(f"Missing team stats for team ids: {missing}")

    home_team_id = team_a_id if home_team == "team1" else team_b_id
    user_series_context = None
    if prediction_context_mode == PREDICTION_MODE_PLAYOFF:
        user_series_context = user_playoff_series_context_features(
            game_number=game_number,
            team_a_series_wins=team_a_series_wins,
            team_b_series_wins=team_b_series_wins,
        )
    features = build_matchup_feature_row(
        season=season,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        prediction_date=prediction_date,
        home_team_id=home_team_id,
        cache_dir=cache_dir,
        feature_season_type=feature_season_type,
        user_series_context=user_series_context,
    )
    prediction_features = dict(features)
    if prediction_context_mode == PREDICTION_MODE_PLAYOFF:
        neutral_series_values = {
            "series_score_diff": 0.0,
            "game_number": 1.0,
            "elimination_game": 0.0,
        }
        for feature, neutral_value in neutral_series_values.items():
            if feature in feature_columns:
                prediction_features[feature] = neutral_value
    feature_frame = pd.DataFrame([prediction_features], columns=feature_columns)

    if debug:
        print("Final feature row before prediction:")
        print(
            json.dumps(
                {column: _json_safe_feature_value(prediction_features.get(column)) for column in FEATURE_COLUMNS},
                indent=2,
                sort_keys=True,
            )
        )

    probability = float(pipeline.predict_proba(feature_frame)[0, 1])
    return probability, features, stats_by_team.loc[team_a_id], stats_by_team.loc[team_b_id], home_team_id


def command_train(args: argparse.Namespace) -> None:
    seasons = season_range(args.start_season, args.end_season)
    training_frame = load_training_frame(
        seasons,
        cache_dir=args.cache_dir,
        feature_season_type=args.feature_season_type,
        processed_dir=args.processed_dir,
        force_refresh=args.force_refresh,
    )

    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    training_csv = _processed_training_path(args.processed_dir, args.feature_season_type)
    comparison_path = Path(args.processed_dir) / "model_comparison.csv"
    ablation_path = Path(args.processed_dir) / "feature_ablation.csv"
    home_ablation_path = Path(args.processed_dir) / "home_feature_ablation.csv"
    feature_importance_path = Path(args.processed_dir) / "feature_importances.csv"
    feature_selection_path = Path(args.processed_dir) / "feature_selection.csv"
    calibration_path = Path(args.processed_dir) / "model_calibration.csv"
    calibrated_feature_audit_path = Path(args.processed_dir) / "calibrated_feature_audit.csv"

    train_seasons = season_range(args.comparison_train_start, args.comparison_train_end)
    test_seasons = season_range(args.comparison_test_start, args.comparison_test_end)

    if set(train_seasons + test_seasons).issubset(set(training_frame["SEASON"].astype(str).unique())):
        feature_selection, selected_features = select_features_with_extra_trees(
            training_frame=training_frame,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
            feature_selection_path=feature_selection_path,
        )
        print("Feature selection:")
        print(feature_selection.to_string(index=False))
        print(f"Using best feature subset with {len(selected_features)} features")
        comparison = compare_models_by_season(
            training_frame=training_frame,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
            model_path=args.model_path,
            comparison_path=comparison_path,
            feature_importance_path=feature_importance_path,
            feature_columns=selected_features,
        )
        print("Model comparison:")
        print(comparison.to_string(index=False))
        ablation = evaluate_feature_group_ablation(
            training_frame=training_frame,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
            ablation_path=ablation_path,
            model_path=args.model_path,
            feature_importance_path=feature_importance_path,
            production_feature_set=PRODUCTION_FEATURE_SET_NAME,
        )
        print("Feature ablation:")
        print(ablation.to_string(index=False))
        home_ablation = evaluate_home_feature_ablation(
            training_frame=training_frame,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
            home_ablation_path=home_ablation_path,
        )
        print("Home feature ablation:")
        print(home_ablation.to_string(index=False))
        home_feature_set_name = PRODUCTION_HOME_FEATURE_SET_NAME
        home_feature_columns = HOME_FEATURE_SET_COLUMNS[home_feature_set_name]
        calibrated_feature_audit = evaluate_calibrated_feature_audit(
            training_frame=training_frame,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
            audit_path=calibrated_feature_audit_path,
            home_feature_columns=home_feature_columns,
        )
        print("Calibrated feature audit:")
        print(calibrated_feature_audit.to_string(index=False))
        artifact = train_production_models(
            training_frame=training_frame,
            train_seasons=train_seasons,
            test_seasons=test_seasons,
            model_path=args.model_path,
            feature_importance_path=feature_importance_path,
            calibration_path=calibration_path,
            home_feature_set_name=home_feature_set_name,
            home_feature_columns=home_feature_columns,
        )
        current_model = artifact["production_models"][PREDICTION_MODE_CURRENT]["metrics"]["model"]
        playoff_model = artifact["production_models"][PREDICTION_MODE_PLAYOFF]["metrics"]["model"]
        print(f"Saved current hypothetical model ({current_model}) to {args.model_path}")
        print(f"Saved playoff context model ({playoff_model}) to {args.model_path}")
        print(f"Selected home feature design: {home_feature_set_name}")
        print(f"Saved feature selection to {feature_selection_path}")
        print(f"Saved model comparison to {comparison_path}")
        print(f"Saved feature ablation to {ablation_path}")
        print(f"Saved home feature ablation to {home_ablation_path}")
        print(f"Saved calibrated feature audit to {calibrated_feature_audit_path}")
        print(f"Saved model calibration to {calibration_path}")
        print(f"Saved feature importances to {feature_importance_path}")
        print(f"Saved training data to {training_csv}")
        return

    metrics = train_model(training_frame, args.model_path)
    print(f"Training rows: {metrics['rows']}")
    print(f"Rows with missing features: {metrics['rows_with_missing_features']}")
    print(f"Total missing feature values: {metrics['missing_feature_values']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(metrics["classification_report"])
    print(f"Saved model to {args.model_path}")
    print(f"Saved training data to {training_csv}")


def command_predict(args: argparse.Namespace) -> None:
    team1_probability, features, team1_stats, team2_stats, home_team_id = _predict_probability(
        team_a=args.team1,
        team_b=args.team2,
        season=args.season,
        prediction_date=args.prediction_date,
        home_team=args.home_team,
        cache_dir=args.cache_dir,
        feature_season_type=args.feature_season_type,
        model_path=args.model_path,
        debug=True,
        prediction_context_mode=args.prediction_context_mode,
        game_number=args.game_number,
        team_a_series_wins=args.team_a_series_wins,
        team_b_series_wins=args.team_b_series_wins,
    )

    print(f"{team1_stats['TEAM_NAME']} vs {team2_stats['TEAM_NAME']} ({args.season})")
    print(f"Prediction date: {args.prediction_date}")
    print(f"Home team: {team1_stats['TEAM_ABBREVIATION'] if home_team_id == int(team1_stats.name) else team2_stats['TEAM_ABBREVIATION']}")
    print(f"{team1_stats['TEAM_ABBREVIATION']} win probability: {team1_probability:.1%}")
    print(f"{team2_stats['TEAM_ABBREVIATION']} win probability: {1 - team1_probability:.1%}")
    print(f"Missing feature count before imputation: {features.get('MISSING_FEATURE_COUNT', 0):.0f}")
    print()
    print("Feature differences:")
    for key in MODEL_FEATURE_COLUMNS:
        value = features.get(key)
        print(f"  {key}: {value:+.3f}")


def command_sanity_home_away(args: argparse.Namespace) -> None:
    cle_home_probability, cle_home_features, _, _, _ = _predict_probability(
        team_a=args.team1,
        team_b=args.team2,
        season=args.season,
        prediction_date=args.prediction_date,
        home_team="team1",
        cache_dir=args.cache_dir,
        feature_season_type=args.feature_season_type,
        model_path=args.model_path,
    )
    nyk_home_probability, nyk_home_features, _, _, _ = _predict_probability(
        team_a=args.team1,
        team_b=args.team2,
        season=args.season,
        prediction_date=args.prediction_date,
        home_team="team2",
        cache_dir=args.cache_dir,
        feature_season_type=args.feature_season_type,
        model_path=args.model_path,
    )

    print(f"{args.team1} home probability: {cle_home_probability:.6f}")
    print(f"{args.team2} home probability: {nyk_home_probability:.6f}")
    print(
        "Home feature rows:",
        {
            "team1_home": {
                key: cle_home_features[key]
                for key in ["home_team_A", "home_win_pct_diff", "away_win_pct_diff", "home_advantage_diff"]
            },
            "team2_home": {
                key: nyk_home_features[key]
                for key in ["home_team_A", "home_win_pct_diff", "away_win_pct_diff", "home_advantage_diff"]
            },
        },
    )

    if cle_home_probability == nyk_home_probability:
        raise AssertionError(
            f"Expected predict({args.team1}, {args.team2}, home={args.team1}) "
            f"!= predict({args.team1}, {args.team2}, home={args.team2})"
        )


def command_visualize(args: argparse.Namespace) -> None:
    from .visualize import save_team_stat_charts

    team_stats = load_team_stats(args.season, cache_dir=args.cache_dir, season_type=args.season_type)
    chart_paths = save_team_stat_charts(team_stats, args.output_dir)
    for path in chart_paths:
        print(f"Saved {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NBA playoff game predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a playoff matchup model")
    train.add_argument("--start-season", default="2015-16")
    train.add_argument("--end-season", default="2023-24")
    train.add_argument("--feature-season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    train.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    train.add_argument("--processed-dir", default=Path("data/processed"))
    train.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    train.add_argument("--force-refresh", action="store_true", help="Rebuild engineered features from NBA API caches/raw endpoints")
    train.add_argument("--comparison-train-start", default="2015-16")
    train.add_argument("--comparison-train-end", default="2022-23")
    train.add_argument("--comparison-test-start", default="2023-24")
    train.add_argument("--comparison-test-end", default="2024-25")
    train.set_defaults(func=command_train)

    predict = subparsers.add_parser("predict", help="Predict a two-team playoff matchup")
    predict.add_argument("--team1", required=True)
    predict.add_argument("--team2", required=True)
    predict.add_argument("--season", default="2023-24")
    predict.add_argument("--prediction-date", default=date.today().isoformat())
    predict.add_argument("--home-team", default="team1", choices=["team1", "team2"])
    predict.add_argument("--feature-season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    predict.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    predict.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    predict.add_argument("--prediction-context-mode", default=PREDICTION_MODE_CURRENT, choices=[PREDICTION_MODE_CURRENT, PREDICTION_MODE_PLAYOFF])
    predict.add_argument("--game-number", type=int, default=1, choices=range(1, 8))
    predict.add_argument("--team-a-series-wins", type=int, default=0, choices=range(0, 4))
    predict.add_argument("--team-b-series-wins", type=int, default=0, choices=range(0, 4))
    predict.set_defaults(func=command_predict)

    sanity = subparsers.add_parser("sanity-home-away", help="Check that changing home team changes prediction")
    sanity.add_argument("--team1", default="CLE")
    sanity.add_argument("--team2", default="NYK")
    sanity.add_argument("--season", default="2024-25")
    sanity.add_argument("--prediction-date", default=date.today().isoformat())
    sanity.add_argument("--feature-season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    sanity.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    sanity.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    sanity.set_defaults(func=command_sanity_home_away)

    visualize = subparsers.add_parser("visualize", help="Create seaborn charts for team metrics")
    visualize.add_argument("--season", default="2023-24")
    visualize.add_argument("--season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    visualize.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    visualize.add_argument("--output-dir", default=DEFAULT_FIGURE_DIR)
    visualize.set_defaults(func=command_visualize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
