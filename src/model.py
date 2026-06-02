from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .nba_data import (
    FEATURE_COLUMNS,
    PLAYOFF_FORM_AUDIT_FEATURES,
    TIER_1_FEATURES,
    TIER_3_FEATURES,
    TIER_4_FEATURES,
    TIER_8_FEATURES,
    TIER_10_FEATURES,
)


DIFF_COLUMNS = FEATURE_COLUMNS
FEATURE_SELECTION_SIZES = [5, 10, 15, 20, 25, "all"]
WEIGHTED_RECENT_FEATURES = [
    "weighted_recent_win_pct_diff",
    "weighted_recent_net_rating_diff",
    "weighted_recent_ts_pct_diff",
    "weighted_recent_def_rating_diff",
]
SEASON_RESET_ELO_FEATURES = ["team_A_season_elo", "team_B_season_elo", "season_elo_diff"]
OFFSEASON_REGRESSED_ELO_FEATURES = {
    0.25: ["season_elo_diff_carryover_0_25"],
    0.5: ["season_elo_diff_carryover_0_5"],
}
OFFSEASON_REGRESSED_ELO_COLUMNS = [
    "season_elo_diff_carryover_0_25",
    "season_elo_diff_carryover_0_5",
]
REST_BACK_TO_BACK_FEATURES = ["rest_days_A", "rest_days_B", "rest_diff", "is_back_to_back_A", "is_back_to_back_B"]
ROLLING_RECENT_FORM_AUDIT_FEATURES = [
    "last_3_win_pct_diff",
    "last_5_win_pct_diff",
    "last_10_win_pct_diff",
    "last_5_net_rating_diff",
    "last_10_net_rating_diff",
    "weighted_recent_point_diff",
]
SEED_DIRECTION_FEATURES = ["seed_difference", "higher_seed_A"]
SERIES_CONTEXT_FEATURES = ["game_number", "elimination_game"]
PRODUCTION_FEATURE_SET_NAME = "baseline_plus_corrected_signs"
OLD_HOME_SPLIT_FEATURES = ["home_team_A", "home_win_pct_diff", "away_win_pct_diff"]
ONLY_HOME_TEAM_FEATURES = ["home_team_A"]
HOME_ADVANTAGE_FEATURES = ["home_team_A", "home_advantage_diff"]
CLIPPED_HOME_SPLIT_FEATURES = ["home_team_A", "clipped_home_win_pct_diff", "clipped_away_win_pct_diff"]
PRODUCTION_HOME_FEATURE_SET_NAME = "clipped_home_split_features"
HOME_FEATURE_SET_COLUMNS = {
    "old_home_split_features": OLD_HOME_SPLIT_FEATURES,
    "only_home_team_A": ONLY_HOME_TEAM_FEATURES,
    "home_advantage_diff": HOME_ADVANTAGE_FEATURES,
    "clipped_home_split_features": CLIPPED_HOME_SPLIT_FEATURES,
}
PRODUCTION_FEATURE_COLUMNS = TIER_1_FEATURES + CLIPPED_HOME_SPLIT_FEATURES + SEED_DIRECTION_FEATURES
# Playoff context is used for display/chat/series simulation, not single-game predict_proba.
PLAYOFF_CONTEXT_FEATURE_COLUMNS = PRODUCTION_FEATURE_COLUMNS
PREDICTION_MODE_CURRENT = "Current Hypothetical"
PREDICTION_MODE_PLAYOFF = "Playoff Series Context"
CALIBRATION_METHOD_RAW = "raw"
CALIBRATION_METHODS = [CALIBRATION_METHOD_RAW, "sigmoid", "isotonic"]
PRODUCTION_MODEL_DEFAULTS = {
    PREDICTION_MODE_CURRENT: ("Random Forest", "isotonic"),
    PREDICTION_MODE_PLAYOFF: ("Random Forest", "isotonic"),
}
NON_LEAKY_AUDIT_EXCLUDED_FEATURES = {"series_score_diff", "game_number", "elimination_game"}


def build_model(random_state: int = 42) -> Pipeline:
    return build_random_forest_model(random_state=random_state)


def build_logistic_regression_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_random_forest_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=4,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_gradient_boosting_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(random_state=random_state)),
        ]
    )


def build_extra_trees_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_svc_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, class_weight="balanced", random_state=random_state)),
        ]
    )


def build_adaboost_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", AdaBoostClassifier(n_estimators=250, random_state=random_state)),
        ]
    )


def build_knn_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7)),
        ]
    )


CALIBRATION_MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression_model,
    "Random Forest": build_random_forest_model,
    "Extra Trees": build_extra_trees_model,
}


def _validate_training_frame(training_frame: pd.DataFrame, feature_columns: list[str] | None = None) -> None:
    if training_frame.empty:
        raise ValueError("No training rows were created. Check the selected seasons and NBA API access.")

    feature_columns = feature_columns or DIFF_COLUMNS
    missing_columns = [column for column in feature_columns if column not in training_frame.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing required feature columns: {missing_columns}")


def _evaluate_model(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probabilities = np.clip(pipeline.predict_proba(x_test)[:, 1], 1e-15, 1 - 1e-15)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "classification_report": classification_report(y_test, predictions),
    }


def _calibration_cv_folds(y_train: pd.Series, preferred: int = 3) -> int | None:
    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        return None
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return None
    return min(preferred, min_class_count)


def _build_probability_estimator(model_name: str, calibration_method: str, y_train: pd.Series):
    builder = CALIBRATION_MODEL_BUILDERS[model_name]
    base_pipeline = builder()
    if calibration_method == CALIBRATION_METHOD_RAW:
        return base_pipeline

    cv_folds = _calibration_cv_folds(y_train)
    if cv_folds is None:
        raise ValueError("Calibration requires at least two examples from each class in the training split.")
    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method=calibration_method,
        cv=cv_folds,
    )


def _expected_calibration_error(y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> float:
    rows = _calibration_curve_rows(y_true, probabilities, n_bins=n_bins)
    if not rows:
        return float("nan")
    total = sum(row["bin_count"] for row in rows)
    if total == 0:
        return float("nan")
    return float(
        sum(
            (row["bin_count"] / total) * abs(row["observed_win_rate"] - row["mean_predicted_probability"])
            for row in rows
            if row["bin_count"] > 0
        )
    )


def _calibration_curve_rows(y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> list[dict]:
    y_values = pd.Series(y_true).astype(float).to_numpy()
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for index in range(n_bins):
        lower = bins[index]
        upper = bins[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count == 0:
            mean_probability = np.nan
            observed_rate = np.nan
        else:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(y_values[mask].mean())
        rows.append(
            {
                "bin_index": index + 1,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_count": count,
                "mean_predicted_probability": mean_probability,
                "observed_win_rate": observed_rate,
                "calibration_gap": float(abs(observed_rate - mean_probability))
                if count > 0
                else np.nan,
            }
        )
    return rows


def _sort_probability_model_results(results: pd.DataFrame) -> pd.DataFrame:
    sorted_results = results.sort_values(
        ["roc_auc", "brier_score", "log_loss", "accuracy", "f1"],
        ascending=[False, True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    sorted_results["validation_rank"] = range(1, len(sorted_results) + 1)
    return sorted_results


def _select_production_probability_row(
    summary: pd.DataFrame,
    prediction_context_mode: str | None,
) -> dict:
    preferred = PRODUCTION_MODEL_DEFAULTS.get(str(prediction_context_mode or ""))
    if preferred:
        preferred_model, preferred_calibration = preferred
        preferred_rows = summary[
            summary["model"].eq(preferred_model)
            & summary["calibration_method"].eq(preferred_calibration)
        ]
        if not preferred_rows.empty:
            row = preferred_rows.iloc[0].to_dict()
            row["selected_by"] = "calibrated_random_forest_production_default"
            row["is_validation_best"] = int(row.get("validation_rank", 0)) == 1
            row["preferred_production_model"] = preferred_model
            row["preferred_calibration_method"] = preferred_calibration
            return row

    row = summary.iloc[0].to_dict()
    row["selected_by"] = "best_validation_metrics"
    row["is_validation_best"] = True
    return row


def _fit_probability_model_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    train_seasons: list[str] | None = None,
    test_seasons: list[str] | None = None,
    feature_set_name: str | None = None,
    prediction_context_mode: str | None = None,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], object]]:
    y_train = train_df["TEAM_A_WON"]
    y_test = test_df["TEAM_A_WON"]
    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]

    summary_rows: list[dict] = []
    calibration_rows: list[dict] = []
    fitted_models: dict[tuple[str, str], object] = {}

    for model_name in CALIBRATION_MODEL_BUILDERS:
        for calibration_method in CALIBRATION_METHODS:
            estimator = _build_probability_estimator(model_name, calibration_method, y_train)
            estimator.fit(x_train, y_train)
            fitted_models[(model_name, calibration_method)] = estimator
            evaluation = _evaluate_model(estimator, x_test, y_test)
            probabilities = np.clip(estimator.predict_proba(x_test)[:, 1], 1e-15, 1 - 1e-15)
            ece = _expected_calibration_error(y_test, probabilities, n_bins=n_bins)
            base_row = {
                "model": model_name,
                "model_type": model_name,
                "calibration_method": calibration_method,
                "train_seasons": ",".join(train_seasons or []),
                "test_seasons": ",".join(test_seasons or []),
                "feature_set": feature_set_name or "",
                "prediction_context_mode": prediction_context_mode or "",
                "n_features": int(len(feature_columns)),
                "features": ",".join(feature_columns),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "accuracy": evaluation["accuracy"],
                "roc_auc": evaluation["roc_auc"],
                "brier_score": evaluation["brier_score"],
                "log_loss": evaluation["log_loss"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1": evaluation["f1"],
                "expected_calibration_error": ece,
            }
            summary_rows.append(base_row)
            for curve_row in _calibration_curve_rows(y_test, probabilities, n_bins=n_bins):
                calibration_rows.append({**base_row, **curve_row})

    summary = _sort_probability_model_results(pd.DataFrame(summary_rows))
    calibration = pd.DataFrame(calibration_rows)
    if not calibration.empty:
        calibration = calibration.sort_values(
            ["prediction_context_mode", "model", "calibration_method", "bin_index"],
            na_position="last",
        ).reset_index(drop=True)
    return summary, calibration, fitted_models


def train_model(training_frame: pd.DataFrame, model_path: str | Path) -> dict:
    _validate_training_frame(training_frame)

    x = training_frame[DIFF_COLUMNS]
    y = training_frame["TEAM_A_WON"]
    stratify = y if y.nunique() == 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_model()
    pipeline.fit(x_train, y_train)

    evaluation = _evaluate_model(pipeline, x_test, y_test)

    metrics = {
        "rows": int(len(training_frame)),
        "feature_columns": DIFF_COLUMNS,
        "missing_feature_values": int(x.isna().sum().sum()),
        "rows_with_missing_features": int(x.isna().any(axis=1).sum()),
        "accuracy": evaluation["accuracy"],
        "roc_auc": evaluation["roc_auc"],
        "brier_score": evaluation["brier_score"],
        "log_loss": evaluation["log_loss"],
        "precision": evaluation["precision"],
        "recall": evaluation["recall"],
        "f1": evaluation["f1"],
        "classification_report": evaluation["classification_report"],
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": DIFF_COLUMNS,
            "metrics": metrics,
        },
        model_path,
    )
    return metrics


def compare_models_by_season(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    model_path: str | Path,
    comparison_path: str | Path,
    feature_importance_path: str | Path | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or DIFF_COLUMNS
    _validate_training_frame(training_frame, feature_columns)

    train_mask = training_frame["SEASON"].astype(str).isin(train_seasons)
    test_mask = training_frame["SEASON"].astype(str).isin(test_seasons)
    train_df = training_frame[train_mask].copy()
    test_df = training_frame[test_mask].copy()

    if train_df.empty:
        raise ValueError(f"No rows found for train seasons: {train_seasons}")
    if test_df.empty:
        raise ValueError(f"No rows found for test seasons: {test_seasons}")

    x_train = train_df[feature_columns]
    y_train = train_df["TEAM_A_WON"]
    x_test = test_df[feature_columns]
    y_test = test_df["TEAM_A_WON"]

    model_builders = {
        "Logistic Regression": build_logistic_regression_model,
        "Random Forest": build_random_forest_model,
        "Gradient Boosting": build_gradient_boosting_model,
        "Extra Trees": build_extra_trees_model,
        "SVC": build_svc_model,
        "AdaBoost": build_adaboost_model,
        "KNN": build_knn_model,
    }
    rows: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, builder in model_builders.items():
        pipeline = builder()
        pipeline.fit(x_train, y_train)
        fitted_models[name] = pipeline
        evaluation = _evaluate_model(pipeline, x_test, y_test)
        rows.append(
            {
                "model": name,
                "train_seasons": ",".join(train_seasons),
                "test_seasons": ",".join(test_seasons),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "missing_feature_values_train": int(x_train.isna().sum().sum()),
                "missing_feature_values_test": int(x_test.isna().sum().sum()),
                "accuracy": evaluation["accuracy"],
                "roc_auc": evaluation["roc_auc"],
                "brier_score": evaluation["brier_score"],
                "log_loss": evaluation["log_loss"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1": evaluation["f1"],
            }
        )

    comparison = pd.DataFrame(rows)
    sort_columns = ["roc_auc", "accuracy", "f1"]
    comparison = comparison.sort_values(sort_columns, ascending=False, na_position="last").reset_index(drop=True)
    winner = comparison.sort_values(sort_columns, ascending=False, na_position="last").iloc[0]
    best_model_name = str(winner["model"])

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": fitted_models[best_model_name],
            "feature_columns": feature_columns,
            "metrics": winner.to_dict(),
            "comparison": comparison.to_dict(orient="records"),
        },
        model_path,
    )

    comparison_path = Path(comparison_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    if feature_importance_path is not None:
        importances = extract_feature_importances(fitted_models[best_model_name], best_model_name, feature_columns)
        feature_importance_path = Path(feature_importance_path)
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        importances.to_csv(feature_importance_path, index=False)
    return comparison


def _available_features(candidates: list[str], training_frame: pd.DataFrame) -> list[str]:
    return [feature for feature in candidates if feature in training_frame.columns]


def _without_series_context(features: list[str], include_series_context: bool = False) -> list[str]:
    if include_series_context:
        return features
    return [feature for feature in features if feature not in SERIES_CONTEXT_FEATURES]


def build_ablation_feature_sets(
    training_frame: pd.DataFrame,
    ranked_features: list[str],
    include_series_context: bool = False,
) -> dict[str, list[str]]:
    """Return named feature groups for season-based ablation testing."""
    baseline = _available_features(TIER_1_FEATURES, training_frame)
    corrected_signs = _available_features(PRODUCTION_FEATURE_COLUMNS, training_frame)
    ranked_features = _without_series_context(ranked_features, include_series_context=include_series_context)
    return {
        "baseline_original_features": baseline,
        "baseline_plus_corrected_signs": corrected_signs,
        "baseline_plus_h2h": _available_features(corrected_signs + TIER_8_FEATURES, training_frame),
        "baseline_plus_style_matchups": _available_features(corrected_signs + TIER_10_FEATURES, training_frame),
        "baseline_plus_weighted_recent_form": _available_features(corrected_signs + WEIGHTED_RECENT_FEATURES, training_frame),
        "baseline_plus_star_features": _available_features(corrected_signs + TIER_3_FEATURES, training_frame),
        "all_features": _available_features(DIFF_COLUMNS, training_frame),
        "top_10_by_importance": _available_features(ranked_features[:10], training_frame),
        "top_15_by_importance": _available_features(ranked_features[:15], training_frame),
        "top_20_by_importance": _available_features(ranked_features[:20], training_frame),
    }


def evaluate_feature_group_ablation(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    ablation_path: str | Path,
    model_path: str | Path,
    feature_importance_path: str | Path | None = None,
    production_feature_set: str = PRODUCTION_FEATURE_SET_NAME,
    include_series_context_in_topn: bool = False,
) -> pd.DataFrame:
    """Evaluate Random Forest and Extra Trees across curated feature groups."""
    _validate_training_frame(training_frame)

    train_mask = training_frame["SEASON"].astype(str).isin(train_seasons)
    test_mask = training_frame["SEASON"].astype(str).isin(test_seasons)
    train_df = training_frame[train_mask].copy()
    test_df = training_frame[test_mask].copy()
    if train_df.empty:
        raise ValueError(f"No rows found for train seasons: {train_seasons}")
    if test_df.empty:
        raise ValueError(f"No rows found for test seasons: {test_seasons}")

    y_train = train_df["TEAM_A_WON"]
    y_test = test_df["TEAM_A_WON"]

    ranker = build_extra_trees_model()
    ranker.fit(train_df[DIFF_COLUMNS], y_train)
    ranked_features = (
        pd.DataFrame(
            {
                "feature": DIFF_COLUMNS,
                "importance": ranker.named_steps["classifier"].feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        ["feature"]
        .tolist()
    )
    feature_sets = build_ablation_feature_sets(
        training_frame,
        ranked_features,
        include_series_context=include_series_context_in_topn,
    )

    model_builders = {
        "Random Forest": build_random_forest_model,
        "Extra Trees": build_extra_trees_model,
    }
    rows: list[dict] = []
    fitted_models: dict[tuple[str, str], Pipeline] = {}

    for feature_set_name, features in feature_sets.items():
        if not features:
            continue
        for model_name, builder in model_builders.items():
            pipeline = builder()
            pipeline.fit(train_df[features], y_train)
            fitted_models[(feature_set_name, model_name)] = pipeline
            evaluation = _evaluate_model(pipeline, test_df[features], y_test)
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n_features": int(len(features)),
                    "features": ",".join(features),
                    "is_production_feature_set": feature_set_name == production_feature_set,
                    "production_feature_set": production_feature_set,
                    "train_seasons": ",".join(train_seasons),
                    "test_seasons": ",".join(test_seasons),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "accuracy": evaluation["accuracy"],
                    "roc_auc": evaluation["roc_auc"],
                    "brier_score": evaluation["brier_score"],
                    "log_loss": evaluation["log_loss"],
                    "precision": evaluation["precision"],
                    "recall": evaluation["recall"],
                    "f1": evaluation["f1"],
                }
            )

    ablation = pd.DataFrame(rows)
    sort_columns = ["roc_auc", "accuracy", "f1"]
    ablation = ablation.sort_values(sort_columns, ascending=False, na_position="last").reset_index(drop=True)

    ablation_path = Path(ablation_path)
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(ablation_path, index=False)

    production_rows = ablation[ablation["feature_set"].eq(production_feature_set)].copy()
    if production_rows.empty:
        production_rows = ablation.head(1).copy()
    production_winner = production_rows.sort_values(sort_columns, ascending=False, na_position="last").iloc[0]
    best_feature_set = str(production_winner["feature_set"])
    best_model_name = str(production_winner["model"])
    best_features = str(production_winner["features"]).split(",")
    best_pipeline = fitted_models[(best_feature_set, best_model_name)]

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_columns": best_features,
            "metrics": production_winner.to_dict(),
            "feature_set": best_feature_set,
            "production_feature_set": production_feature_set,
            "ablation": ablation.to_dict(orient="records"),
        },
        model_path,
    )

    if feature_importance_path is not None:
        importances = extract_feature_importances(best_pipeline, best_model_name, best_features)
        feature_importance_path = Path(feature_importance_path)
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        importances.to_csv(feature_importance_path, index=False)

    return ablation


def evaluate_home_feature_ablation(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    home_ablation_path: str | Path,
) -> pd.DataFrame:
    """Compare home-court feature designs without changing the wider model groups."""
    _validate_training_frame(training_frame)

    train_mask = training_frame["SEASON"].astype(str).isin(train_seasons)
    test_mask = training_frame["SEASON"].astype(str).isin(test_seasons)
    train_df = training_frame[train_mask].copy()
    test_df = training_frame[test_mask].copy()
    if train_df.empty:
        raise ValueError(f"No rows found for train seasons: {train_seasons}")
    if test_df.empty:
        raise ValueError(f"No rows found for test seasons: {test_seasons}")

    feature_sets = {
        name: TIER_1_FEATURES + home_features + SEED_DIRECTION_FEATURES
        for name, home_features in HOME_FEATURE_SET_COLUMNS.items()
    }
    model_builders = {
        "Random Forest": build_random_forest_model,
        "Extra Trees": build_extra_trees_model,
    }
    y_train = train_df["TEAM_A_WON"]
    y_test = test_df["TEAM_A_WON"]
    rows: list[dict] = []

    for feature_set_name, features in feature_sets.items():
        available = _available_features(features, training_frame)
        if len(available) != len(features):
            continue
        for model_name, builder in model_builders.items():
            pipeline = builder()
            pipeline.fit(train_df[available], y_train)
            evaluation = _evaluate_model(pipeline, test_df[available], y_test)
            rows.append(
                {
                    "home_feature_set": feature_set_name,
                    "model": model_name,
                    "n_features": int(len(available)),
                    "features": ",".join(available),
                    "is_selected_production_home_design": feature_set_name == PRODUCTION_HOME_FEATURE_SET_NAME,
                    "train_seasons": ",".join(train_seasons),
                    "test_seasons": ",".join(test_seasons),
                    "accuracy": evaluation["accuracy"],
                    "roc_auc": evaluation["roc_auc"],
                    "brier_score": evaluation["brier_score"],
                    "log_loss": evaluation["log_loss"],
                    "precision": evaluation["precision"],
                    "recall": evaluation["recall"],
                    "f1": evaluation["f1"],
                }
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["roc_auc", "accuracy", "f1"], ascending=False, na_position="last").reset_index(drop=True)
        selected_name, _selected_columns = select_production_home_feature_set(results)
        results["is_selected_production_home_design"] = results["home_feature_set"].eq(selected_name)
        results["selected_home_feature_design"] = selected_name

    home_ablation_path = Path(home_ablation_path)
    home_ablation_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(home_ablation_path, index=False)
    return results


def build_calibrated_feature_audit_sets(
    training_frame: pd.DataFrame,
    home_feature_columns: list[str] | None = None,
) -> dict[str, tuple[list[str], str, str]]:
    """Return calibrated Random Forest audit groups without changing production defaults."""
    home_feature_columns = home_feature_columns or HOME_FEATURE_SET_COLUMNS[PRODUCTION_HOME_FEATURE_SET_NAME]
    production = _available_features(TIER_1_FEATURES + home_feature_columns + SEED_DIRECTION_FEATURES, training_frame)
    clipped_home_splits = _available_features(
        TIER_1_FEATURES + CLIPPED_HOME_SPLIT_FEATURES + SEED_DIRECTION_FEATURES,
        training_frame,
    )
    home_advantage_plus_clipped_splits = _available_features(
        TIER_1_FEATURES
        + HOME_ADVANTAGE_FEATURES
        + ["clipped_home_win_pct_diff", "clipped_away_win_pct_diff"]
        + SEED_DIRECTION_FEATURES,
        training_frame,
    )
    playoff_form = _available_features(PLAYOFF_FORM_AUDIT_FEATURES, training_frame)
    elo_features = _available_features(SEASON_RESET_ELO_FEATURES + OFFSEASON_REGRESSED_ELO_COLUMNS, training_frame)
    rest_features = _available_features(REST_BACK_TO_BACK_FEATURES, training_frame)

    current_calibration = PRODUCTION_MODEL_DEFAULTS[PREDICTION_MODE_CURRENT][1]
    return {
        "current_production_features": (production, PREDICTION_MODE_CURRENT, current_calibration),
        "prior_clipped_home_away_splits": (clipped_home_splits, PREDICTION_MODE_CURRENT, current_calibration),
        "production_plus_clipped_home_away_splits": (
            home_advantage_plus_clipped_splits,
            PREDICTION_MODE_CURRENT,
            current_calibration,
        ),
        "production_plus_playoff_form": (
            _available_features(production + playoff_form, training_frame),
            PREDICTION_MODE_CURRENT,
            current_calibration,
        ),
        "production_plus_elo": (
            _available_features(production + elo_features, training_frame),
            PREDICTION_MODE_CURRENT,
            current_calibration,
        ),
        "production_plus_rest": (
            _available_features(production + rest_features, training_frame),
            PREDICTION_MODE_CURRENT,
            current_calibration,
        ),
        "production_plus_playoff_form_elo_rest": (
            _available_features(production + playoff_form + elo_features + rest_features, training_frame),
            PREDICTION_MODE_CURRENT,
            current_calibration,
        ),
    }


def evaluate_calibrated_feature_audit(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    audit_path: str | Path,
    home_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Audit calibrated Random Forest validation performance across existing feature groups."""
    train_df, test_df = _season_split(training_frame, train_seasons, test_seasons)
    feature_sets = build_calibrated_feature_audit_sets(training_frame, home_feature_columns=home_feature_columns)
    rows: list[dict] = []

    for feature_set_name, (features, prediction_context_mode, calibration_method) in feature_sets.items():
        if not features:
            continue
        _validate_training_frame(training_frame, features)
        estimator = _build_probability_estimator("Random Forest", calibration_method, train_df["TEAM_A_WON"])
        estimator.fit(train_df[features], train_df["TEAM_A_WON"])
        evaluation = _evaluate_model(estimator, test_df[features], test_df["TEAM_A_WON"])
        probabilities = np.clip(estimator.predict_proba(test_df[features])[:, 1], 1e-15, 1 - 1e-15)
        rows.append(
            {
                "feature_set": feature_set_name,
                "model": "Random Forest",
                "model_type": "Random Forest",
                "calibration_method": calibration_method,
                "prediction_context_mode": prediction_context_mode,
                "n_features": int(len(features)),
                "features": ",".join(features),
                "train_seasons": ",".join(train_seasons),
                "test_seasons": ",".join(test_seasons),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "roc_auc": evaluation["roc_auc"],
                "brier_score": evaluation["brier_score"],
                "log_loss": evaluation["log_loss"],
                "accuracy": evaluation["accuracy"],
                "f1": evaluation["f1"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "expected_calibration_error": _expected_calibration_error(test_df["TEAM_A_WON"], probabilities),
            }
        )
        print(
            f"{feature_set_name}: "
            f"roc_auc={evaluation['roc_auc']:.4f} "
            f"brier_score={evaluation['brier_score']:.4f} "
            f"log_loss={evaluation['log_loss']:.4f} "
            f"accuracy={evaluation['accuracy']:.4f} "
            f"f1={evaluation['f1']:.4f}"
        )

    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["roc_auc", "brier_score", "log_loss", "accuracy", "f1"],
            ascending=[False, True, True, False, False],
            na_position="last",
        ).reset_index(drop=True)
        audit["audit_rank"] = range(1, len(audit) + 1)
        audit["is_best_feature_set"] = audit["audit_rank"].eq(1)

    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    production = _available_features(
        TIER_1_FEATURES
        + (home_feature_columns or HOME_FEATURE_SET_COLUMNS[PRODUCTION_HOME_FEATURE_SET_NAME])
        + SEED_DIRECTION_FEATURES,
        training_frame,
    )
    focused_audit = evaluate_focused_elo_carryover_audit(
        training_frame=training_frame,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        audit_path=audit_path.with_name("elo_carryover_focused_audit.csv"),
        production_features=production,
    )
    _save_elo_carryover_feature_diagnostics(
        training_frame=training_frame,
        production_features=production,
        summary_path=audit_path.with_name("elo_carryover_feature_summary.csv"),
        correlation_path=audit_path.with_name("elo_carryover_feature_correlation.csv"),
    )
    return audit


def _evaluate_calibrated_random_forest_sets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    train_seasons: list[str],
    test_seasons: list[str],
    calibration_method: str,
    prediction_context_mode: str = PREDICTION_MODE_CURRENT,
) -> pd.DataFrame:
    rows: list[dict] = []
    for feature_set_name, features in feature_sets.items():
        if not features:
            continue
        estimator = _build_probability_estimator("Random Forest", calibration_method, train_df["TEAM_A_WON"])
        estimator.fit(train_df[features], train_df["TEAM_A_WON"])
        evaluation = _evaluate_model(estimator, test_df[features], test_df["TEAM_A_WON"])
        probabilities = np.clip(estimator.predict_proba(test_df[features])[:, 1], 1e-15, 1 - 1e-15)
        rows.append(
            {
                "feature_set": feature_set_name,
                "model": "Random Forest",
                "model_type": "Random Forest",
                "calibration_method": calibration_method,
                "prediction_context_mode": prediction_context_mode,
                "n_features": int(len(features)),
                "features": ",".join(features),
                "train_seasons": ",".join(train_seasons),
                "test_seasons": ",".join(test_seasons),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "roc_auc": evaluation["roc_auc"],
                "brier_score": evaluation["brier_score"],
                "log_loss": evaluation["log_loss"],
                "accuracy": evaluation["accuracy"],
                "f1": evaluation["f1"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "expected_calibration_error": _expected_calibration_error(test_df["TEAM_A_WON"], probabilities),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["roc_auc", "brier_score", "log_loss"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        result["focused_audit_rank"] = range(1, len(result) + 1)
    return result


def evaluate_focused_elo_carryover_audit(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    audit_path: str | Path,
    production_features: list[str] | None = None,
) -> pd.DataFrame:
    """Focused calibrated Random Forest audit for offseason-regressed Elo variants."""
    production_features = production_features or _available_features(PRODUCTION_FEATURE_COLUMNS, training_frame)
    feature_sets = {
        "current_production_features": _available_features(production_features, training_frame),
        "production_plus_carryover_0_25_only": _available_features(
            production_features + ["season_elo_diff_carryover_0_25"],
            training_frame,
        ),
        "production_plus_carryover_0_5_only": _available_features(
            production_features + ["season_elo_diff_carryover_0_5"],
            training_frame,
        ),
    }
    for features in feature_sets.values():
        _validate_training_frame(training_frame, features)
    train_df, test_df = _season_split(training_frame, train_seasons, test_seasons)
    result = _evaluate_calibrated_random_forest_sets(
        train_df=train_df,
        test_df=test_df,
        feature_sets=feature_sets,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        calibration_method=PRODUCTION_MODEL_DEFAULTS[PREDICTION_MODE_CURRENT][1],
    )
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(audit_path, index=False)
    return result


def _save_elo_carryover_feature_diagnostics(
    training_frame: pd.DataFrame,
    production_features: list[str],
    summary_path: str | Path,
    correlation_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save summary stats and pairwise correlations for production and carryover Elo features."""
    diagnostic_features = _available_features(production_features + OFFSEASON_REGRESSED_ELO_COLUMNS, training_frame)
    numeric = training_frame[diagnostic_features].apply(pd.to_numeric, errors="coerce")
    summary = numeric.describe().T.reset_index().rename(columns={"index": "feature"})
    summary["missing_count"] = numeric.isna().sum().reindex(summary["feature"]).to_numpy()
    summary["nunique"] = numeric.nunique(dropna=True).reindex(summary["feature"]).to_numpy()

    correlation_rows: list[dict] = []
    corr = numeric.corr()
    for left in diagnostic_features:
        for right in diagnostic_features:
            if left == right:
                continue
            paired = numeric[[left, right]].dropna()
            mean_abs_difference = np.nan
            identical_values = False
            if not paired.empty:
                difference = (paired[left] - paired[right]).abs()
                mean_abs_difference = float(difference.mean())
                identical_values = bool(difference.eq(0).all())
            correlation_rows.append(
                {
                    "feature_x": left,
                    "feature_y": right,
                    "correlation": corr.loc[left, right] if left in corr.index and right in corr.columns else np.nan,
                    "mean_abs_difference": mean_abs_difference,
                    "identical_values": identical_values,
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    summary_path = Path(summary_path)
    correlation_path = Path(correlation_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    correlation_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    return summary, correlations


def select_production_home_feature_set(home_ablation: pd.DataFrame) -> tuple[str, list[str]]:
    """Choose the best sane home feature design from the home-court ablation."""
    if home_ablation.empty:
        return PRODUCTION_HOME_FEATURE_SET_NAME, HOME_FEATURE_SET_COLUMNS[PRODUCTION_HOME_FEATURE_SET_NAME]

    sane_designs = ["only_home_team_A", "home_advantage_diff", "clipped_home_split_features"]
    candidates = home_ablation[home_ablation["home_feature_set"].isin(sane_designs)].copy()
    if candidates.empty:
        return PRODUCTION_HOME_FEATURE_SET_NAME, HOME_FEATURE_SET_COLUMNS[PRODUCTION_HOME_FEATURE_SET_NAME]

    winner = candidates.sort_values(["roc_auc", "accuracy", "f1"], ascending=False, na_position="last").iloc[0]
    selected_name = str(winner["home_feature_set"])
    return selected_name, HOME_FEATURE_SET_COLUMNS[selected_name]


def _season_split(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = training_frame["SEASON"].astype(str).isin(train_seasons)
    test_mask = training_frame["SEASON"].astype(str).isin(test_seasons)
    train_df = training_frame[train_mask].copy()
    test_df = training_frame[test_mask].copy()
    if train_df.empty:
        raise ValueError(f"No rows found for train seasons: {train_seasons}")
    if test_df.empty:
        raise ValueError(f"No rows found for test seasons: {test_seasons}")
    return train_df, test_df


def _fit_best_production_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    train_seasons: list[str] | None = None,
    test_seasons: list[str] | None = None,
    feature_set_name: str | None = None,
    prediction_context_mode: str | None = None,
) -> tuple[str, object, dict, pd.DataFrame]:
    summary, calibration, fitted = _fit_probability_model_candidates(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        feature_set_name=feature_set_name,
        prediction_context_mode=prediction_context_mode,
    )
    winner = _select_production_probability_row(summary, prediction_context_mode)
    model_name = str(winner["model"])
    calibration_method = str(winner["calibration_method"])
    winner["classification_report"] = classification_report(
        test_df["TEAM_A_WON"],
        (fitted[(model_name, calibration_method)].predict_proba(test_df[feature_columns])[:, 1] >= 0.5).astype(int),
    )
    return model_name, fitted[(model_name, calibration_method)], winner, calibration


def train_production_models(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    model_path: str | Path,
    feature_importance_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    home_feature_set_name: str = PRODUCTION_HOME_FEATURE_SET_NAME,
    home_feature_columns: list[str] | None = None,
) -> dict:
    """Train and save production artifacts for normal and playoff-context predictions.

    Both modes use the same single-game feature set. Playoff series inputs stay in
    metadata/context and feed series win probability simulation outside predict_proba.
    """
    home_feature_columns = home_feature_columns or HOME_FEATURE_SET_COLUMNS.get(
        home_feature_set_name,
        HOME_FEATURE_SET_COLUMNS[PRODUCTION_HOME_FEATURE_SET_NAME],
    )
    current_feature_columns = TIER_1_FEATURES + home_feature_columns + SEED_DIRECTION_FEATURES
    playoff_feature_columns = list(current_feature_columns)

    for columns in [current_feature_columns, playoff_feature_columns]:
        _validate_training_frame(training_frame, columns)

    train_df, test_df = _season_split(training_frame, train_seasons, test_seasons)
    current_model_name, current_pipeline, current_metrics, current_calibration = _fit_best_production_pipeline(
        train_df,
        test_df,
        current_feature_columns,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        feature_set_name=PRODUCTION_FEATURE_SET_NAME,
        prediction_context_mode=PREDICTION_MODE_CURRENT,
    )
    playoff_model_name, playoff_pipeline, playoff_metrics, playoff_calibration = _fit_best_production_pipeline(
        train_df,
        test_df,
        playoff_feature_columns,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
        feature_set_name=PRODUCTION_FEATURE_SET_NAME,
        prediction_context_mode=PREDICTION_MODE_PLAYOFF,
    )

    current_metrics.update(
        {
            "model_type": current_model_name,
            "train_seasons": ",".join(train_seasons),
            "test_seasons": ",".join(test_seasons),
            "feature_set": PRODUCTION_FEATURE_SET_NAME,
            "features": ",".join(current_feature_columns),
        }
    )
    playoff_metrics.update(
        {
            "model_type": playoff_model_name,
            "train_seasons": ",".join(train_seasons),
            "test_seasons": ",".join(test_seasons),
            "feature_set": PRODUCTION_FEATURE_SET_NAME,
            "features": ",".join(playoff_feature_columns),
            "series_context_used_for_game_prediction": False,
        }
    )

    artifact = {
        "pipeline": current_pipeline,
        "feature_columns": current_feature_columns,
        "current_hypothetical_model": current_pipeline,
        "current_hypothetical_features": current_feature_columns,
        "playoff_context_model": playoff_pipeline,
        "playoff_context_features": playoff_feature_columns,
        "metrics": {
            **current_metrics,
            "feature_set": PRODUCTION_FEATURE_SET_NAME,
            "prediction_context_mode": PREDICTION_MODE_CURRENT,
            "home_feature_set": home_feature_set_name,
            "selected_home_feature_design": home_feature_set_name,
        },
        "feature_set": PRODUCTION_FEATURE_SET_NAME,
        "production_feature_set": PRODUCTION_FEATURE_SET_NAME,
        "home_feature_set": home_feature_set_name,
        "selected_home_feature_design": home_feature_set_name,
        "production_models": {
            PREDICTION_MODE_CURRENT: {
                "pipeline": current_pipeline,
                "feature_columns": current_feature_columns,
                "metrics": current_metrics,
                "feature_set": PRODUCTION_FEATURE_SET_NAME,
                "home_feature_set": home_feature_set_name,
                "selected_home_feature_design": home_feature_set_name,
            },
            PREDICTION_MODE_PLAYOFF: {
                "pipeline": playoff_pipeline,
                "feature_columns": playoff_feature_columns,
                "metrics": playoff_metrics,
                "feature_set": PRODUCTION_FEATURE_SET_NAME,
                "home_feature_set": home_feature_set_name,
                "selected_home_feature_design": home_feature_set_name,
            },
        },
        "models": {
            PREDICTION_MODE_CURRENT: {
                "pipeline": current_pipeline,
                "feature_columns": current_feature_columns,
                "metrics": current_metrics,
                "feature_set": PRODUCTION_FEATURE_SET_NAME,
                "home_feature_set": home_feature_set_name,
                "selected_home_feature_design": home_feature_set_name,
            },
            PREDICTION_MODE_PLAYOFF: {
                "pipeline": playoff_pipeline,
                "feature_columns": playoff_feature_columns,
                "metrics": playoff_metrics,
                "feature_set": PRODUCTION_FEATURE_SET_NAME,
                "home_feature_set": home_feature_set_name,
                "selected_home_feature_design": home_feature_set_name,
            },
        },
        "metadata": {
            "train_seasons": train_seasons,
            "test_seasons": test_seasons,
            "current_hypothetical_features": current_feature_columns,
            "playoff_context_features": playoff_feature_columns,
            "series_features_neutral_in_current_hypothetical": True,
            "series_context_used_for_game_prediction": False,
            "series_context_used_for_series_probability": True,
            "game_prediction_feature_policy": (
                "Current Hypothetical and Playoff Series Context both use the current "
                "production feature set for single-game predict_proba. Playoff series "
                "context affects display, chatbot/debug context, and separate series "
                "win probability simulation only."
            ),
            "series_context_metadata_features": ["game_number", "elimination_game", "series_score_diff"],
            "home_feature_set": home_feature_set_name,
            "selected_home_feature_design": home_feature_set_name,
            "model_selection": "season validation sorted by ROC-AUC, then Brier score, log loss, accuracy, and F1",
            "selected_models": {
                PREDICTION_MODE_CURRENT: {
                    "model_type": current_metrics.get("model_type"),
                    "calibration_method": current_metrics.get("calibration_method"),
                    "train_seasons": train_seasons,
                    "test_seasons": test_seasons,
                    "feature_set": PRODUCTION_FEATURE_SET_NAME,
                    "metrics": current_metrics,
                },
                PREDICTION_MODE_PLAYOFF: {
                    "model_type": playoff_metrics.get("model_type"),
                    "calibration_method": playoff_metrics.get("calibration_method"),
                    "train_seasons": train_seasons,
                    "test_seasons": test_seasons,
                    "feature_set": PRODUCTION_FEATURE_SET_NAME,
                    "metrics": playoff_metrics,
                },
            },
        },
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    if calibration_path is not None:
        calibration = pd.concat([current_calibration, playoff_calibration], ignore_index=True)
        calibration_path = Path(calibration_path)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration.to_csv(calibration_path, index=False)

    if feature_importance_path is not None:
        current_importances = extract_feature_importances(
            current_pipeline,
            current_model_name,
            current_feature_columns,
        )
        playoff_importances = extract_feature_importances(
            playoff_pipeline,
            playoff_model_name,
            playoff_feature_columns,
        )
        current_importances["prediction_context_mode"] = PREDICTION_MODE_CURRENT
        playoff_importances["prediction_context_mode"] = PREDICTION_MODE_PLAYOFF
        importances = pd.concat([current_importances, playoff_importances], ignore_index=True)
        feature_importance_path = Path(feature_importance_path)
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        importances.to_csv(feature_importance_path, index=False)

    return artifact


def get_model_entry_for_mode(model_bundle: dict, prediction_context_mode: str) -> dict:
    """Return the saved pipeline/feature set for the requested prediction mode."""
    models = model_bundle.get("production_models") or model_bundle.get("models") or {}
    if prediction_context_mode in models:
        return models[prediction_context_mode]
    return model_bundle


def select_features_with_extra_trees(
    training_frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
    feature_selection_path: str | Path,
    candidate_sizes: list[int | str] | None = None,
    include_series_context: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Rank features with Extra Trees, evaluate top-N subsets, and return the best subset."""
    candidate_sizes = candidate_sizes or FEATURE_SELECTION_SIZES
    _validate_training_frame(training_frame)

    train_mask = training_frame["SEASON"].astype(str).isin(train_seasons)
    test_mask = training_frame["SEASON"].astype(str).isin(test_seasons)
    train_df = training_frame[train_mask].copy()
    test_df = training_frame[test_mask].copy()
    if train_df.empty:
        raise ValueError(f"No rows found for train seasons: {train_seasons}")
    if test_df.empty:
        raise ValueError(f"No rows found for test seasons: {test_seasons}")

    x_train_all = train_df[DIFF_COLUMNS]
    y_train = train_df["TEAM_A_WON"]
    x_test_all = test_df[DIFF_COLUMNS]
    y_test = test_df["TEAM_A_WON"]

    ranker = build_extra_trees_model()
    ranker.fit(x_train_all, y_train)
    importances = ranker.named_steps["classifier"].feature_importances_
    ranked_features = (
        pd.DataFrame({"feature": DIFF_COLUMNS, "importance": importances})
        .sort_values("importance", ascending=False)
        ["feature"]
        .tolist()
    )
    ranked_features = _without_series_context(ranked_features, include_series_context=include_series_context)

    rows: list[dict] = []
    total_features = len(DIFF_COLUMNS)
    for size in candidate_sizes:
        n_features = total_features if size == "all" else min(int(size), total_features)
        selected_features = ranked_features[:n_features]
        pipeline = build_extra_trees_model()
        pipeline.fit(x_train_all[selected_features], y_train)
        evaluation = _evaluate_model(pipeline, x_test_all[selected_features], y_test)
        rows.append(
            {
                "n_features": size,
                "resolved_n_features": n_features,
                "features": ",".join(selected_features),
                "production_feature_set": PRODUCTION_FEATURE_SET_NAME,
                "is_production_feature_set": False,
                "accuracy": evaluation["accuracy"],
                "roc_auc": evaluation["roc_auc"],
                "brier_score": evaluation["brier_score"],
                "log_loss": evaluation["log_loss"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1": evaluation["f1"],
            }
        )

    results = pd.DataFrame(rows).sort_values(["roc_auc", "accuracy", "f1"], ascending=False, na_position="last")
    results = results.reset_index(drop=True)
    feature_selection_path = Path(feature_selection_path)
    feature_selection_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(feature_selection_path, index=False)
    best_features = str(results.iloc[0]["features"]).split(",")
    return results, best_features


def extract_feature_importances(
    pipeline,
    model_name: str,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or DIFF_COLUMNS
    if not hasattr(pipeline, "named_steps"):
        return pd.DataFrame(
            columns=["model", "feature", "importance", "importance_type"],
            data=[[model_name, feature, np.nan, "not_available_for_calibrated_model"] for feature in feature_columns],
        )
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_).ravel()
    else:
        return pd.DataFrame(
            columns=["model", "feature", "importance", "importance_type"],
            data=[[model_name, feature, np.nan, "not_available"] for feature in feature_columns],
        )

    return (
        pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_columns,
                "importance": values,
                "importance_type": "feature_importances_or_abs_coef",
            }
        )
        .sort_values("importance", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def load_model(model_path: str | Path) -> dict:
    return joblib.load(model_path)
