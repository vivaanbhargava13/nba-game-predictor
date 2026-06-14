from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import PREDICTION_MODE_CURRENT, get_model_entry_for_mode, load_model
from .nba_data import add_team_game_metrics, season_range
from .series_context import (
    SERIES_CONTEXT_FEATURE_COLUMNS,
    build_series_state_features,
    reverse_series_context_frame,
    symmetric_model_probability,
)

DEFAULT_BASE_TRAINING_PATH = Path("data/processed/training_matchups_regular_season.csv")
DEFAULT_BASE_MODEL_PATH = Path("models/playoff_predictor.joblib")
DEFAULT_OUTPUT_DIR = Path("data/processed")
EXPERIMENT_SCHEMA_VERSION = "2026-06-12-playoff-recent-form-v1"

WINDOW_METRICS = {
    "win_pct": "WON",
    "point_margin": "POINT_DIFF",
    "net_rating": "NET_RATING_GAME",
    "off_rating": "OFF_RATING_GAME",
    "def_rating": "DEF_RATING_GAME",
    "pace": "PACE_GAME",
}
RECENT_WINDOWS = (3, 5)
TEAM_RECENT_FEATURE_COLUMNS = [
    f"last_{window}_{metric}_diff"
    for window in RECENT_WINDOWS
    for metric in WINDOW_METRICS
] + [
    f"last_{window}_{metric}_vs_season_baseline_diff"
    for window in RECENT_WINDOWS
    for metric in WINDOW_METRICS
]
EXPERIMENT_SERIES_FEATURE_COLUMNS = [
    *SERIES_CONTEXT_FEATURE_COLUMNS,
    "team_a_closeout_game",
    "team_b_closeout_game",
    "cumulative_series_margin_before_game",
    "average_series_margin_before_game",
    "rest_days_diff",
]


def _normalize_game_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True)


def _read_cached_season_games(season: str, cache_dir: Path) -> pd.DataFrame:
    """Read cached regular-season and playoff rows without making API calls."""
    regular_frames = []
    for path in sorted(cache_dir.glob(f"team_game_log_{season}_regular_season_*.csv")):
        frame = pd.read_csv(path)
        frame = frame.rename(columns={"Team_ID": "TEAM_ID", "Game_ID": "GAME_ID"})
        if not frame.empty:
            frame["SEASON_TYPE"] = "Regular Season"
            regular_frames.append(frame)
    if not regular_frames:
        raise FileNotFoundError(f"No cached regular-season TeamGameLog files found for {season}.")

    playoff_path = cache_dir / f"playoff_games_{season}_playoffs.csv"
    if not playoff_path.exists():
        raise FileNotFoundError(f"Cached playoff games not found: {playoff_path}")
    playoffs = pd.read_csv(playoff_path)
    playoffs["SEASON_TYPE"] = "Playoffs"

    games = pd.concat([*regular_frames, playoffs], ignore_index=True, sort=False)
    games["TEAM_ID"] = pd.to_numeric(games["TEAM_ID"], errors="raise").astype(int)
    games["GAME_ID"] = _normalize_game_id(games["GAME_ID"])
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], format="mixed")
    games["WON"] = games["WL"].eq("W").astype(int)
    games["IS_HOME"] = games["MATCHUP"].str.contains("vs.", regex=False, na=False).astype(int)
    games = games.drop_duplicates(["TEAM_ID", "GAME_ID", "SEASON_TYPE"])
    games = add_team_game_metrics(games)
    games["PACE_GAME"] = (
        pd.to_numeric(games["POSS"], errors="coerce")
        + pd.to_numeric(games["OPP_POSS"], errors="coerce")
    ) / 2.0
    return games.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)


def _window_values(
    history: pd.DataFrame,
    baseline: pd.DataFrame,
    window: int,
) -> dict[str, float]:
    recent = history.tail(window)
    values: dict[str, float] = {}
    for metric, column in WINDOW_METRICS.items():
        recent_value = pd.to_numeric(recent[column], errors="coerce").mean()
        baseline_value = pd.to_numeric(baseline[column], errors="coerce").mean()
        values[f"last_{window}_{metric}"] = float(recent_value)
        values[f"last_{window}_{metric}_vs_season_baseline"] = float(
            recent_value - baseline_value
        )
    return values


def build_recent_form_row(
    *,
    season: str,
    game_id: str,
    game_date: pd.Timestamp,
    team_a: pd.Series,
    team_b: pd.Series,
    season_games: pd.DataFrame,
    prior_series_games: list[dict],
    team_a_series_wins: int,
    team_b_series_wins: int,
) -> dict:
    """Build one target row using strictly earlier games."""
    prior = season_games[season_games["GAME_DATE"] < pd.Timestamp(game_date)]
    history_a = prior[prior["TEAM_ID"].eq(int(team_a["TEAM_ID"]))]
    history_b = prior[prior["TEAM_ID"].eq(int(team_b["TEAM_ID"]))]
    baseline_a = history_a[history_a["SEASON_TYPE"].eq("Regular Season")]
    baseline_b = history_b[history_b["SEASON_TYPE"].eq("Regular Season")]

    row = {
        "EXPERIMENT_SCHEMA_VERSION": EXPERIMENT_SCHEMA_VERSION,
        "SEASON": season,
        "GAME_ID": str(game_id),
        "GAME_DATE": pd.Timestamp(game_date),
        "TEAM_A_ID": int(team_a["TEAM_ID"]),
        "TEAM_A": str(team_a["TEAM_ABBREVIATION"]),
        "TEAM_B_ID": int(team_b["TEAM_ID"]),
        "TEAM_B": str(team_b["TEAM_ABBREVIATION"]),
        "TEAM_A_WON": int(team_a["WON"]),
    }
    for window in RECENT_WINDOWS:
        values_a = _window_values(history_a, baseline_a, window)
        values_b = _window_values(history_b, baseline_b, window)
        for metric in WINDOW_METRICS:
            key = f"last_{window}_{metric}"
            baseline_key = f"{key}_vs_season_baseline"
            row[f"{key}_diff"] = values_a[key] - values_b[key]
            row[f"{baseline_key}_diff"] = values_a[baseline_key] - values_b[baseline_key]

    series_state = build_series_state_features(
        game_number=len(prior_series_games) + 1,
        team_a_series_wins=team_a_series_wins,
        team_b_series_wins=team_b_series_wins,
        team_a_home=bool(team_a["IS_HOME"]),
        prior_games=prior_series_games,
    )
    prior_margins = [float(game["team_a_margin"]) for game in prior_series_games]
    row.update(series_state)
    row["team_a_closeout_game"] = float(team_a_series_wins == 3)
    row["team_b_closeout_game"] = float(team_b_series_wins == 3)
    row["cumulative_series_margin_before_game"] = float(sum(prior_margins))
    row["average_series_margin_before_game"] = (
        float(np.mean(prior_margins)) if prior_margins else 0.0
    )
    last_a = history_a["GAME_DATE"].max()
    last_b = history_b["GAME_DATE"].max()
    rest_a = (pd.Timestamp(game_date) - last_a).days if pd.notna(last_a) else np.nan
    rest_b = (pd.Timestamp(game_date) - last_b).days if pd.notna(last_b) else np.nan
    row["rest_days_diff"] = float(rest_a - rest_b)
    return row


def build_recent_form_dataset(
    seasons: list[str],
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    rows: list[dict] = []
    for season in seasons:
        print(f"Building leakage-safe recent-form rows: {season}")
        season_games = _read_cached_season_games(season, cache_dir)
        playoffs = season_games[season_games["SEASON_TYPE"].eq("Playoffs")].copy()
        paired_games = []
        for game_id, group in playoffs.groupby("GAME_ID", sort=False):
            if len(group) != 2:
                continue
            ordered = group.sort_values("TEAM_ID").reset_index(drop=True)
            team_a = ordered.iloc[0]
            team_b = ordered.iloc[1]
            paired_games.append(
                {
                    "GAME_ID": str(game_id),
                    "GAME_DATE": pd.Timestamp(team_a["GAME_DATE"]),
                    "PAIR": (int(team_a["TEAM_ID"]), int(team_b["TEAM_ID"])),
                    "TEAM_A": team_a,
                    "TEAM_B": team_b,
                }
            )
        paired = sorted(paired_games, key=lambda game: (game["GAME_DATE"], game["GAME_ID"]))
        for pair in sorted({game["PAIR"] for game in paired}):
            series_games = [game for game in paired if game["PAIR"] == pair]
            history: list[dict] = []
            team_a_wins = 0
            team_b_wins = 0
            for game in series_games:
                team_a = game["TEAM_A"]
                team_b = game["TEAM_B"]
                rows.append(
                    build_recent_form_row(
                        season=season,
                        game_id=game["GAME_ID"],
                        game_date=game["GAME_DATE"],
                        team_a=team_a,
                        team_b=team_b,
                        season_games=season_games,
                        prior_series_games=history,
                        team_a_series_wins=team_a_wins,
                        team_b_series_wins=team_b_wins,
                    )
                )
                team_a_won = int(team_a["WON"])
                history.append(
                    {
                        "team_a_home": int(team_a["IS_HOME"]),
                        "team_a_won": team_a_won,
                        "team_a_margin": float(team_a["POINT_DIFF"]),
                    }
                )
                team_a_wins += team_a_won
                team_b_wins += 1 - team_a_won
    return pd.DataFrame(rows)


def reverse_recent_form_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reversed_frame = frame.copy()
    for column in TEAM_RECENT_FEATURE_COLUMNS:
        reversed_frame[column] = -pd.to_numeric(frame[column], errors="coerce")
    context = reverse_series_context_frame(frame[SERIES_CONTEXT_FEATURE_COLUMNS])
    for column in SERIES_CONTEXT_FEATURE_COLUMNS:
        reversed_frame[column] = context[column]
    reversed_frame["team_a_closeout_game"] = frame["team_b_closeout_game"]
    reversed_frame["team_b_closeout_game"] = frame["team_a_closeout_game"]
    for column in [
        "cumulative_series_margin_before_game",
        "average_series_margin_before_game",
        "rest_days_diff",
    ]:
        reversed_frame[column] = -pd.to_numeric(frame[column], errors="coerce")
    return reversed_frame


def _logistic_estimator() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _fit_symmetric_logistic(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline:
    direct = frame[feature_columns].reset_index(drop=True)
    reverse = reverse_recent_form_frame(frame)[feature_columns].reset_index(drop=True)
    target = frame["TEAM_A_WON"].astype(int).reset_index(drop=True)
    x = pd.concat([direct, reverse], ignore_index=True)
    y = pd.concat([target, 1 - target], ignore_index=True)
    return _logistic_estimator().fit(x, y)


def _symmetric_probability(
    model,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    direct = frame[feature_columns]
    reverse = reverse_recent_form_frame(frame)[feature_columns]
    return symmetric_model_probability(model, direct, reverse)


def _reverse_base_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    reversed_frame = frame[feature_columns].copy()
    for column in [
        "OFF_RATING_DIFF",
        "DEF_RATING_DIFF",
        "NET_RATING_DIFF",
        "W_PCT_DIFF",
        "PLUS_MINUS_DIFF",
        "PACE_DIFF",
        "clipped_home_win_pct_diff",
        "clipped_away_win_pct_diff",
        "seed_difference",
    ]:
        if column in reversed_frame:
            reversed_frame[column] = -pd.to_numeric(frame[column], errors="coerce")
    if "home_team_A" in reversed_frame:
        reversed_frame["home_team_A"] = 1 - pd.to_numeric(frame["home_team_A"], errors="coerce")
    if "higher_seed_A" in reversed_frame:
        reversed_frame["higher_seed_A"] = 1 - pd.to_numeric(
            frame["higher_seed_A"], errors="coerce"
        )
    return reversed_frame


def _logit(probability) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def _fit_blend(probabilities: list[np.ndarray], target: pd.Series) -> LogisticRegression:
    x = np.column_stack([_logit(values) for values in probabilities])
    y = target.astype(int).to_numpy()
    return LogisticRegression(
        fit_intercept=False,
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        solver="liblinear",
    ).fit(np.vstack([x, -x]), np.concatenate([y, 1 - y]))


def _blend_probability(model, probabilities: list[np.ndarray]) -> np.ndarray:
    x = np.column_stack([_logit(values) for values in probabilities])
    direct = model.predict_proba(x)[:, 1]
    reverse = 1 - model.predict_proba(-x)[:, 1]
    return (direct + reverse) / 2


def _metrics(name: str, target: pd.Series, probability: np.ndarray) -> dict:
    predicted = (probability >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "accuracy": float(accuracy_score(target, predicted)),
    }


def _calibration_rows(
    model_name: str,
    target: pd.Series,
    probability: np.ndarray,
    bins: int = 10,
) -> list[dict]:
    bucket = pd.cut(
        probability,
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        labels=False,
    )
    frame = pd.DataFrame(
        {"bucket": bucket, "target": target.to_numpy(), "probability": probability}
    )
    rows = []
    for bucket_index, group in frame.groupby("bucket", dropna=False):
        rows.append(
            {
                "model": model_name,
                "bucket": int(bucket_index),
                "count": int(len(group)),
                "mean_predicted_probability": float(group["probability"].mean()),
                "observed_win_rate": float(group["target"].mean()),
            }
        )
    return rows


def _merge_base_features(base: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    recent = recent.copy()
    base["GAME_ID"] = _normalize_game_id(base["GAME_ID"])
    recent["GAME_ID"] = _normalize_game_id(recent["GAME_ID"])
    metadata = [
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_A_ID",
        "TEAM_A",
        "TEAM_B_ID",
        "TEAM_B",
        "TEAM_A_WON",
        *TEAM_RECENT_FEATURE_COLUMNS,
        *EXPERIMENT_SERIES_FEATURE_COLUMNS,
    ]
    return base.drop(
        columns=[
            column
            for column in [*TEAM_RECENT_FEATURE_COLUMNS, *EXPERIMENT_SERIES_FEATURE_COLUMNS]
            if column in base.columns
        ],
        errors="ignore",
    ).merge(
        recent[metadata],
        on=["SEASON", "GAME_ID"],
        how="inner",
        suffixes=("", "_recent"),
        validate="one_to_one",
    )


def run_recent_form_experiment(
    *,
    cache_dir: str | Path = "data/raw",
    base_training_path: str | Path = DEFAULT_BASE_TRAINING_PATH,
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_train_seasons: list[str] | None = None,
    blend_seasons: list[str] | None = None,
    test_seasons: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    model_train_seasons = model_train_seasons or season_range("2015-16", "2022-23")
    blend_seasons = blend_seasons or season_range("2018-19", "2022-23")
    test_seasons = test_seasons or ["2023-24", "2024-25"]
    all_seasons = list(dict.fromkeys([*model_train_seasons, *blend_seasons, *test_seasons]))

    recent = build_recent_form_dataset(all_seasons, cache_dir)
    base = pd.read_csv(base_training_path)
    merged = _merge_base_features(base, recent)
    if merged.empty:
        raise ValueError("No matching base and recent-form rows were found.")

    artifact = load_model(base_model_path)
    entry = get_model_entry_for_mode(artifact, PREDICTION_MODE_CURRENT)
    base_template = entry["pipeline"]
    base_features = list(entry["feature_columns"])
    model_train = merged[merged["SEASON"].isin(model_train_seasons)].copy()
    blend = merged[merged["SEASON"].isin(blend_seasons)].copy()
    test = merged[merged["SEASON"].isin(test_seasons)].copy()
    if model_train.empty or blend.empty or test.empty:
        raise ValueError("Model-train, blend, and held-out test seasons must all contain rows.")

    def component_probabilities(
        frame: pd.DataFrame,
        base_model,
        context_model,
        recent_model,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            base_probability = symmetric_model_probability(
                base_model,
                frame[base_features],
                _reverse_base_frame(frame, base_features),
            )
            context_probability = _symmetric_probability(
                context_model, frame, EXPERIMENT_SERIES_FEATURE_COLUMNS
            )
            recent_probability = _symmetric_probability(
                recent_model, frame, TEAM_RECENT_FEATURE_COLUMNS
            )
        return base_probability, context_probability, recent_probability

    out_of_fold_rows = []
    for season in blend_seasons:
        fold_train = model_train[model_train["SEASON"].ne(season)]
        fold_validation = model_train[model_train["SEASON"].eq(season)]
        if fold_train.empty or fold_validation.empty:
            continue
        print(f"Fitting out-of-season blend fold: {season}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fold_base_model = clone(base_template).fit(
                fold_train[base_features],
                fold_train["TEAM_A_WON"].astype(int),
            )
        fold_context_model = _fit_symmetric_logistic(
            fold_train, EXPERIMENT_SERIES_FEATURE_COLUMNS
        )
        fold_recent_model = _fit_symmetric_logistic(
            fold_train, TEAM_RECENT_FEATURE_COLUMNS
        )
        fold_base, fold_context, fold_recent = component_probabilities(
            fold_validation,
            fold_base_model,
            fold_context_model,
            fold_recent_model,
        )
        out_of_fold_rows.append(
            pd.DataFrame(
                {
                    "base_probability": fold_base,
                    "context_probability": fold_context,
                    "recent_probability": fold_recent,
                    "target": fold_validation["TEAM_A_WON"].astype(int).to_numpy(),
                }
            )
        )
    if not out_of_fold_rows:
        raise ValueError("Could not create out-of-season blend predictions.")
    blend_predictions = pd.concat(out_of_fold_rows, ignore_index=True)
    base_context_blend = _fit_blend(
        [
            blend_predictions["base_probability"].to_numpy(),
            blend_predictions["context_probability"].to_numpy(),
        ],
        blend_predictions["target"],
    )
    full_blend = _fit_blend(
        [
            blend_predictions["base_probability"].to_numpy(),
            blend_predictions["context_probability"].to_numpy(),
            blend_predictions["recent_probability"].to_numpy(),
        ],
        blend_predictions["target"],
    )

    context_model = _fit_symmetric_logistic(
        model_train, EXPERIMENT_SERIES_FEATURE_COLUMNS
    )
    recent_model = _fit_symmetric_logistic(model_train, TEAM_RECENT_FEATURE_COLUMNS)
    base_probability, context_probability, recent_probability = component_probabilities(
        test,
        base_template,
        context_model,
        recent_model,
    )
    base_context_probability = _blend_probability(
        base_context_blend, [base_probability, context_probability]
    )
    full_probability = _blend_probability(
        full_blend, [base_probability, context_probability, recent_probability]
    )
    predictions = {
        "base_only": base_probability,
        "series_context_only": context_probability,
        "recent_form_only": recent_probability,
        "base_plus_series_context": base_context_probability,
        "base_plus_series_context_plus_recent_form": full_probability,
    }
    metrics = pd.DataFrame(
        [_metrics(name, test["TEAM_A_WON"], probability) for name, probability in predictions.items()]
    )
    metrics["model_train_seasons"] = ",".join(model_train_seasons)
    metrics["blend_seasons"] = ",".join(blend_seasons)
    metrics["test_seasons"] = ",".join(test_seasons)

    calibration = pd.DataFrame(
        [
            row
            for name, probability in predictions.items()
            for row in _calibration_rows(name, test["TEAM_A_WON"], probability)
        ]
    )
    classifier = recent_model.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {
            "feature": TEAM_RECENT_FEATURE_COLUMNS,
            "standardized_coefficient": classifier.coef_[0].astype(float),
        }
    )
    coefficients["absolute_coefficient"] = coefficients["standardized_coefficient"].abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)

    examples = test[
        ["SEASON", "GAME_ID", "GAME_DATE", "TEAM_A", "TEAM_B", "TEAM_A_WON"]
    ].copy()
    examples["base_plus_series_probability"] = base_context_probability
    examples["with_recent_form_probability"] = full_probability
    examples["recent_form_probability_delta"] = (
        examples["with_recent_form_probability"]
        - examples["base_plus_series_probability"]
    )
    examples = examples.reindex(
        examples["recent_form_probability_delta"].abs().sort_values(ascending=False).index
    ).head(20)

    feature_summary = pd.DataFrame(
        [
            {
                "feature": column,
                "group": (
                    "team_recent_form"
                    if column in TEAM_RECENT_FEATURE_COLUMNS
                    else "series_context"
                ),
                "non_null_rows": int(merged[column].notna().sum()),
                "missing_rows": int(merged[column].isna().sum()),
                "mean": float(pd.to_numeric(merged[column], errors="coerce").mean()),
                "std": float(pd.to_numeric(merged[column], errors="coerce").std()),
            }
            for column in [*TEAM_RECENT_FEATURE_COLUMNS, *EXPERIMENT_SERIES_FEATURE_COLUMNS]
        ]
    )

    baseline_metrics = metrics.set_index("model").loc["base_plus_series_context"]
    full_metrics = metrics.set_index("model").loc[
        "base_plus_series_context_plus_recent_form"
    ]
    brier_improved = full_metrics["brier_score"] < baseline_metrics["brier_score"]
    log_loss_improved = full_metrics["log_loss"] < baseline_metrics["log_loss"]
    base_only_metrics = metrics.set_index("model").loc["base_only"]
    beats_base_brier = full_metrics["brier_score"] < base_only_metrics["brier_score"]
    beats_base_log_loss = full_metrics["log_loss"] < base_only_metrics["log_loss"]
    recommendation = (
        "DEPLOY CANDIDATE: recent form improved held-out probability quality over both "
        "the base model and the base-plus-series layer. Review stability before deployment."
        if (brier_improved or log_loss_improved)
        and (beats_base_brier or beats_base_log_loss)
        else (
            "DO NOT DEPLOY: recent form did not improve held-out probability quality "
            "over both the existing base model and series-context layer."
        )
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recent.to_csv(output_dir / "recent_form_experiment_features.csv", index=False)
    metrics.to_csv(output_dir / "recent_form_experiment_metrics.csv", index=False)
    calibration.to_csv(output_dir / "recent_form_experiment_calibration.csv", index=False)
    coefficients.to_csv(output_dir / "recent_form_experiment_coefficients.csv", index=False)
    examples.to_csv(output_dir / "recent_form_experiment_examples.csv", index=False)
    feature_summary.to_csv(output_dir / "recent_form_experiment_feature_summary.csv", index=False)
    (output_dir / "recent_form_experiment_recommendation.txt").write_text(
        recommendation + "\n",
        encoding="utf-8",
    )
    return metrics, recommendation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline held-out NBA playoff recent-form experiment"
    )
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--base-training-path", default=DEFAULT_BASE_TRAINING_PATH)
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics, recommendation = run_recent_form_experiment(
        cache_dir=args.cache_dir,
        base_training_path=args.base_training_path,
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
    )
    print(metrics.to_string(index=False))
    print(recommendation)


if __name__ == "__main__":
    main()
