from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .model import (
    PREDICTION_MODE_CURRENT,
    get_model_entry_for_mode,
    load_model,
)
from .nba_data import season_range
from .recent_form_experiment import (
    _blend_probability,
    _calibration_rows,
    _fit_blend,
    _logistic_estimator,
    _normalize_game_id,
    _reverse_base_frame,
)
from .series_context import (
    SERIES_CONTEXT_FEATURE_COLUMNS,
    reverse_series_context_frame,
    symmetric_model_probability,
)

DEFAULT_TEAM_ROWS_PATH = Path(
    "data/processed/advanced_boxscore_v3_team_rows.csv"
)
DEFAULT_BASE_TRAINING_PATH = Path(
    "data/processed/training_matchups_regular_season.csv"
)
DEFAULT_SERIES_CONTEXT_PATH = Path(
    "data/processed/series_context_training.csv"
)
DEFAULT_BASE_MODEL_PATH = Path("models/playoff_predictor.joblib")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_DOC_PATH = Path(
    "docs/team_advanced_recent_deltas_experiment.md"
)
EXPERIMENT_SCHEMA_VERSION = "2026-06-13-team-advanced-recent-v1"

ADVANCED_METRICS = {
    "offensiveRating": "offensiveRating",
    "defensiveRating": "defensiveRating",
    "netRating": "netRating",
    "pace": "pace",
    "possessions": "possessions",
    "trueShootingPercentage": "trueShootingPercentage",
    "effectiveFieldGoalPercentage": "effectiveFieldGoalPercentage",
    "turnoverPercentage": "turnoverRatio",
    "reboundPercentage": "reboundPercentage",
    "assistRatio": "assistRatio",
    "pie": "PIE",
}
WINDOW_LABELS = {
    "last_1": 1,
    "last_2": 2,
    "last_3": 3,
    "series_to_date": None,
}
WINDOW_FEATURE_COLUMNS = [
    f"{label}_{feature_name}_diff"
    for label in WINDOW_LABELS
    for feature_name in ADVANCED_METRICS
]
BASELINE_FEATURE_COLUMNS = [
    "recent_offensiveRating_minus_prior_playoff_baseline_diff",
    "recent_defensiveRating_minus_prior_playoff_baseline_diff",
    "recent_netRating_minus_prior_playoff_baseline_diff",
    "recent_pace_minus_prior_playoff_baseline_diff",
    "recent_shooting_efficiency_minus_prior_playoff_baseline_diff",
]
TREND_FEATURE_COLUMNS = [
    "recent_netRating_trend",
    "recent_offensiveRating_trend",
    "recent_defensiveRating_trend",
    "recent_pace_trend",
    "recent_turnover_trend",
    "recent_pie_trend",
]
ADVANCED_FEATURE_COLUMNS = [
    *WINDOW_FEATURE_COLUMNS,
    *BASELINE_FEATURE_COLUMNS,
    *TREND_FEATURE_COLUMNS,
]


def _canonical_game_id(values: pd.Series) -> pd.Series:
    return _normalize_game_id(values).str.zfill(10)


def _mean(history: list[dict], column: str, window: int | None) -> float:
    selected = history[-window:] if window else history
    if not selected:
        return np.nan
    return float(
        pd.to_numeric(
            pd.Series([row[column] for row in selected]),
            errors="coerce",
        ).mean()
    )


def _trend(history: list[dict], column: str) -> float:
    selected = history[-3:]
    if len(selected) < 2:
        return np.nan
    values = pd.to_numeric(
        pd.Series([row[column] for row in selected]),
        errors="coerce",
    ).to_numpy(dtype=float)
    if np.isnan(values).any():
        return np.nan
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def _recent_baseline_delta(
    history: list[dict],
    column: str,
) -> float:
    if not history:
        return np.nan
    return _mean(history, column, 3) - _mean(history, column, None)


def build_advanced_feature_row(
    *,
    target: pd.Series,
    team_history: dict[int, list[dict]],
    series_history: dict[tuple[int, int], dict[int, list[dict]]],
) -> dict:
    """Build Team A minus Team B features from games strictly before target."""
    team_a_id = int(target["TEAM_A_ID"])
    team_b_id = int(target["TEAM_B_ID"])
    history_a = team_history.get(team_a_id, [])
    history_b = team_history.get(team_b_id, [])
    pair = tuple(sorted((team_a_id, team_b_id)))
    pair_history = series_history.get(pair, {})
    series_a = pair_history.get(team_a_id, [])
    series_b = pair_history.get(team_b_id, [])

    row = {
        "EXPERIMENT_SCHEMA_VERSION": EXPERIMENT_SCHEMA_VERSION,
        "SEASON": target["SEASON"],
        "GAME_ID": target["GAME_ID"],
        "GAME_DATE": target["GAME_DATE"],
        "TEAM_A_ID": team_a_id,
        "TEAM_A": target["TEAM_A"],
        "TEAM_B_ID": team_b_id,
        "TEAM_B": target["TEAM_B"],
        "TEAM_A_WON": int(target["TEAM_A_WON"]),
    }
    for label, window in WINDOW_LABELS.items():
        source_a = series_a if label == "series_to_date" else history_a
        source_b = series_b if label == "series_to_date" else history_b
        for feature_name, column in ADVANCED_METRICS.items():
            row[f"{label}_{feature_name}_diff"] = (
                _mean(source_a, column, window)
                - _mean(source_b, column, window)
            )

    for metric in (
        "offensiveRating",
        "defensiveRating",
        "netRating",
        "pace",
    ):
        row[
            f"recent_{metric}_minus_prior_playoff_baseline_diff"
        ] = (
            _recent_baseline_delta(history_a, metric)
            - _recent_baseline_delta(history_b, metric)
        )
    shooting_a = np.nanmean(
        [
            _recent_baseline_delta(
                history_a, "trueShootingPercentage"
            ),
            _recent_baseline_delta(
                history_a, "effectiveFieldGoalPercentage"
            ),
        ]
    ) if history_a else np.nan
    shooting_b = np.nanmean(
        [
            _recent_baseline_delta(
                history_b, "trueShootingPercentage"
            ),
            _recent_baseline_delta(
                history_b, "effectiveFieldGoalPercentage"
            ),
        ]
    ) if history_b else np.nan
    row[
        "recent_shooting_efficiency_minus_prior_playoff_baseline_diff"
    ] = shooting_a - shooting_b

    trend_metrics = {
        "recent_netRating_trend": "netRating",
        "recent_offensiveRating_trend": "offensiveRating",
        "recent_defensiveRating_trend": "defensiveRating",
        "recent_pace_trend": "pace",
        "recent_turnover_trend": "turnoverRatio",
        "recent_pie_trend": "PIE",
    }
    for feature, column in trend_metrics.items():
        row[feature] = _trend(history_a, column) - _trend(
            history_b, column
        )

    used_ids = sorted(
        {
            prior["gameId"]
            for prior in [*history_a[-3:], *history_b[-3:]]
        }
    )
    used_dates = [
        pd.Timestamp(prior["gameDate"])
        for prior in [*history_a[-3:], *history_b[-3:]]
    ]
    row.update(
        {
            "game_number_audit": len(series_a) + 1,
            "v3_games_used_in_rolling_window": ",".join(used_ids),
            "latest_v3_game_date_used": (
                max(used_dates) if used_dates else pd.NaT
            ),
            "target_game_excluded": str(target["GAME_ID"])
            not in used_ids,
            "postgame_state_excluded": True,
        }
    )
    return row


def build_advanced_dataset(
    base_frame: pd.DataFrame,
    team_rows: pd.DataFrame,
) -> pd.DataFrame:
    base = base_frame.copy()
    advanced = team_rows.copy()
    base["GAME_ID"] = _canonical_game_id(base["GAME_ID"])
    base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
    advanced["gameId"] = _canonical_game_id(advanced["gameId"])
    advanced["gameDate"] = pd.to_datetime(advanced["gameDate"])
    advanced["teamId"] = advanced["teamId"].astype(int)
    advanced_lookup = {
        (str(row.season), str(row.gameId), int(row.teamId)): row._asdict()
        for row in advanced.itertuples(index=False)
    }

    rows = []
    for season, targets in base.groupby("SEASON", sort=True):
        print(f"Building leakage-safe V3 advanced rows: {season}")
        team_history: dict[int, list[dict]] = {}
        series_history: dict[
            tuple[int, int], dict[int, list[dict]]
        ] = {}
        targets = targets.sort_values(
            ["GAME_DATE", "GAME_ID"]
        ).reset_index(drop=True)
        for _, target in targets.iterrows():
            row = build_advanced_feature_row(
                target=target,
                team_history=team_history,
                series_history=series_history,
            )
            rows.append(row)

            team_a_id = int(target["TEAM_A_ID"])
            team_b_id = int(target["TEAM_B_ID"])
            pair = tuple(sorted((team_a_id, team_b_id)))
            pair_history = series_history.setdefault(pair, {})
            for team_id in (team_a_id, team_b_id):
                key = (str(season), str(target["GAME_ID"]), team_id)
                game_stats = advanced_lookup.get(key)
                if game_stats is None:
                    raise ValueError(
                        f"Missing V3 team row for {key}"
                    )
                team_history.setdefault(team_id, []).append(game_stats)
                pair_history.setdefault(team_id, []).append(game_stats)
    return pd.DataFrame(rows)


def reverse_advanced_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reversed_frame = frame.copy()
    for column in ADVANCED_FEATURE_COLUMNS:
        reversed_frame[column] = -pd.to_numeric(
            frame[column], errors="coerce"
        )
    return reversed_frame


def _fit_advanced_model(frame: pd.DataFrame):
    direct = frame[ADVANCED_FEATURE_COLUMNS].reset_index(drop=True)
    reverse = reverse_advanced_frame(frame)[
        ADVANCED_FEATURE_COLUMNS
    ].reset_index(drop=True)
    target = frame["TEAM_A_WON"].astype(int).reset_index(drop=True)
    augmented = pd.concat([direct, reverse], ignore_index=True)
    augmented_target = pd.concat(
        [target, 1 - target], ignore_index=True
    )
    return _logistic_estimator().fit(augmented, augmented_target)


def _fit_series_model(frame: pd.DataFrame):
    direct = frame[SERIES_CONTEXT_FEATURE_COLUMNS].reset_index(
        drop=True
    )
    reverse = reverse_series_context_frame(direct)
    target = frame["TEAM_A_WON"].astype(int).reset_index(drop=True)
    return _logistic_estimator().fit(
        pd.concat([direct, reverse], ignore_index=True),
        pd.concat([target, 1 - target], ignore_index=True),
    )


def _series_probability(model, frame: pd.DataFrame) -> np.ndarray:
    direct = frame[SERIES_CONTEXT_FEATURE_COLUMNS]
    return symmetric_model_probability(
        model,
        direct,
        reverse_series_context_frame(direct),
    )


def _advanced_probability(model, frame: pd.DataFrame) -> np.ndarray:
    return symmetric_model_probability(
        model,
        frame[ADVANCED_FEATURE_COLUMNS],
        reverse_advanced_frame(frame)[ADVANCED_FEATURE_COLUMNS],
    )


def _metrics(
    name: str,
    target: pd.Series,
    probability: np.ndarray,
) -> dict:
    predicted = (np.asarray(probability) >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier_score": float(
            brier_score_loss(target, probability)
        ),
        "log_loss": float(log_loss(target, probability)),
        "accuracy": float(accuracy_score(target, predicted)),
        "f1": float(f1_score(target, predicted)),
    }


def _merge_inputs(
    base: pd.DataFrame,
    advanced: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    for frame in (base, advanced, context):
        frame["GAME_ID"] = _canonical_game_id(frame["GAME_ID"])
    base = base.drop(
        columns=[
            column
            for column in SERIES_CONTEXT_FEATURE_COLUMNS
            if column in base
        ],
        errors="ignore",
    )
    metadata = [
        "SEASON",
        "GAME_ID",
        *ADVANCED_FEATURE_COLUMNS,
        "game_number_audit",
        "v3_games_used_in_rolling_window",
        "latest_v3_game_date_used",
        "target_game_excluded",
        "postgame_state_excluded",
    ]
    return (
        base.merge(
            advanced[metadata],
            on=["SEASON", "GAME_ID"],
            validate="one_to_one",
        )
        .merge(
            context[
                [
                    "SEASON",
                    "GAME_ID",
                    *SERIES_CONTEXT_FEATURE_COLUMNS,
                ]
            ],
            on=["SEASON", "GAME_ID"],
            validate="one_to_one",
        )
    )


def _component_probabilities(
    frame: pd.DataFrame,
    *,
    base_model,
    base_features: list[str],
    series_model,
    advanced_model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_probability = symmetric_model_probability(
            base_model,
            frame[base_features],
            _reverse_base_frame(frame, base_features),
        )
        series_probability = _series_probability(series_model, frame)
        advanced_probability = _advanced_probability(
            advanced_model, frame
        )
    return base_probability, series_probability, advanced_probability


def _write_doc(
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    recommendation: str,
    audit: pd.DataFrame,
    output_path: Path,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.select_dtypes(include="number").columns:
            display[column] = display[column].map(
                lambda value: f"{value:.4f}"
                if pd.notna(value)
                else ""
            )
        headers = [str(column) for column in display.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        for values in display.astype(str).itertuples(
            index=False, name=None
        ):
            lines.append(
                "| " + " | ".join(values) + " |"
            )
        return "\n".join(lines)

    lines = [
        "# Team Advanced Recent Deltas Experiment",
        "",
        "Offline held-out experiment only. No production model or application behavior changed.",
        "",
        "## Held-Out Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Strongest Advanced Features",
        "",
        markdown_table(coefficients.head(10)),
        "",
        "## Leakage Audit",
        "",
        markdown_table(audit),
        "",
        "## Decision",
        "",
        recommendation,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    team_rows_path: str | Path = DEFAULT_TEAM_ROWS_PATH,
    base_training_path: str | Path = DEFAULT_BASE_TRAINING_PATH,
    series_context_path: str | Path = DEFAULT_SERIES_CONTEXT_PATH,
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    train_seasons: list[str] | None = None,
    blend_seasons: list[str] | None = None,
    test_seasons: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    train_seasons = train_seasons or season_range(
        "2015-16", "2022-23"
    )
    blend_seasons = blend_seasons or season_range(
        "2018-19", "2022-23"
    )
    test_seasons = test_seasons or ["2023-24", "2024-25"]

    base = pd.read_csv(base_training_path)
    team_rows = pd.read_csv(team_rows_path)
    context = pd.read_csv(series_context_path)
    advanced = build_advanced_dataset(base, team_rows)
    merged = _merge_inputs(base, advanced, context)
    train = merged[merged["SEASON"].isin(train_seasons)].copy()
    test = merged[merged["SEASON"].isin(test_seasons)].copy()

    artifact = load_model(base_model_path)
    entry = get_model_entry_for_mode(
        artifact, PREDICTION_MODE_CURRENT
    )
    base_template = entry["pipeline"]
    base_features = list(entry["feature_columns"])

    out_of_fold = []
    for season in blend_seasons:
        fold_train = train[train["SEASON"].ne(season)]
        validation = train[train["SEASON"].eq(season)]
        if fold_train.empty or validation.empty:
            continue
        print(f"Fitting advanced blend fold: {season}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fold_base = clone(base_template).fit(
                fold_train[base_features],
                fold_train["TEAM_A_WON"].astype(int),
            )
        fold_series = _fit_series_model(fold_train)
        fold_advanced = _fit_advanced_model(fold_train)
        base_probability, series_probability, advanced_probability = (
            _component_probabilities(
                validation,
                base_model=fold_base,
                base_features=base_features,
                series_model=fold_series,
                advanced_model=fold_advanced,
            )
        )
        out_of_fold.append(
            pd.DataFrame(
                {
                    "base": base_probability,
                    "series": series_probability,
                    "advanced": advanced_probability,
                    "target": validation[
                        "TEAM_A_WON"
                    ].astype(int).to_numpy(),
                }
            )
        )
    blend = pd.concat(out_of_fold, ignore_index=True)
    base_advanced_blend = _fit_blend(
        [blend["base"], blend["advanced"]], blend["target"]
    )
    base_series_blend = _fit_blend(
        [blend["base"], blend["series"]], blend["target"]
    )
    full_blend = _fit_blend(
        [blend["base"], blend["series"], blend["advanced"]],
        blend["target"],
    )

    series_model = _fit_series_model(train)
    advanced_model = _fit_advanced_model(train)
    base_probability, series_probability, advanced_probability = (
        _component_probabilities(
            test,
            base_model=base_template,
            base_features=base_features,
            series_model=series_model,
            advanced_model=advanced_model,
        )
    )
    base_advanced_probability = _blend_probability(
        base_advanced_blend,
        [base_probability, advanced_probability],
    )
    base_series_probability = _blend_probability(
        base_series_blend,
        [base_probability, series_probability],
    )
    full_probability = _blend_probability(
        full_blend,
        [base_probability, series_probability, advanced_probability],
    )
    predictions = {
        "base_only": base_probability,
        "series_context_only": series_probability,
        "team_advanced_only": advanced_probability,
        "base_plus_team_advanced": base_advanced_probability,
        "base_plus_series": base_series_probability,
        "base_plus_series_plus_team_advanced": full_probability,
    }
    metrics = pd.DataFrame(
        [
            _metrics(name, test["TEAM_A_WON"], probability)
            for name, probability in predictions.items()
        ]
    )
    metrics["train_seasons"] = ",".join(train_seasons)
    metrics["blend_seasons"] = ",".join(blend_seasons)
    metrics["test_seasons"] = ",".join(test_seasons)

    calibration = pd.DataFrame(
        [
            row
            for name, probability in predictions.items()
            for row in _calibration_rows(
                name, test["TEAM_A_WON"], probability
            )
        ]
    )
    classifier = advanced_model.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {
            "feature": ADVANCED_FEATURE_COLUMNS,
            "standardized_coefficient": classifier.coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients[
        "standardized_coefficient"
    ].abs()
    coefficients = coefficients.sort_values(
        "absolute_coefficient", ascending=False
    )

    examples = test[
        [
            "SEASON",
            "GAME_ID",
            "GAME_DATE",
            "TEAM_A",
            "TEAM_B",
            "TEAM_A_WON",
        ]
    ].copy()
    for name, probability in predictions.items():
        examples[f"{name}_probability"] = probability
    examples["advanced_delta_vs_base"] = (
        base_advanced_probability - base_probability
    )
    examples["full_delta_vs_base_series"] = (
        full_probability - base_series_probability
    )
    examples = examples.reindex(
        examples["full_delta_vs_base_series"]
        .abs()
        .sort_values(ascending=False)
        .index
    ).head(25)

    audit = merged[
        merged["game_number_audit"].gt(1)
    ][
        [
            "SEASON",
            "GAME_ID",
            "GAME_DATE",
            "game_number_audit",
            "TEAM_A",
            "TEAM_B",
            "v3_games_used_in_rolling_window",
            "latest_v3_game_date_used",
            "target_game_excluded",
            "postgame_state_excluded",
        ]
    ].head(5)
    print("\nLeakage audit samples:")
    print(audit.to_string(index=False))

    metric_index = metrics.set_index("model")
    base_metrics = metric_index.loc["base_only"]
    advanced_candidates = metric_index.loc[
        [
            "base_plus_team_advanced",
            "base_plus_series_plus_team_advanced",
        ]
    ]
    improved = bool(
        (
            advanced_candidates["brier_score"]
            < base_metrics["brier_score"]
        ).any()
        or (
            advanced_candidates["log_loss"]
            < base_metrics["log_loss"]
        ).any()
    )
    recommendation = (
        "DEPLOY CANDIDATE: an advanced-feature combination improved held-out "
        "Brier score or log loss over base only. Require stability review "
        "before any production change."
        if improved
        else "DO NOT DEPLOY: advanced recent deltas did not improve held-out "
        "Brier score or log loss over base only. Keep research-only."
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "team_advanced_recent_deltas_metrics.csv",
        index=False,
    )
    coefficients.to_csv(
        output_dir / "team_advanced_recent_deltas_coefficients.csv",
        index=False,
    )
    calibration.to_csv(
        output_dir / "team_advanced_recent_deltas_calibration.csv",
        index=False,
    )
    examples.to_csv(
        output_dir / "team_advanced_recent_deltas_examples.csv",
        index=False,
    )
    audit.to_csv(
        output_dir / "team_advanced_recent_deltas_leakage_audit.csv",
        index=False,
    )
    _write_doc(
        metrics,
        coefficients,
        recommendation,
        audit,
        Path(doc_path),
    )
    return metrics, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline V3 team advanced recent-deltas experiment"
    )
    parser.add_argument(
        "--team-rows-path", default=DEFAULT_TEAM_ROWS_PATH
    )
    parser.add_argument(
        "--base-training-path", default=DEFAULT_BASE_TRAINING_PATH
    )
    parser.add_argument(
        "--series-context-path", default=DEFAULT_SERIES_CONTEXT_PATH
    )
    parser.add_argument(
        "--base-model-path", default=DEFAULT_BASE_MODEL_PATH
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()
    metrics, recommendation = run_experiment(
        team_rows_path=args.team_rows_path,
        base_training_path=args.base_training_path,
        series_context_path=args.series_context_path,
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        doc_path=args.doc_path,
    )
    print(metrics.to_string(index=False))
    print(recommendation)


if __name__ == "__main__":
    main()
