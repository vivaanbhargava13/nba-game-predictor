from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import PREDICTION_MODE_CURRENT, get_model_entry_for_mode, load_model
from .momentum_state_experiment import (
    _playoff_margin_rows,
    build_cross_fitted_expectations,
    series_win_probability,
)
from .nba_data import season_range
from .recent_form_experiment import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_BASE_TRAINING_PATH,
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_SERIES_FEATURE_COLUMNS,
    TEAM_RECENT_FEATURE_COLUMNS,
    _fit_symmetric_logistic,
    _normalize_game_id,
    _reverse_base_frame,
    _symmetric_probability,
)
from .series_context import symmetric_model_probability

UPSET_THRESHOLDS = (0.50, 0.45, 0.40)
MIN_ROC_GAMES = 30
UPSET_FEATURE_COLUMNS = [
    "previous_game_residual_win",
    "previous_game_residual_margin",
    "previous_game_series_win_probability_swing",
    "upset_winner_home_or_road",
    "upset_winner_seed_gap",
    "upset_winner_was_trailing_series",
    "upset_winner_was_facing_elimination",
    "previous_game_margin",
    "prior_base_probability_of_upset_winner",
    "next_game_changes_venue",
    "upset_winner_is_now_home",
    "upset_winner_can_close_series",
    "upset_loser_is_facing_elimination",
]


def previous_winner_was_lower_probability_team(
    winner_probability: float,
    threshold: float = 0.50,
) -> bool:
    return float(winner_probability) < float(threshold)


def _logit(values) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _response_model() -> Pipeline:
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


def _response_design(
    frame: pd.DataFrame,
    include_series: bool,
) -> pd.DataFrame:
    design = frame[UPSET_FEATURE_COLUMNS].copy()
    design.insert(0, "base_logit", _logit(frame["base_probability_upset_winner"]))
    if include_series:
        design.insert(
            1,
            "series_logit",
            _logit(frame["series_probability_upset_winner"]),
        )
    return design


def _fit_response_model(frame: pd.DataFrame, include_series: bool):
    return _response_model().fit(
        _response_design(frame, include_series),
        frame["upset_winner_won_next_game"].astype(int),
    )


def _response_probability(model, frame: pd.DataFrame, include_series: bool) -> np.ndarray:
    return model.predict_proba(_response_design(frame, include_series))[:, 1]


def _margin_bucket(margin: float) -> str:
    if margin <= 5:
        return "close_win"
    if margin >= 15:
        return "blowout_win"
    return "normal_win"


def build_upset_response_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Create next-game rows from the immediately preceding series game only."""
    rows = []
    for (_season, _team_a, _team_b), series in frame.groupby(
        ["SEASON", "TEAM_A_ID", "TEAM_B_ID"], sort=False
    ):
        series = series.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        team_a_wins = 0
        team_b_wins = 0
        previous = None
        for _, game in series.iterrows():
            if previous is not None:
                upset_winner_is_a = bool(previous["TEAM_A_WON"])
                winner_probability = (
                    float(previous["EXPECTED_WIN_PROBABILITY_A"])
                    if upset_winner_is_a
                    else 1 - float(previous["EXPECTED_WIN_PROBABILITY_A"])
                )
                winner_margin = abs(float(previous["TEAM_A_MARGIN"]))
                expected_margin_winner = (
                    float(previous["EXPECTED_MARGIN_A"])
                    if upset_winner_is_a
                    else -float(previous["EXPECTED_MARGIN_A"])
                )
                wins_before_winner = (
                    previous["team_a_wins_before"]
                    if upset_winner_is_a
                    else previous["team_b_wins_before"]
                )
                wins_before_loser = (
                    previous["team_b_wins_before"]
                    if upset_winner_is_a
                    else previous["team_a_wins_before"]
                )
                wins_after_winner = wins_before_winner + 1
                wins_after_loser = wins_before_loser
                series_before = series_win_probability(
                    winner_probability,
                    wins_before_winner,
                    wins_before_loser,
                )
                series_after = series_win_probability(
                    winner_probability,
                    wins_after_winner,
                    wins_after_loser,
                )
                previous_winner_home = (
                    int(previous["TEAM_A_HOME"])
                    if upset_winner_is_a
                    else 1 - int(previous["TEAM_A_HOME"])
                )
                next_winner_home = (
                    int(game["TEAM_A_HOME"])
                    if upset_winner_is_a
                    else 1 - int(game["TEAM_A_HOME"])
                )
                winner_seed = float(
                    previous["seed_A"] if upset_winner_is_a else previous["seed_B"]
                )
                loser_seed = float(
                    previous["seed_B"] if upset_winner_is_a else previous["seed_A"]
                )
                base_probability = (
                    float(game["EXPECTED_WIN_PROBABILITY_A"])
                    if upset_winner_is_a
                    else 1 - float(game["EXPECTED_WIN_PROBABILITY_A"])
                )
                context_probability = (
                    float(game["SERIES_CONTEXT_PROBABILITY_A"])
                    if upset_winner_is_a
                    else 1 - float(game["SERIES_CONTEXT_PROBABILITY_A"])
                )
                rows.append(
                    {
                        "SEASON": game["SEASON"],
                        "GAME_ID": str(game["GAME_ID"]),
                        "GAME_DATE": game["GAME_DATE"],
                        "upset_winner": (
                            game["TEAM_A"] if upset_winner_is_a else game["TEAM_B"]
                        ),
                        "upset_loser": (
                            game["TEAM_B"] if upset_winner_is_a else game["TEAM_A"]
                        ),
                        "previous_game_id": str(previous["GAME_ID"]),
                        "previous_game_result_excluded": (
                            str(game["GAME_ID"]) != str(previous["GAME_ID"])
                        ),
                        "previous_winner_probability": winner_probability,
                        "previous_game_residual_win": 1 - winner_probability,
                        "previous_game_residual_margin": (
                            winner_margin - expected_margin_winner
                        ),
                        "previous_game_series_win_probability_swing": (
                            series_after - series_before
                        ),
                        "upset_winner_home_or_road": previous_winner_home,
                        "upset_winner_seed_gap": winner_seed - loser_seed,
                        "upset_winner_was_trailing_series": int(
                            wins_before_winner < wins_before_loser
                        ),
                        "upset_winner_was_facing_elimination": int(
                            wins_before_loser == 3
                        ),
                        "previous_game_margin": winner_margin,
                        "prior_base_probability_of_upset_winner": winner_probability,
                        "next_game_changes_venue": int(
                            previous_winner_home != next_winner_home
                        ),
                        "upset_winner_is_now_home": next_winner_home,
                        "upset_winner_can_close_series": int(wins_after_winner == 3),
                        "upset_loser_is_facing_elimination": int(
                            wins_after_winner == 3
                        ),
                        "margin_bucket": _margin_bucket(winner_margin),
                        "base_probability_upset_winner": base_probability,
                        "series_probability_upset_winner": context_probability,
                        "upset_winner_won_next_game": int(
                            bool(game["TEAM_A_WON"]) == upset_winner_is_a
                        ),
                    }
                )

            team_a_won = int(game["TEAM_A_WON"])
            previous = game.to_dict()
            previous["team_a_wins_before"] = team_a_wins
            previous["team_b_wins_before"] = team_b_wins
            team_a_wins += team_a_won
            team_b_wins += 1 - team_a_won
    return pd.DataFrame(rows)


def _cross_fitted_series_probabilities(
    frame: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
) -> pd.DataFrame:
    rows = []
    train = frame[frame["SEASON"].isin(train_seasons)]
    for season in train_seasons:
        fold_train = train[train["SEASON"].ne(season)]
        validation = train[train["SEASON"].eq(season)]
        model = _fit_symmetric_logistic(
            fold_train, EXPERIMENT_SERIES_FEATURE_COLUMNS
        )
        rows.append(
            validation[["SEASON", "GAME_ID"]].assign(
                SERIES_CONTEXT_PROBABILITY_A=_symmetric_probability(
                    model,
                    validation,
                    EXPERIMENT_SERIES_FEATURE_COLUMNS,
                )
            )
        )
    model = _fit_symmetric_logistic(train, EXPERIMENT_SERIES_FEATURE_COLUMNS)
    heldout = frame[frame["SEASON"].isin(test_seasons)]
    rows.append(
        heldout[["SEASON", "GAME_ID"]].assign(
            SERIES_CONTEXT_PROBABILITY_A=_symmetric_probability(
                model,
                heldout,
                EXPERIMENT_SERIES_FEATURE_COLUMNS,
            )
        )
    )
    return pd.concat(rows, ignore_index=True)


def _metrics(
    *,
    threshold: float,
    margin_bucket: str,
    scope: str,
    model: str,
    target: pd.Series,
    probability: np.ndarray,
    reference: np.ndarray,
) -> dict:
    n_games = len(target)
    class_counts = target.value_counts()
    roc_auc = np.nan
    warning = ""
    if n_games >= MIN_ROC_GAMES and len(class_counts) == 2 and class_counts.min() >= 5:
        roc_auc = float(roc_auc_score(target, probability))
    else:
        warning = f"small_sample_roc_withheld_n_{n_games}"
    return {
        "threshold": threshold,
        "margin_bucket": margin_bucket,
        "scope": scope,
        "model": model,
        "n_games": n_games,
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": roc_auc,
        "average_probability_movement": float(
            np.mean(np.abs(probability - reference))
        ),
        "small_sample_warning": warning,
    }


def _calibration_rows(
    threshold: float,
    model: str,
    target: pd.Series,
    probability: np.ndarray,
) -> list[dict]:
    bucket = pd.cut(
        probability,
        bins=np.linspace(0, 1, 11),
        labels=False,
        include_lowest=True,
    )
    frame = pd.DataFrame(
        {"bucket": bucket, "target": target.to_numpy(), "probability": probability}
    )
    return [
        {
            "threshold": threshold,
            "model": model,
            "bucket": int(index),
            "count": len(group),
            "mean_probability": float(group["probability"].mean()),
            "observed_win_rate": float(group["target"].mean()),
        }
        for index, group in frame.groupby("bucket")
    ]


def run_upset_response_experiment(
    *,
    base_training_path: str | Path = DEFAULT_BASE_TRAINING_PATH,
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    recent_features_path: str | Path = (
        "data/processed/recent_form_experiment_features.csv"
    ),
    broad_predictions_path: str | Path = (
        "data/processed/momentum_state_experiment_predictions.csv"
    ),
    cache_dir: str | Path = "data/raw",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train_seasons = season_range("2015-16", "2022-23")
    test_seasons = ["2023-24", "2024-25"]
    all_seasons = [*train_seasons, *test_seasons]

    base = pd.read_csv(base_training_path)
    base["GAME_ID"] = _normalize_game_id(base["GAME_ID"])
    margins = _playoff_margin_rows(all_seasons, Path(cache_dir))
    base = base.merge(
        margins[
            ["SEASON", "GAME_ID", "TEAM_A_MARGIN", "TEAM_A_HOME"]
        ],
        on=["SEASON", "GAME_ID"],
        validate="one_to_one",
    )
    recent = pd.read_csv(recent_features_path)
    recent["GAME_ID"] = _normalize_game_id(recent["GAME_ID"])
    base = base.drop(
        columns=[
            column
            for column in [
                *TEAM_RECENT_FEATURE_COLUMNS,
                *EXPERIMENT_SERIES_FEATURE_COLUMNS,
            ]
            if column in base.columns
        ],
        errors="ignore",
    ).merge(
        recent[
            [
                "SEASON",
                "GAME_ID",
                *TEAM_RECENT_FEATURE_COLUMNS,
                *EXPERIMENT_SERIES_FEATURE_COLUMNS,
            ]
        ],
        on=["SEASON", "GAME_ID"],
        validate="one_to_one",
    )
    artifact = load_model(base_model_path)
    entry = get_model_entry_for_mode(artifact, PREDICTION_MODE_CURRENT)
    base_model = entry["pipeline"]
    base_features = list(entry["feature_columns"])
    expectations = build_cross_fitted_expectations(
        base,
        base_model=base_model,
        base_features=base_features,
        train_seasons=train_seasons,
        test_seasons=test_seasons,
    )
    base = base.merge(expectations, on=["SEASON", "GAME_ID"], validate="one_to_one")
    series_probabilities = _cross_fitted_series_probabilities(
        base, train_seasons, test_seasons
    )
    base = base.merge(
        series_probabilities,
        on=["SEASON", "GAME_ID"],
        validate="one_to_one",
    )
    response = build_upset_response_rows(base)

    broad = pd.read_csv(broad_predictions_path)
    broad["GAME_ID"] = _normalize_game_id(broad["GAME_ID"])
    broad_columns = [
        "SEASON",
        "GAME_ID",
        "TEAM_A",
        "base_probability",
        "base_plus_series_probability",
        "base_plus_momentum_probability",
    ]
    response = response.merge(
        broad[broad_columns],
        on=["SEASON", "GAME_ID"],
        how="left",
        validate="many_to_one",
    )
    is_upset_winner_a = response["upset_winner"].eq(response["TEAM_A"])
    for source, target in [
        ("base_probability", "broad_base_probability_upset_winner"),
        (
            "base_plus_series_probability",
            "broad_series_probability_upset_winner",
        ),
        (
            "base_plus_momentum_probability",
            "broad_momentum_probability_upset_winner",
        ),
    ]:
        response[target] = np.where(
            is_upset_winner_a,
            response[source],
            1 - response[source],
        )

    train = response[response["SEASON"].isin(train_seasons)].copy()
    test = response[response["SEASON"].isin(test_seasons)].copy()
    metrics_rows = []
    calibration_rows = []
    example_rows = []

    for threshold in UPSET_THRESHOLDS:
        train_gate = train[
            train["previous_winner_probability"].lt(threshold)
        ].copy()
        test_gate = test[
            test["previous_winner_probability"].lt(threshold)
        ].copy()
        if train_gate.empty or test_gate.empty:
            continue
        base_response = _fit_response_model(train_gate, include_series=False)
        series_response = _fit_response_model(train_gate, include_series=True)
        base_response_probability = _response_probability(
            base_response, test_gate, include_series=False
        )
        series_response_probability = _response_probability(
            series_response, test_gate, include_series=True
        )
        comparison_probabilities = {
            "base_only": test_gate["broad_base_probability_upset_winner"].to_numpy(),
            "base_plus_broad_momentum": test_gate[
                "broad_momentum_probability_upset_winner"
            ].to_numpy(),
            "base_plus_upset_response": base_response_probability,
            "base_plus_series": test_gate[
                "broad_series_probability_upset_winner"
            ].to_numpy(),
            "base_plus_series_plus_upset_response": series_response_probability,
        }
        target = test_gate["upset_winner_won_next_game"].astype(int)
        reference = comparison_probabilities["base_only"]
        for margin_bucket in ["all", "close_win", "normal_win", "blowout_win"]:
            mask = (
                np.ones(len(test_gate), dtype=bool)
                if margin_bucket == "all"
                else test_gate["margin_bucket"].eq(margin_bucket).to_numpy()
            )
            if not mask.any():
                continue
            for model_name, probability in comparison_probabilities.items():
                metrics_rows.append(
                    _metrics(
                        threshold=threshold,
                        margin_bucket=margin_bucket,
                        scope="gated_subgroup",
                        model=model_name,
                        target=target[mask],
                        probability=np.asarray(probability)[mask],
                        reference=reference[mask],
                    )
                )
        for model_name, probability in comparison_probabilities.items():
            calibration_rows.extend(
                _calibration_rows(threshold, model_name, target, probability)
            )

        test_policy = test.copy()
        policy_gate = test_policy["previous_winner_probability"].lt(threshold)
        policy_base = test_policy[
            "broad_base_probability_upset_winner"
        ].to_numpy()
        policy_probability = policy_base.copy()
        policy_probability[policy_gate.to_numpy()] = base_response_probability
        policy_target = test_policy["upset_winner_won_next_game"].astype(int)
        for model_name, probability in {
            "base_only": policy_base,
            "gated_upset_response_policy": policy_probability,
        }.items():
            metrics_rows.append(
                _metrics(
                    threshold=threshold,
                    margin_bucket="all",
                    scope="all_next_games_policy",
                    model=model_name,
                    target=policy_target,
                    probability=probability,
                    reference=policy_base,
                )
            )

        if threshold == 0.50:
            examples = test_gate[
                [
                    "SEASON",
                    "GAME_ID",
                    "GAME_DATE",
                    "upset_winner",
                    "upset_loser",
                    "upset_winner_won_next_game",
                    "previous_winner_probability",
                    "previous_game_margin",
                    "margin_bucket",
                ]
            ].copy()
            examples["base_probability"] = reference
            examples["upset_response_probability"] = base_response_probability
            examples["absolute_error_change"] = (
                examples["upset_response_probability"]
                - examples["upset_winner_won_next_game"]
            ).abs() - (
                examples["base_probability"]
                - examples["upset_winner_won_next_game"]
            ).abs()
            for effect, selected in [
                ("helped", examples.nsmallest(10, "absolute_error_change")),
                ("hurt", examples.nlargest(10, "absolute_error_change")),
            ]:
                selected = selected.copy()
                selected["effect"] = effect
                example_rows.extend(selected.to_dict(orient="records"))

    metrics = pd.DataFrame(metrics_rows)
    calibration = pd.DataFrame(calibration_rows)
    examples = pd.DataFrame(example_rows)
    gate_metrics = metrics[
        metrics["scope"].eq("gated_subgroup")
        & metrics["margin_bucket"].eq("all")
        & metrics["model"].eq("base_plus_upset_response")
    ].copy()
    policy_metrics = metrics[
        metrics["scope"].eq("all_next_games_policy")
    ].pivot(index="threshold", columns="model", values=["brier_score", "log_loss"])
    best = gate_metrics.sort_values(["brier_score", "log_loss"]).iloc[0]
    threshold = best["threshold"]
    policy_brier_improved = (
        policy_metrics.loc[threshold, ("brier_score", "gated_upset_response_policy")]
        < policy_metrics.loc[threshold, ("brier_score", "base_only")]
    )
    policy_log_loss_improved = (
        policy_metrics.loc[threshold, ("log_loss", "gated_upset_response_policy")]
        < policy_metrics.loc[threshold, ("log_loss", "base_only")]
    )
    if not (
        best["brier_score"]
        < metrics[
            metrics["scope"].eq("gated_subgroup")
            & metrics["margin_bucket"].eq("all")
            & metrics["threshold"].eq(threshold)
            & metrics["model"].eq("base_only")
        ]["brier_score"].iloc[0]
        or best["log_loss"]
        < metrics[
            metrics["scope"].eq("gated_subgroup")
            & metrics["margin_bucket"].eq("all")
            & metrics["threshold"].eq(threshold)
            & metrics["model"].eq("base_only")
        ]["log_loss"].iloc[0]
    ):
        recommendation = "no value"
    elif best["n_games"] < MIN_ROC_GAMES:
        recommendation = "promising but too small sample"
    elif policy_brier_improved or policy_log_loss_improved:
        recommendation = "production candidate"
    else:
        recommendation = "explanation-only value"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "upset_response_momentum_metrics.csv", index=False
    )
    examples.to_csv(
        output_dir / "upset_response_momentum_examples.csv", index=False
    )
    calibration.to_csv(
        output_dir / "upset_response_momentum_calibration.csv", index=False
    )
    print(metrics.to_string(index=False))
    print(f"Recommendation: {recommendation}")
    return metrics, examples, calibration, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline expectation-breaking playoff response experiment"
    )
    parser.add_argument("--base-training-path", default=DEFAULT_BASE_TRAINING_PATH)
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument(
        "--recent-features-path",
        default="data/processed/recent_form_experiment_features.csv",
    )
    parser.add_argument(
        "--broad-predictions-path",
        default="data/processed/momentum_state_experiment_predictions.csv",
    )
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_upset_response_experiment(
        base_training_path=args.base_training_path,
        base_model_path=args.base_model_path,
        recent_features_path=args.recent_features_path,
        broad_predictions_path=args.broad_predictions_path,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
