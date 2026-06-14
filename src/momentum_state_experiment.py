from __future__ import annotations

import argparse
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from .model import PREDICTION_MODE_CURRENT, get_model_entry_for_mode, load_model
from .nba_data import season_range
from .recent_form_experiment import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_BASE_TRAINING_PATH,
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_SERIES_FEATURE_COLUMNS,
    TEAM_RECENT_FEATURE_COLUMNS,
    _blend_probability,
    _calibration_rows,
    _fit_blend,
    _fit_symmetric_logistic,
    _logistic_estimator,
    _metrics,
    _normalize_game_id,
    _reverse_base_frame,
    _symmetric_probability,
)
from .series_context import symmetric_model_probability

MOMENTUM_SCHEMA_VERSION = "2026-06-12-surprise-adjusted-momentum-v1"
DECAY_VALUES = (0.50, 0.65, 0.80)
MOMENTUM_FEATURE_COLUMNS = [
    "expected_win_probability_before_previous_game_diff",
    "expected_margin_before_previous_game_diff",
    "actual_margin_previous_game_diff",
    "residual_margin_previous_game_diff",
    "residual_win_previous_game_diff",
    "ewma_residual_margin_2_diff",
    "ewma_residual_margin_3_diff",
    "ewma_residual_margin_series_diff",
    "ewma_residual_win_series_diff",
    *[
        f"ewma_residual_margin_series_decay_{str(decay).replace('.', '_')}_diff"
        for decay in DECAY_VALUES
    ],
    *[
        f"ewma_residual_win_series_decay_{str(decay).replace('.', '_')}_diff"
        for decay in DECAY_VALUES
    ],
    "consecutive_positive_residual_games_diff",
    "consecutive_negative_residual_games_diff",
    "positive_residual_share_series_diff",
    "last_2_positive_residual_count_diff",
    "last_3_positive_residual_count_diff",
    "series_win_prob_before_previous_game_diff",
    "series_win_prob_after_previous_game_diff",
    "series_win_prob_swing_previous_game_diff",
    "cumulative_series_win_prob_swing_diff",
    "closeout_swing_created_diff",
    "elimination_survival_swing_created_diff",
    "residual_margin_in_elimination_game_diff",
    "residual_margin_in_closeout_game_diff",
    "residual_margin_when_trailing_series_diff",
    "residual_margin_when_facing_elimination_diff",
    "comeback_win_over_expected_indicator_diff",
    "blown_game_under_expected_indicator_diff",
]


def _logit(values) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


@lru_cache(maxsize=None)
def series_win_probability(
    game_win_probability: float,
    team_wins: int,
    opponent_wins: int,
) -> float:
    """Best-of-seven win probability under a constant pregame strength estimate."""
    probability = min(1.0, max(0.0, float(game_win_probability)))
    if team_wins >= 4:
        return 1.0
    if opponent_wins >= 4:
        return 0.0
    return (
        probability
        * series_win_probability(probability, team_wins + 1, opponent_wins)
        + (1 - probability)
        * series_win_probability(probability, team_wins, opponent_wins + 1)
    )


def _playoff_margin_rows(seasons: list[str], cache_dir: Path) -> pd.DataFrame:
    rows = []
    for season in seasons:
        path = cache_dir / f"playoff_games_{season}_playoffs.csv"
        games = pd.read_csv(path)
        games["GAME_ID"] = _normalize_game_id(games["GAME_ID"])
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], format="mixed")
        for game_id, group in games.groupby("GAME_ID", sort=False):
            if len(group) != 2:
                continue
            ordered = group.sort_values("TEAM_ID").reset_index(drop=True)
            team_a, team_b = ordered.iloc[0], ordered.iloc[1]
            rows.append(
                {
                    "SEASON": season,
                    "GAME_ID": str(game_id),
                    "GAME_DATE": pd.Timestamp(team_a["GAME_DATE"]),
                    "TEAM_A_ID": int(team_a["TEAM_ID"]),
                    "TEAM_A": str(team_a["TEAM_ABBREVIATION"]),
                    "TEAM_B_ID": int(team_b["TEAM_ID"]),
                    "TEAM_B": str(team_b["TEAM_ABBREVIATION"]),
                    "TEAM_A_HOME": int("vs." in str(team_a["MATCHUP"])),
                    "TEAM_A_WON": int(str(team_a["WL"]) == "W"),
                    "TEAM_A_MARGIN": float(team_a["PTS"]) - float(team_b["PTS"]),
                }
            )
    return pd.DataFrame(rows)


def build_cross_fitted_expectations(
    base: pd.DataFrame,
    *,
    base_model,
    base_features: list[str],
    train_seasons: list[str],
    test_seasons: list[str],
) -> pd.DataFrame:
    """Create pregame expectations without fitting on the target season."""
    rows = []
    train = base[base["SEASON"].isin(train_seasons)].copy()
    for season in train_seasons:
        fold_train = train[train["SEASON"].ne(season)]
        validation = train[train["SEASON"].eq(season)]
        if fold_train.empty or validation.empty:
            continue
        print(f"Cross-fitting expected performance: {season}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model = clone(base_model).fit(
                fold_train[base_features],
                fold_train["TEAM_A_WON"].astype(int),
            )
            train_probability = symmetric_model_probability(
                model,
                fold_train[base_features],
                _reverse_base_frame(fold_train, base_features),
            )
            validation_probability = symmetric_model_probability(
                model,
                validation[base_features],
                _reverse_base_frame(validation, base_features),
            )
        margin_model = LinearRegression().fit(
            _logit(train_probability).reshape(-1, 1),
            fold_train["TEAM_A_MARGIN"].to_numpy(),
        )
        fold = validation[["SEASON", "GAME_ID"]].copy()
        fold["EXPECTED_WIN_PROBABILITY_A"] = validation_probability
        fold["EXPECTED_MARGIN_A"] = margin_model.predict(
            _logit(validation_probability).reshape(-1, 1)
        )
        rows.append(fold)

    heldout = base[base["SEASON"].isin(test_seasons)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        train_probability = symmetric_model_probability(
            base_model,
            train[base_features],
            _reverse_base_frame(train, base_features),
        )
        heldout_probability = symmetric_model_probability(
            base_model,
            heldout[base_features],
            _reverse_base_frame(heldout, base_features),
        )
    margin_model = LinearRegression().fit(
        _logit(train_probability).reshape(-1, 1),
        train["TEAM_A_MARGIN"].to_numpy(),
    )
    heldout_rows = heldout[["SEASON", "GAME_ID"]].copy()
    heldout_rows["EXPECTED_WIN_PROBABILITY_A"] = heldout_probability
    heldout_rows["EXPECTED_MARGIN_A"] = margin_model.predict(
        _logit(heldout_probability).reshape(-1, 1)
    )
    rows.append(heldout_rows)
    return pd.concat(rows, ignore_index=True)


def _ewma(values: list[float], decay: float, limit: int | None = None) -> float:
    selected = values[-limit:] if limit else values
    if not selected:
        return 0.0
    weight = 1.0
    weighted_sum = 0.0
    total_weight = 0.0
    for value in reversed(selected):
        weighted_sum += float(value) * weight
        total_weight += weight
        weight *= decay
    return weighted_sum / total_weight


def _streak(values: list[float], positive: bool) -> int:
    count = 0
    for value in reversed(values):
        if (value > 0) == positive and value != 0:
            count += 1
        else:
            break
    return count


def _team_momentum_state(history: list[dict], side: str) -> dict[str, float]:
    sign = 1.0 if side == "a" else -1.0
    win_key = "team_a_won"
    probability_key = "expected_probability_a"
    residual_margins = [sign * game["residual_margin_a"] for game in history]
    residual_wins = [sign * game["residual_win_a"] for game in history]
    latest = history[-1] if history else None
    latest_won = (
        float(latest[win_key]) if side == "a" else float(1 - latest[win_key])
    ) if latest else 0.0
    latest_probability = (
        float(latest[probability_key])
        if side == "a"
        else float(1 - latest[probability_key])
    ) if latest else 0.5
    latest_margin = sign * float(latest["team_a_margin"]) if latest else 0.0
    latest_expected_margin = sign * float(latest["expected_margin_a"]) if latest else 0.0
    series_before = (
        float(latest["series_probability_before_a"])
        if side == "a"
        else 1 - float(latest["series_probability_before_a"])
    ) if latest else 0.5
    series_after = (
        float(latest["series_probability_after_a"])
        if side == "a"
        else 1 - float(latest["series_probability_after_a"])
    ) if latest else 0.5
    swings = [
        float(game["series_probability_swing_a"]) * sign for game in history
    ]

    positive_count = sum(value > 0 for value in residual_margins)
    state = {
        "expected_win_probability_before_previous_game": latest_probability,
        "expected_margin_before_previous_game": latest_expected_margin,
        "actual_margin_previous_game": latest_margin,
        "residual_margin_previous_game": residual_margins[-1] if history else 0.0,
        "residual_win_previous_game": residual_wins[-1] if history else 0.0,
        "ewma_residual_margin_2": _ewma(residual_margins, 0.65, 2),
        "ewma_residual_margin_3": _ewma(residual_margins, 0.65, 3),
        "ewma_residual_margin_series": _ewma(residual_margins, 0.65),
        "ewma_residual_win_series": _ewma(residual_wins, 0.65),
        "consecutive_positive_residual_games": float(
            _streak(residual_margins, True)
        ),
        "consecutive_negative_residual_games": float(
            _streak(residual_margins, False)
        ),
        "positive_residual_share_series": (
            float(positive_count / len(history)) if history else 0.0
        ),
        "last_2_positive_residual_count": float(
            sum(value > 0 for value in residual_margins[-2:])
        ),
        "last_3_positive_residual_count": float(
            sum(value > 0 for value in residual_margins[-3:])
        ),
        "series_win_prob_before_previous_game": series_before,
        "series_win_prob_after_previous_game": series_after,
        "series_win_prob_swing_previous_game": (
            series_after - series_before if latest else 0.0
        ),
        "cumulative_series_win_prob_swing": float(sum(swings)),
        "closeout_swing_created": 0.0,
        "elimination_survival_swing_created": 0.0,
        "residual_margin_in_elimination_game": 0.0,
        "residual_margin_in_closeout_game": 0.0,
        "residual_margin_when_trailing_series": 0.0,
        "residual_margin_when_facing_elimination": 0.0,
        "comeback_win_over_expected_indicator": 0.0,
        "blown_game_under_expected_indicator": 0.0,
    }
    for decay in DECAY_VALUES:
        label = str(decay).replace(".", "_")
        state[f"ewma_residual_margin_series_decay_{label}"] = _ewma(
            residual_margins, decay
        )
        state[f"ewma_residual_win_series_decay_{label}"] = _ewma(
            residual_wins, decay
        )
    if latest:
        side_wins_before = (
            latest["team_a_wins_before"]
            if side == "a"
            else latest["team_b_wins_before"]
        )
        opponent_wins_before = (
            latest["team_b_wins_before"]
            if side == "a"
            else latest["team_a_wins_before"]
        )
        side_wins_after = side_wins_before + int(latest_won)
        residual_margin = residual_margins[-1]
        swing = series_after - series_before
        state["closeout_swing_created"] = (
            swing if latest_won and side_wins_after == 3 else 0.0
        )
        state["elimination_survival_swing_created"] = (
            swing if latest_won and opponent_wins_before == 3 else 0.0
        )
        state["residual_margin_in_elimination_game"] = (
            residual_margin if opponent_wins_before == 3 else 0.0
        )
        state["residual_margin_in_closeout_game"] = (
            residual_margin if side_wins_before == 3 else 0.0
        )
        state["residual_margin_when_trailing_series"] = (
            residual_margin if side_wins_before < opponent_wins_before else 0.0
        )
        state["residual_margin_when_facing_elimination"] = (
            residual_margin if opponent_wins_before == 3 else 0.0
        )
        state["comeback_win_over_expected_indicator"] = float(
            latest_won and latest_probability < 0.5
        )
        state["blown_game_under_expected_indicator"] = float(
            not latest_won and latest_probability > 0.5
        )
    return state


def build_momentum_features(history: list[dict]) -> dict[str, float]:
    team_a = _team_momentum_state(history, "a")
    team_b = _team_momentum_state(history, "b")
    return {
        f"{feature}_diff": float(team_a[feature] - team_b[feature])
        for feature in [column.removesuffix("_diff") for column in MOMENTUM_FEATURE_COLUMNS]
    }


def build_momentum_dataset(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, season_frame in base.groupby("SEASON", sort=True):
        print(f"Building surprise-adjusted momentum rows: {season}")
        for _pair, series in season_frame.groupby(
            ["TEAM_A_ID", "TEAM_B_ID"], sort=False
        ):
            series = series.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
            history: list[dict] = []
            team_a_wins = 0
            team_b_wins = 0
            for _, game in series.iterrows():
                team_a_state = _team_momentum_state(history, "a")
                team_b_state = _team_momentum_state(history, "b")
                previous = history[-1] if history else None
                team_a_home = int(game.get("TEAM_A_HOME", 0))
                row = {
                    "MOMENTUM_SCHEMA_VERSION": MOMENTUM_SCHEMA_VERSION,
                    "SEASON": season,
                    "GAME_ID": str(game["GAME_ID"]),
                    "GAME_DATE": pd.Timestamp(game["GAME_DATE"]),
                    "TEAM_A_ID": int(game["TEAM_A_ID"]),
                    "TEAM_A": game["TEAM_A"],
                    "TEAM_B_ID": int(game["TEAM_B_ID"]),
                    "TEAM_B": game["TEAM_B"],
                    "TEAM_A_WON": int(game["TEAM_A_WON"]),
                    **build_momentum_features(history),
                    "games_used_to_compute_momentum": ",".join(
                        prior["game_id"] for prior in history
                    ),
                    "latest_game_used": history[-1]["game_id"] if history else "",
                    "target_game_excluded": str(game["GAME_ID"])
                    not in {prior["game_id"] for prior in history},
                    "postgame_series_score_excluded": True,
                    "previous_feature_game_id": (
                        history[-1]["game_id"] if history else ""
                    ),
                    "pregame_series_score": f"{team_a_wins}-{team_b_wins}",
                    "game_number_audit": len(history) + 1,
                    "team_a_series_wins_audit": team_a_wins,
                    "team_b_series_wins_audit": team_b_wins,
                    "team_a_home_audit": team_a_home,
                    "previous_game_winner_audit": (
                        1 if previous and previous["team_a_won"] else
                        -1 if previous else 0
                    ),
                    "team_a_won_previous_two_audit": (
                        int(len(history) >= 2)
                        * int(sum(item["team_a_won"] for item in history[-2:]) == 2)
                    ),
                    "team_b_won_previous_two_audit": (
                        int(len(history) >= 2)
                        * int(sum(1 - item["team_a_won"] for item in history[-2:]) == 2)
                    ),
                    "previous_margin_audit": (
                        float(previous["team_a_margin"]) if previous else 0.0
                    ),
                    "previous_expected_probability_a_audit": (
                        float(previous["expected_probability_a"]) if previous else 0.5
                    ),
                    "previous_residual_margin_a_audit": (
                        float(previous["residual_margin_a"]) if previous else 0.0
                    ),
                    "team_a_last_two_positive_residuals_audit": int(
                        len(history) >= 2
                        and all(item["residual_margin_a"] > 0 for item in history[-2:])
                    ),
                    "team_b_last_two_positive_residuals_audit": int(
                        len(history) >= 2
                        and all(item["residual_margin_a"] < 0 for item in history[-2:])
                    ),
                    "team_a_momentum_state_audit": float(
                        team_a_state["ewma_residual_margin_series"]
                    ),
                    "team_b_momentum_state_audit": float(
                        team_b_state["ewma_residual_margin_series"]
                    ),
                    "home_court_swing_audit": int(
                        previous is not None
                        and int(previous["team_a_home"]) != team_a_home
                    ),
                }
                rows.append(row)

                expected_probability = float(game["EXPECTED_WIN_PROBABILITY_A"])
                expected_margin = float(game["EXPECTED_MARGIN_A"])
                actual_margin = float(game["TEAM_A_MARGIN"])
                team_a_won = int(game["TEAM_A_WON"])
                series_before = series_win_probability(
                    expected_probability, team_a_wins, team_b_wins
                )
                wins_after_a = team_a_wins + team_a_won
                wins_after_b = team_b_wins + (1 - team_a_won)
                series_after = series_win_probability(
                    expected_probability, wins_after_a, wins_after_b
                )
                history.append(
                    {
                        "game_id": str(game["GAME_ID"]),
                        "team_a_won": team_a_won,
                        "team_a_margin": actual_margin,
                        "expected_probability_a": expected_probability,
                        "expected_margin_a": expected_margin,
                        "residual_margin_a": actual_margin - expected_margin,
                        "residual_win_a": team_a_won - expected_probability,
                        "team_a_wins_before": team_a_wins,
                        "team_b_wins_before": team_b_wins,
                        "series_probability_before_a": series_before,
                        "series_probability_after_a": series_after,
                        "series_probability_swing_a": series_after - series_before,
                        "team_a_home": team_a_home,
                    }
                )
                team_a_wins = wins_after_a
                team_b_wins = wins_after_b
    return pd.DataFrame(rows)


def reverse_momentum_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reversed_frame = frame.copy()
    for column in MOMENTUM_FEATURE_COLUMNS:
        reversed_frame[column] = -pd.to_numeric(frame[column], errors="coerce")
    return reversed_frame


def _fit_momentum_model(frame: pd.DataFrame):
    direct = frame[MOMENTUM_FEATURE_COLUMNS].reset_index(drop=True)
    reverse = reverse_momentum_frame(frame)[MOMENTUM_FEATURE_COLUMNS].reset_index(
        drop=True
    )
    target = frame["TEAM_A_WON"].astype(int).reset_index(drop=True)
    augmented = pd.concat([direct, reverse], ignore_index=True)
    augmented_target = pd.concat([target, 1 - target], ignore_index=True)
    return _logistic_estimator().fit(augmented, augmented_target)


def _momentum_probability(model, frame: pd.DataFrame) -> np.ndarray:
    return symmetric_model_probability(
        model,
        frame[MOMENTUM_FEATURE_COLUMNS],
        reverse_momentum_frame(frame)[MOMENTUM_FEATURE_COLUMNS],
    )


def _merge_experiment_data(
    base: pd.DataFrame,
    momentum: pd.DataFrame,
    recent_path: Path,
) -> pd.DataFrame:
    momentum_columns = [
        "SEASON",
        "GAME_ID",
        *MOMENTUM_FEATURE_COLUMNS,
        "games_used_to_compute_momentum",
        "latest_game_used",
        "target_game_excluded",
        "postgame_series_score_excluded",
        "previous_feature_game_id",
        "pregame_series_score",
        "game_number_audit",
        "team_a_series_wins_audit",
        "team_b_series_wins_audit",
        "team_a_home_audit",
        "previous_game_winner_audit",
        "team_a_won_previous_two_audit",
        "team_b_won_previous_two_audit",
        "previous_margin_audit",
        "previous_expected_probability_a_audit",
        "previous_residual_margin_a_audit",
        "team_a_last_two_positive_residuals_audit",
        "team_b_last_two_positive_residuals_audit",
        "team_a_momentum_state_audit",
        "team_b_momentum_state_audit",
        "home_court_swing_audit",
    ]
    merged = base.merge(
        momentum[momentum_columns],
        on=["SEASON", "GAME_ID"],
        validate="one_to_one",
    )
    if recent_path.exists():
        recent = pd.read_csv(recent_path)
        recent["GAME_ID"] = _normalize_game_id(recent["GAME_ID"])
        recent_columns = [
            "SEASON",
            "GAME_ID",
            *TEAM_RECENT_FEATURE_COLUMNS,
            *EXPERIMENT_SERIES_FEATURE_COLUMNS,
        ]
        merged = merged.drop(
            columns=[
                column
                for column in [
                    *TEAM_RECENT_FEATURE_COLUMNS,
                    *EXPERIMENT_SERIES_FEATURE_COLUMNS,
                ]
                if column in merged.columns
            ],
            errors="ignore",
        ).merge(
            recent[recent_columns],
            on=["SEASON", "GAME_ID"],
            validate="one_to_one",
        )
    else:
        raise FileNotFoundError(
            "Run src.recent_form_experiment first so comparable recent and series "
            "features are available."
        )
    return merged


def run_momentum_state_experiment(
    *,
    cache_dir: str | Path = "data/raw",
    base_training_path: str | Path = DEFAULT_BASE_TRAINING_PATH,
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    recent_path: str | Path = "data/processed/recent_form_experiment_features.csv",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    train_seasons: list[str] | None = None,
    blend_seasons: list[str] | None = None,
    test_seasons: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    train_seasons = train_seasons or season_range("2015-16", "2022-23")
    blend_seasons = blend_seasons or season_range("2018-19", "2022-23")
    test_seasons = test_seasons or ["2023-24", "2024-25"]
    all_seasons = [*train_seasons, *test_seasons]

    base = pd.read_csv(base_training_path)
    base["GAME_ID"] = _normalize_game_id(base["GAME_ID"])
    margins = _playoff_margin_rows(all_seasons, Path(cache_dir))
    base = base.merge(
        margins[
            [
                "SEASON",
                "GAME_ID",
                "TEAM_A_MARGIN",
                "TEAM_A_HOME",
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
    base = base.merge(
        expectations,
        on=["SEASON", "GAME_ID"],
        validate="one_to_one",
    )
    momentum = build_momentum_dataset(base)
    merged = _merge_experiment_data(base, momentum, Path(recent_path))
    train = merged[merged["SEASON"].isin(train_seasons)].copy()
    test = merged[merged["SEASON"].isin(test_seasons)].copy()

    out_of_fold = []
    for season in blend_seasons:
        fold_train = train[train["SEASON"].ne(season)]
        validation = train[train["SEASON"].eq(season)]
        print(f"Fitting momentum blend fold: {season}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fold_base = clone(base_model).fit(
                fold_train[base_features], fold_train["TEAM_A_WON"].astype(int)
            )
            base_probability = symmetric_model_probability(
                fold_base,
                validation[base_features],
                _reverse_base_frame(validation, base_features),
            )
        context_model = _fit_symmetric_logistic(
            fold_train, EXPERIMENT_SERIES_FEATURE_COLUMNS
        )
        recent_model = _fit_symmetric_logistic(
            fold_train, TEAM_RECENT_FEATURE_COLUMNS
        )
        momentum_model = _fit_momentum_model(fold_train)
        out_of_fold.append(
            pd.DataFrame(
                {
                    "base": base_probability,
                    "series": _symmetric_probability(
                        context_model, validation, EXPERIMENT_SERIES_FEATURE_COLUMNS
                    ),
                    "recent": _symmetric_probability(
                        recent_model, validation, TEAM_RECENT_FEATURE_COLUMNS
                    ),
                    "momentum": _momentum_probability(momentum_model, validation),
                    "target": validation["TEAM_A_WON"].astype(int).to_numpy(),
                }
            )
        )
    blend_frame = pd.concat(out_of_fold, ignore_index=True)
    base_series_blend = _fit_blend(
        [blend_frame["base"], blend_frame["series"]], blend_frame["target"]
    )
    base_momentum_blend = _fit_blend(
        [blend_frame["base"], blend_frame["momentum"]], blend_frame["target"]
    )
    full_blend = _fit_blend(
        [blend_frame["base"], blend_frame["series"], blend_frame["momentum"]],
        blend_frame["target"],
    )

    context_model = _fit_symmetric_logistic(train, EXPERIMENT_SERIES_FEATURE_COLUMNS)
    recent_model = _fit_symmetric_logistic(train, TEAM_RECENT_FEATURE_COLUMNS)
    momentum_model = _fit_momentum_model(train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_probability = symmetric_model_probability(
            base_model,
            test[base_features],
            _reverse_base_frame(test, base_features),
        )
    series_probability = _symmetric_probability(
        context_model, test, EXPERIMENT_SERIES_FEATURE_COLUMNS
    )
    recent_probability = _symmetric_probability(
        recent_model, test, TEAM_RECENT_FEATURE_COLUMNS
    )
    momentum_probability = _momentum_probability(momentum_model, test)
    base_series_probability = _blend_probability(
        base_series_blend, [base_probability, series_probability]
    )
    base_momentum_probability = _blend_probability(
        base_momentum_blend, [base_probability, momentum_probability]
    )
    full_probability = _blend_probability(
        full_blend, [base_probability, series_probability, momentum_probability]
    )
    predictions = {
        "base_only": base_probability,
        "series_context_only": series_probability,
        "recent_form_only": recent_probability,
        "momentum_state_only": momentum_probability,
        "base_plus_series": base_series_probability,
        "base_plus_momentum_state": base_momentum_probability,
        "base_plus_series_plus_momentum_state": full_probability,
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
            for row in _calibration_rows(name, test["TEAM_A_WON"], probability)
        ]
    )
    classifier = momentum_model.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {
            "feature": MOMENTUM_FEATURE_COLUMNS,
            "standardized_coefficient": classifier.coef_[0].astype(float),
        }
    )
    coefficients["absolute_coefficient"] = coefficients["standardized_coefficient"].abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)

    examples = test[
        ["SEASON", "GAME_ID", "GAME_DATE", "TEAM_A", "TEAM_B", "TEAM_A_WON"]
    ].copy()
    examples["base_probability"] = base_probability
    examples["base_plus_series_probability"] = base_series_probability
    examples["with_momentum_probability"] = full_probability
    examples["momentum_probability_delta_vs_base"] = full_probability - base_probability
    examples["momentum_probability_delta_vs_base_series"] = (
        full_probability - base_series_probability
    )
    examples = examples.reindex(
        examples["momentum_probability_delta_vs_base"].abs().sort_values(
            ascending=False
        ).index
    ).head(20)

    prediction_rows = test[
        [
            "SEASON",
            "GAME_ID",
            "GAME_DATE",
            "TEAM_A",
            "TEAM_B",
            "TEAM_A_WON",
            "seed_A",
            "seed_B",
            "game_number_audit",
            "team_a_series_wins_audit",
            "team_b_series_wins_audit",
            "team_a_home_audit",
            "previous_game_winner_audit",
            "team_a_won_previous_two_audit",
            "team_b_won_previous_two_audit",
            "previous_margin_audit",
            "previous_expected_probability_a_audit",
            "previous_residual_margin_a_audit",
            "team_a_last_two_positive_residuals_audit",
            "team_b_last_two_positive_residuals_audit",
            "team_a_momentum_state_audit",
            "team_b_momentum_state_audit",
            "home_court_swing_audit",
        ]
    ].copy()
    prediction_rows["base_probability"] = base_probability
    prediction_rows["base_plus_series_probability"] = base_series_probability
    prediction_rows["base_plus_momentum_probability"] = base_momentum_probability
    prediction_rows["base_plus_series_plus_momentum_probability"] = full_probability

    audit_columns = [
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_A",
        "TEAM_B",
        "TEAM_A_WON",
        "games_used_to_compute_momentum",
        "latest_game_used",
        "target_game_excluded",
        "postgame_series_score_excluded",
        "previous_feature_game_id",
        "pregame_series_score",
    ]
    audit = merged[merged["games_used_to_compute_momentum"].ne("")][
        audit_columns
    ].head(5)
    print("\nLeakage audit samples:")
    print(audit.to_string(index=False))

    metric_index = metrics.set_index("model")
    base_metrics = metric_index.loc["base_only"]
    full_metrics = metric_index.loc["base_plus_series_plus_momentum_state"]
    recommendation = (
        "DEPLOY CANDIDATE: momentum improved held-out Brier score or log loss over "
        "base only. Review stability before any production change."
        if (
            full_metrics["brier_score"] < base_metrics["brier_score"]
            or full_metrics["log_loss"] < base_metrics["log_loss"]
        )
        else (
            "DO NOT DEPLOY: surprise-adjusted momentum did not improve held-out "
            "Brier score or log loss over base only."
        )
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "momentum_state_experiment_metrics.csv", index=False)
    coefficients.to_csv(
        output_dir / "momentum_state_experiment_coefficients.csv", index=False
    )
    calibration.to_csv(
        output_dir / "momentum_state_experiment_calibration.csv", index=False
    )
    examples.to_csv(
        output_dir / "momentum_state_experiment_examples.csv", index=False
    )
    prediction_rows.to_csv(
        output_dir / "momentum_state_experiment_predictions.csv", index=False
    )
    print(recommendation)
    return metrics, recommendation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline surprise-adjusted NBA playoff momentum experiment"
    )
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--base-training-path", default=DEFAULT_BASE_TRAINING_PATH)
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument(
        "--recent-path",
        default="data/processed/recent_form_experiment_features.csv",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics, recommendation = run_momentum_state_experiment(
        cache_dir=args.cache_dir,
        base_training_path=args.base_training_path,
        base_model_path=args.base_model_path,
        recent_path=args.recent_path,
        output_dir=args.output_dir,
    )
    print(metrics.to_string(index=False))
    print(recommendation)


if __name__ == "__main__":
    main()
