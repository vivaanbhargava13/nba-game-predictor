from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from .model import (
    PREDICTION_MODE_CURRENT,
    get_model_entry_for_mode,
    load_model,
)
from .recent_form_experiment import (
    _blend_probability,
    _calibration_rows,
    _fit_blend,
    _logistic_estimator,
    _reverse_base_frame,
)
from .series_context import (
    SERIES_CONTEXT_FEATURE_COLUMNS,
    symmetric_model_probability,
)
from .team_advanced_broader_validation import (
    ALL_SEASONS,
    expected_calibration_error,
    playoff_round_from_game_id,
    validation_folds,
)
from .team_advanced_recent_deltas_experiment import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_BASE_TRAINING_PATH,
    DEFAULT_SERIES_CONTEXT_PATH,
    _canonical_game_id,
    _fit_series_model,
    _metrics,
    _series_probability,
)

DEFAULT_PLAYER_ROWS_PATH = Path(
    "data/processed/advanced_boxscore_v3_player_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_DOC_PATH = Path("docs/player_rotation_strength_experiment.md")
EXPERIMENT_SCHEMA_VERSION = "2026-06-13-player-rotation-v1"
FINAL_TRAIN_SEASONS = ALL_SEASONS[:8]
FINAL_BLEND_SEASONS = ALL_SEASONS[3:8]
FINAL_TEST_SEASONS = ALL_SEASONS[8:]

ROTATION_FEATURE_COLUMNS = [
    "last_1_top_5_minutes_share_diff",
    "last_1_top_7_minutes_share_diff",
    "last_1_top_8_minutes_share_diff",
    "last_3_top_5_minutes_share_diff",
    "last_3_top_7_minutes_share_diff",
    "last_3_top_8_minutes_share_diff",
    "series_to_date_top_7_minutes_share_diff",
    "last_3_top_3_usage_share_diff",
    "last_3_top_5_usage_share_diff",
    "series_to_date_top_3_usage_share_diff",
    "series_to_date_top_5_usage_share_diff",
    "minutes_weighted_net_rating_diff",
    "minutes_weighted_off_rating_diff",
    "minutes_weighted_def_rating_diff",
    "minutes_weighted_pie_diff",
    "usage_weighted_pie_diff",
    "usage_weighted_true_shooting_diff",
    "repeated_top_7_players_share_diff",
    "top_7_continuity_from_previous_game_diff",
    "minutes_distribution_entropy_diff",
    "bench_minutes_share_diff",
    "top_3_minutes_load_last_3_diff",
    "top_5_minutes_load_last_3_diff",
    "players_over_38_minutes_last_game_diff",
    "players_over_40_minutes_last_game_diff",
]


def parse_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" not in text:
        return float(pd.to_numeric(text, errors="coerce") or 0.0)
    minutes, seconds = text.split(":", 1)
    return float(minutes) + float(seconds) / 60.0


def _weighted_average(
    frame: pd.DataFrame,
    value_column: str,
    weight: pd.Series,
) -> float:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    valid = values.notna() & weight.notna() & weight.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weight[valid]))


def build_team_game_snapshot(group: pd.DataFrame) -> dict:
    players = group.copy()
    players["minutes_value"] = players["minutes"].map(parse_minutes)
    players = players[players["minutes_value"].gt(0)].copy()
    if players.empty:
        raise ValueError("Player game snapshot has no positive minutes.")
    players["personId"] = pd.to_numeric(
        players["personId"], errors="raise"
    ).astype(int)
    players = players.sort_values(
        ["minutes_value", "personId"], ascending=[False, True]
    )
    total_minutes = float(players["minutes_value"].sum())
    minute_share = players["minutes_value"] / total_minutes
    usage = pd.to_numeric(
        players["usagePercentage"], errors="coerce"
    ).fillna(0.0)
    usage_load = players["minutes_value"] * usage.clip(lower=0)
    usage_total = float(usage_load.sum())

    snapshot = {
        "season": str(players.iloc[0]["season"]),
        "gameId": str(players.iloc[0]["gameId"]),
        "gameDate": pd.Timestamp(players.iloc[0]["gameDate"]),
        "teamId": int(players.iloc[0]["teamId"]),
        "top_5_minutes_share": float(minute_share.head(5).sum()),
        "top_7_minutes_share": float(minute_share.head(7).sum()),
        "top_8_minutes_share": float(minute_share.head(8).sum()),
        "top_3_usage_share": (
            float(usage_load.nlargest(3).sum() / usage_total)
            if usage_total > 0
            else float("nan")
        ),
        "top_5_usage_share": (
            float(usage_load.nlargest(5).sum() / usage_total)
            if usage_total > 0
            else float("nan")
        ),
        "minutes_weighted_net_rating": _weighted_average(
            players, "netRating", players["minutes_value"]
        ),
        "minutes_weighted_off_rating": _weighted_average(
            players, "offensiveRating", players["minutes_value"]
        ),
        "minutes_weighted_def_rating": _weighted_average(
            players, "defensiveRating", players["minutes_value"]
        ),
        "minutes_weighted_pie": _weighted_average(
            players, "PIE", players["minutes_value"]
        ),
        "usage_weighted_pie": _weighted_average(
            players, "PIE", usage_load
        ),
        "usage_weighted_true_shooting": _weighted_average(
            players, "trueShootingPercentage", usage_load
        ),
        "minutes_distribution_entropy": (
            float(
                -np.sum(minute_share * np.log(minute_share))
                / math.log(len(minute_share))
            )
            if len(minute_share) > 1
            else 0.0
        ),
        "bench_minutes_share": float(1.0 - minute_share.head(5).sum()),
        "top_3_minutes_load": float(
            players["minutes_value"].head(3).sum()
        ),
        "top_5_minutes_load": float(
            players["minutes_value"].head(5).sum()
        ),
        "players_over_38_minutes": int(
            players["minutes_value"].gt(38).sum()
        ),
        "players_over_40_minutes": int(
            players["minutes_value"].gt(40).sum()
        ),
        "top_7_ids": tuple(players["personId"].head(7).tolist()),
    }
    return snapshot


def build_snapshot_lookup(player_rows: pd.DataFrame) -> dict:
    frame = player_rows.copy()
    frame["gameId"] = _canonical_game_id(frame["gameId"])
    frame["gameDate"] = pd.to_datetime(frame["gameDate"])
    frame["teamId"] = pd.to_numeric(
        frame["teamId"], errors="raise"
    ).astype(int)
    return {
        (season, game_id, team_id): build_team_game_snapshot(group)
        for (season, game_id, team_id), group in frame.groupby(
            ["season", "gameId", "teamId"], sort=False
        )
    }


def _history_mean(
    history: list[dict], key: str, window: int | None
) -> float:
    selected = history[-window:] if window else history
    if not selected:
        return float("nan")
    return float(
        pd.to_numeric(
            pd.Series([row[key] for row in selected]), errors="coerce"
        ).mean()
    )


def _repeated_top_seven_share(history: list[dict]) -> float:
    selected = history[-3:]
    if len(selected) < 2:
        return float("nan")
    latest = set(selected[-1]["top_7_ids"])
    repeated = latest.intersection(
        *(set(row["top_7_ids"]) for row in selected[:-1])
    )
    return float(len(repeated) / max(1, len(latest)))


def _top_seven_continuity(history: list[dict]) -> float:
    if len(history) < 2:
        return float("nan")
    latest = set(history[-1]["top_7_ids"])
    previous = set(history[-2]["top_7_ids"])
    return float(len(latest & previous) / max(1, len(latest)))


def _team_rotation_values(
    history: list[dict],
    series_history: list[dict],
) -> dict:
    return {
        "last_1_top_5_minutes_share": _history_mean(
            history, "top_5_minutes_share", 1
        ),
        "last_1_top_7_minutes_share": _history_mean(
            history, "top_7_minutes_share", 1
        ),
        "last_1_top_8_minutes_share": _history_mean(
            history, "top_8_minutes_share", 1
        ),
        "last_3_top_5_minutes_share": _history_mean(
            history, "top_5_minutes_share", 3
        ),
        "last_3_top_7_minutes_share": _history_mean(
            history, "top_7_minutes_share", 3
        ),
        "last_3_top_8_minutes_share": _history_mean(
            history, "top_8_minutes_share", 3
        ),
        "series_to_date_top_7_minutes_share": _history_mean(
            series_history, "top_7_minutes_share", None
        ),
        "last_3_top_3_usage_share": _history_mean(
            history, "top_3_usage_share", 3
        ),
        "last_3_top_5_usage_share": _history_mean(
            history, "top_5_usage_share", 3
        ),
        "series_to_date_top_3_usage_share": _history_mean(
            series_history, "top_3_usage_share", None
        ),
        "series_to_date_top_5_usage_share": _history_mean(
            series_history, "top_5_usage_share", None
        ),
        "minutes_weighted_net_rating": _history_mean(
            history, "minutes_weighted_net_rating", 3
        ),
        "minutes_weighted_off_rating": _history_mean(
            history, "minutes_weighted_off_rating", 3
        ),
        "minutes_weighted_def_rating": _history_mean(
            history, "minutes_weighted_def_rating", 3
        ),
        "minutes_weighted_pie": _history_mean(
            history, "minutes_weighted_pie", 3
        ),
        "usage_weighted_pie": _history_mean(
            history, "usage_weighted_pie", 3
        ),
        "usage_weighted_true_shooting": _history_mean(
            history, "usage_weighted_true_shooting", 3
        ),
        "repeated_top_7_players_share": _repeated_top_seven_share(
            history
        ),
        "top_7_continuity_from_previous_game": _top_seven_continuity(
            history
        ),
        "minutes_distribution_entropy": _history_mean(
            history, "minutes_distribution_entropy", 3
        ),
        "bench_minutes_share": _history_mean(
            history, "bench_minutes_share", 3
        ),
        "top_3_minutes_load_last_3": float(
            sum(row["top_3_minutes_load"] for row in history[-3:])
        )
        if history
        else float("nan"),
        "top_5_minutes_load_last_3": float(
            sum(row["top_5_minutes_load"] for row in history[-3:])
        )
        if history
        else float("nan"),
        "players_over_38_minutes_last_game": _history_mean(
            history, "players_over_38_minutes", 1
        ),
        "players_over_40_minutes_last_game": _history_mean(
            history, "players_over_40_minutes", 1
        ),
    }


def build_rotation_feature_row(
    *,
    target: pd.Series,
    team_history: dict[int, list[dict]],
    series_history: dict[tuple[int, int], dict[int, list[dict]]],
) -> dict:
    team_a_id = int(target["TEAM_A_ID"])
    team_b_id = int(target["TEAM_B_ID"])
    history_a = team_history.get(team_a_id, [])
    history_b = team_history.get(team_b_id, [])
    pair = tuple(sorted((team_a_id, team_b_id)))
    pair_history = series_history.get(pair, {})
    series_a = pair_history.get(team_a_id, [])
    series_b = pair_history.get(team_b_id, [])
    values_a = _team_rotation_values(history_a, series_a)
    values_b = _team_rotation_values(history_b, series_b)

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
    for feature in ROTATION_FEATURE_COLUMNS:
        key = feature.removesuffix("_diff")
        row[feature] = values_a[key] - values_b[key]

    used = [*history_a[-3:], *history_b[-3:]]
    used_ids = sorted({row["gameId"] for row in used})
    used_dates = [pd.Timestamp(row["gameDate"]) for row in used]
    row.update(
        {
            "game_number_audit": len(series_a) + 1,
            "player_games_used_in_rolling_window": ",".join(used_ids),
            "latest_player_game_date_used": (
                max(used_dates) if used_dates else pd.NaT
            ),
            "target_game_excluded": str(target["GAME_ID"])
            not in used_ids,
            "postgame_state_excluded": True,
        }
    )
    return row


def build_rotation_dataset(
    base_frame: pd.DataFrame,
    player_rows: pd.DataFrame,
) -> pd.DataFrame:
    base = base_frame.copy()
    base["GAME_ID"] = _canonical_game_id(base["GAME_ID"])
    base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
    snapshots = build_snapshot_lookup(player_rows)
    rows = []
    for season, targets in base.groupby("SEASON", sort=True):
        print(f"Building leakage-safe player rotation rows: {season}")
        team_history: dict[int, list[dict]] = {}
        series_history: dict[
            tuple[int, int], dict[int, list[dict]]
        ] = {}
        targets = targets.sort_values(
            ["GAME_DATE", "GAME_ID"]
        ).reset_index(drop=True)
        for _, target in targets.iterrows():
            rows.append(
                build_rotation_feature_row(
                    target=target,
                    team_history=team_history,
                    series_history=series_history,
                )
            )
            team_ids = (
                int(target["TEAM_A_ID"]),
                int(target["TEAM_B_ID"]),
            )
            pair = tuple(sorted(team_ids))
            pair_history = series_history.setdefault(pair, {})
            for team_id in team_ids:
                key = (str(season), str(target["GAME_ID"]), team_id)
                snapshot = snapshots.get(key)
                if snapshot is None:
                    raise ValueError(
                        f"Missing V3 player rows for {key}"
                    )
                team_history.setdefault(team_id, []).append(snapshot)
                pair_history.setdefault(team_id, []).append(snapshot)
    return pd.DataFrame(rows)


def reverse_rotation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reversed_frame = frame.copy()
    for column in ROTATION_FEATURE_COLUMNS:
        reversed_frame[column] = -pd.to_numeric(
            frame[column], errors="coerce"
        )
    return reversed_frame


def _fit_rotation_model(frame: pd.DataFrame):
    direct = frame[ROTATION_FEATURE_COLUMNS].reset_index(drop=True)
    reverse = reverse_rotation_frame(frame)[
        ROTATION_FEATURE_COLUMNS
    ].reset_index(drop=True)
    target = frame["TEAM_A_WON"].astype(int).reset_index(drop=True)
    return _logistic_estimator().fit(
        pd.concat([direct, reverse], ignore_index=True),
        pd.concat([target, 1 - target], ignore_index=True),
    )


def _rotation_probability(model, frame: pd.DataFrame) -> np.ndarray:
    return symmetric_model_probability(
        model,
        frame[ROTATION_FEATURE_COLUMNS],
        reverse_rotation_frame(frame)[ROTATION_FEATURE_COLUMNS],
    )


def _merge_inputs(
    base: pd.DataFrame,
    rotation: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    frames = [base.copy(), rotation.copy(), context.copy()]
    for frame in frames:
        frame["GAME_ID"] = _canonical_game_id(frame["GAME_ID"])
    base, rotation, context = frames
    base = base.drop(
        columns=[
            column
            for column in SERIES_CONTEXT_FEATURE_COLUMNS
            if column in base
        ],
        errors="ignore",
    )
    rotation_columns = [
        "SEASON",
        "GAME_ID",
        *ROTATION_FEATURE_COLUMNS,
        "game_number_audit",
        "player_games_used_in_rolling_window",
        "latest_player_game_date_used",
        "target_game_excluded",
        "postgame_state_excluded",
    ]
    return (
        base.merge(
            rotation[rotation_columns],
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


def _base_probability(
    model,
    frame: pd.DataFrame,
    base_features: list[str],
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return symmetric_model_probability(
            model,
            frame[base_features],
            _reverse_base_frame(frame, base_features),
        )


def _fit_components(
    train: pd.DataFrame,
    base_template,
    base_features: list[str],
    cache: dict | None = None,
):
    key = tuple(sorted(train["SEASON"].astype(str).unique()))
    if cache is not None and key in cache:
        return cache[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_model = clone(base_template).fit(
            train[base_features],
            train["TEAM_A_WON"].astype(int),
        )
        rotation_model = _fit_rotation_model(train)
    result = base_model, rotation_model
    if cache is not None:
        cache[key] = result
    return result


def _fit_rotation_blend(
    train: pd.DataFrame,
    base_template,
    base_features: list[str],
    cache: dict | None = None,
):
    rows = []
    for season in sorted(train["SEASON"].unique()):
        inner_train = train[train["SEASON"].ne(season)]
        validation = train[train["SEASON"].eq(season)]
        if inner_train["SEASON"].nunique() < 2 or validation.empty:
            continue
        base_model, rotation_model = _fit_components(
            inner_train, base_template, base_features, cache
        )
        rows.append(
            pd.DataFrame(
                {
                    "base": _base_probability(
                        base_model, validation, base_features
                    ),
                    "rotation": _rotation_probability(
                        rotation_model, validation
                    ),
                    "target": validation["TEAM_A_WON"]
                    .astype(int)
                    .to_numpy(),
                }
            )
        )
    if not rows:
        raise ValueError("Could not create rotation blend folds.")
    oof = pd.concat(rows, ignore_index=True)
    return _fit_blend(
        [oof["base"], oof["rotation"]], oof["target"]
    )


def _evaluation_row(
    name: str,
    target: pd.Series,
    probability: np.ndarray,
) -> dict:
    row = _metrics(name, target, probability)
    row["expected_calibration_error"] = expected_calibration_error(
        target, probability
    )
    return row


def _held_out_evaluation(
    merged: pd.DataFrame,
    base_template,
    base_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = merged[merged["SEASON"].isin(FINAL_TRAIN_SEASONS)].copy()
    test = merged[merged["SEASON"].isin(FINAL_TEST_SEASONS)].copy()
    cache = {}
    oof = []
    for season in FINAL_BLEND_SEASONS:
        fold_train = train[train["SEASON"].ne(season)]
        validation = train[train["SEASON"].eq(season)]
        base_model, rotation_model = _fit_components(
            fold_train, base_template, base_features, cache
        )
        series_model = _fit_series_model(fold_train)
        oof.append(
            pd.DataFrame(
                {
                    "base": _base_probability(
                        base_model, validation, base_features
                    ),
                    "series": _series_probability(
                        series_model, validation
                    ),
                    "rotation": _rotation_probability(
                        rotation_model, validation
                    ),
                    "target": validation["TEAM_A_WON"].to_numpy(),
                }
            )
        )
    blend_rows = pd.concat(oof, ignore_index=True)
    base_rotation_blend = _fit_blend(
        [blend_rows["base"], blend_rows["rotation"]],
        blend_rows["target"],
    )
    base_series_blend = _fit_blend(
        [blend_rows["base"], blend_rows["series"]],
        blend_rows["target"],
    )
    full_blend = _fit_blend(
        [
            blend_rows["base"],
            blend_rows["series"],
            blend_rows["rotation"],
        ],
        blend_rows["target"],
    )

    rotation_model = _fit_rotation_model(train)
    series_model = _fit_series_model(train)
    base_probability = _base_probability(
        base_template, test, base_features
    )
    rotation_probability = _rotation_probability(rotation_model, test)
    series_probability = _series_probability(series_model, test)
    predictions = {
        "base_only": base_probability,
        "rotation_only": rotation_probability,
        "base_plus_rotation": _blend_probability(
            base_rotation_blend,
            [base_probability, rotation_probability],
        ),
        "base_plus_series": _blend_probability(
            base_series_blend, [base_probability, series_probability]
        ),
        "base_plus_series_plus_rotation": _blend_probability(
            full_blend,
            [base_probability, series_probability, rotation_probability],
        ),
    }
    metrics = pd.DataFrame(
        [
            {
                "evaluation_scope": "final_held_out",
                "validation_design": "held_out_2023_24_2024_25",
                "fold": "final_test",
                "test_seasons": ",".join(FINAL_TEST_SEASONS),
                **_evaluation_row(
                    name, test["TEAM_A_WON"], probability
                ),
            }
            for name, probability in predictions.items()
        ]
    )
    calibration = pd.DataFrame(
        [
            row
            for name, probability in predictions.items()
            for row in _calibration_rows(
                name, test["TEAM_A_WON"], probability
            )
        ]
    )
    calibration["evaluation_scope"] = "final_held_out"
    calibration["test_seasons"] = ",".join(FINAL_TEST_SEASONS)

    classifier = rotation_model.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {
            "feature": ROTATION_FEATURE_COLUMNS,
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
    examples["rotation_delta_vs_base"] = (
        predictions["base_plus_rotation"] - base_probability
    )
    examples["base_brier_loss"] = (
        base_probability - examples["TEAM_A_WON"]
    ) ** 2
    examples["rotation_brier_loss"] = (
        predictions["base_plus_rotation"]
        - examples["TEAM_A_WON"]
    ) ** 2
    examples["brier_improvement"] = (
        examples["base_brier_loss"]
        - examples["rotation_brier_loss"]
    )
    examples = pd.concat(
        [
            examples.nlargest(15, "brier_improvement").assign(
                example_type="rotation_helped"
            ),
            examples.nsmallest(15, "brier_improvement").assign(
                example_type="rotation_hurt"
            ),
        ]
    )
    return metrics, calibration, coefficients, examples


def _broader_validation(
    merged: pd.DataFrame,
    base_template,
    base_features: list[str],
) -> pd.DataFrame:
    rows = []
    cache = {}
    for config in validation_folds(ALL_SEASONS):
        print(
            "Evaluating rotation "
            f"{config['validation_design']}: {config['test_season']}"
        )
        train = merged[
            merged["SEASON"].isin(config["train_seasons"])
        ].copy()
        test = merged[
            merged["SEASON"].eq(config["test_season"])
        ].copy()
        blend = _fit_rotation_blend(
            train, base_template, base_features, cache
        )
        base_model, rotation_model = _fit_components(
            train, base_template, base_features, cache
        )
        base_probability = _base_probability(
            base_model, test, base_features
        )
        combined_probability = _blend_probability(
            blend,
            [
                base_probability,
                _rotation_probability(rotation_model, test),
            ],
        )
        movement = float(
            np.mean(np.abs(combined_probability - base_probability))
        )
        metadata = {
            "evaluation_scope": "broader_validation",
            "validation_design": config["validation_design"],
            "fold": config["fold"],
            "train_seasons": ",".join(config["train_seasons"]),
            "test_seasons": config["test_season"],
            "train_games": len(train),
            "test_games": len(test),
            "average_probability_movement": movement,
        }
        for name, probability in (
            ("base_only", base_probability),
            ("base_plus_rotation", combined_probability),
        ):
            rows.append(
                {
                    **metadata,
                    **_evaluation_row(
                        name, test["TEAM_A_WON"], probability
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_broader_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    broader = metrics[
        metrics["evaluation_scope"].eq("broader_validation")
    ]
    pivot = broader.pivot_table(
        index=["validation_design", "fold", "test_seasons"],
        columns="model",
        values=[
            "roc_auc",
            "brier_score",
            "log_loss",
            "accuracy",
            "f1",
            "expected_calibration_error",
            "average_probability_movement",
        ],
        aggfunc="first",
    )
    rows = []
    for design, group in pivot.groupby(level="validation_design"):
        flat = group.droplevel("validation_design")
        deltas = {
            metric: (
                flat[(metric, "base_plus_rotation")]
                - flat[(metric, "base_only")]
            )
            for metric in (
                "roc_auc",
                "brier_score",
                "log_loss",
                "accuracy",
                "f1",
                "expected_calibration_error",
            )
        }
        rows.append(
            {
                "validation_design": design,
                "folds": len(flat),
                "brier_improved_folds": int(
                    deltas["brier_score"].lt(0).sum()
                ),
                "log_loss_improved_folds": int(
                    deltas["log_loss"].lt(0).sum()
                ),
                "roc_auc_improved_folds": int(
                    deltas["roc_auc"].gt(0).sum()
                ),
                "average_roc_auc_delta": float(
                    deltas["roc_auc"].mean()
                ),
                "average_brier_delta": float(
                    deltas["brier_score"].mean()
                ),
                "average_log_loss_delta": float(
                    deltas["log_loss"].mean()
                ),
                "average_accuracy_delta": float(
                    deltas["accuracy"].mean()
                ),
                "average_f1_delta": float(deltas["f1"].mean()),
                "average_ece_delta": float(
                    deltas["expected_calibration_error"].mean()
                ),
                "average_probability_movement": float(
                    flat[
                        (
                            "average_probability_movement",
                            "base_plus_rotation",
                        )
                    ].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def deployment_recommendation(summary: pd.DataFrame) -> str:
    total_folds = int(summary["folds"].sum())
    brier_wins = int(summary["brier_improved_folds"].sum())
    log_wins = int(summary["log_loss_improved_folds"].sum())
    roc_wins = int(summary["roc_auc_improved_folds"].sum())
    probability_improves_most = (
        brier_wins > total_folds / 2
        or log_wins > total_folds / 2
    )
    roc_not_consistently_damaged = (
        roc_wins >= total_folds / 2
        or summary["average_roc_auc_delta"].mean() >= -0.002
    )
    if probability_improves_most and roc_not_consistently_damaged:
        return (
            "DEPLOY CANDIDATE: Base + Rotation improves Brier or log loss "
            "across most broader-validation folds without consistent ROC-AUC "
            "damage. Require a locked production review before deployment."
        )
    return (
        "RESEARCH-ONLY: rotation strength does not improve probability "
        "quality across most broader-validation folds without recurring "
        "ROC-AUC damage."
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else ""
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for values in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_doc(
    held_out: pd.DataFrame,
    broader: pd.DataFrame,
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    audit: pd.DataFrame,
    recommendation: str,
    path: Path,
) -> None:
    lines = [
        "# Player Rotation Strength Experiment",
        "",
        "Offline research experiment only. No deployed models or application behavior changed.",
        "",
        "Usage concentration is based on minutes multiplied by V3 usage percentage. Player-quality features use minutes or usage-load weighting. Every target row is built before its player rows enter history.",
        "",
        "## Final Held-Out Metrics",
        "",
        _markdown_table(held_out),
        "",
        "## Broader Fold Metrics",
        "",
        _markdown_table(broader),
        "",
        "## Broader Summary",
        "",
        _markdown_table(summary),
        "",
        "## Strongest Rotation Coefficients",
        "",
        _markdown_table(coefficients.head(12)),
        "",
        "## Leakage Audit",
        "",
        _markdown_table(audit),
        "",
        "## Decision",
        "",
        recommendation,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    player_rows_path: str | Path = DEFAULT_PLAYER_ROWS_PATH,
    base_training_path: str | Path = DEFAULT_BASE_TRAINING_PATH,
    series_context_path: str | Path = DEFAULT_SERIES_CONTEXT_PATH,
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> tuple[pd.DataFrame, str]:
    base = pd.read_csv(base_training_path)
    player_rows = pd.read_csv(player_rows_path)
    context = pd.read_csv(series_context_path)
    rotation = build_rotation_dataset(base, player_rows)
    merged = _merge_inputs(base, rotation, context)

    artifact = load_model(base_model_path)
    entry = get_model_entry_for_mode(
        artifact, PREDICTION_MODE_CURRENT
    )
    base_template = entry["pipeline"]
    base_features = list(entry["feature_columns"])

    held_out, calibration, coefficients, examples = (
        _held_out_evaluation(merged, base_template, base_features)
    )
    broader = _broader_validation(
        merged, base_template, base_features
    )
    metrics = pd.concat([held_out, broader], ignore_index=True)
    summary = summarize_broader_validation(metrics)
    recommendation = deployment_recommendation(summary)

    audit = merged[merged["game_number_audit"].gt(1)][
        [
            "SEASON",
            "GAME_ID",
            "GAME_DATE",
            "game_number_audit",
            "TEAM_A",
            "TEAM_B",
            "player_games_used_in_rolling_window",
            "latest_player_game_date_used",
            "target_game_excluded",
            "postgame_state_excluded",
        ]
    ].head(5)
    print("\nPlayer rotation leakage audit:")
    print(audit.to_string(index=False))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        output_dir / "player_rotation_strength_metrics.csv",
        index=False,
    )
    coefficients.to_csv(
        output_dir / "player_rotation_strength_coefficients.csv",
        index=False,
    )
    calibration.to_csv(
        output_dir / "player_rotation_strength_calibration.csv",
        index=False,
    )
    examples.to_csv(
        output_dir / "player_rotation_strength_examples.csv",
        index=False,
    )
    _write_doc(
        held_out,
        broader,
        summary,
        coefficients,
        audit,
        recommendation,
        Path(doc_path),
    )
    return metrics, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline V3 player rotation strength experiment"
    )
    parser.add_argument(
        "--player-rows-path", default=DEFAULT_PLAYER_ROWS_PATH
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
        player_rows_path=args.player_rows_path,
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
