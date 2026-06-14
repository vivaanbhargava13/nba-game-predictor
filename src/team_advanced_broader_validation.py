from __future__ import annotations

import argparse
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
from .recent_form_experiment import (
    _blend_probability,
    _fit_blend,
    _reverse_base_frame,
)
from .series_context import symmetric_model_probability
from .team_advanced_recent_deltas_experiment import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_BASE_TRAINING_PATH,
    DEFAULT_SERIES_CONTEXT_PATH,
    DEFAULT_TEAM_ROWS_PATH,
    _advanced_probability,
    _fit_advanced_model,
    _merge_inputs,
    build_advanced_dataset,
)

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_DOC_PATH = Path("docs/team_advanced_broader_validation.md")
ALL_SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]
EXPANDING_TEST_SEASONS = ALL_SEASONS[3:]
MODEL_NAMES = ("base_only", "base_plus_team_advanced")


def expected_calibration_error(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    target_values = np.asarray(target, dtype=float)
    probability_values = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(target_values)
    if total == 0:
        return float("nan")
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (probability_values >= lower)
            & (
                probability_values <= upper
                if index == bins - 1
                else probability_values < upper
            )
        )
        if not mask.any():
            continue
        error += (
            float(mask.mean())
            * abs(
                float(target_values[mask].mean())
                - float(probability_values[mask].mean())
            )
        )
    return float(error)


def playoff_round_from_game_id(game_id: str) -> str:
    normalized = str(game_id).replace(".0", "").zfill(10)
    round_code = normalized[-4:-2]
    return {
        "01": "first_round",
        "02": "conference_semifinals",
        "03": "conference_finals",
        "04": "nba_finals",
    }.get(round_code, "unknown_round")


def validation_folds(
    seasons: list[str],
) -> list[dict[str, object]]:
    expanding = [
        {
            "validation_design": "expanding_window",
            "fold": f"through_{seasons[index - 1]}_test_{season}",
            "train_seasons": seasons[:index],
            "test_season": season,
        }
        for index, season in enumerate(seasons)
        if season in EXPANDING_TEST_SEASONS
    ]
    leave_one_out = [
        {
            "validation_design": "leave_one_season_out",
            "fold": f"leave_out_{season}",
            "train_seasons": [
                candidate for candidate in seasons if candidate != season
            ],
            "test_season": season,
        }
        for season in seasons
    ]
    return [*expanding, *leave_one_out]


def _safe_roc_auc(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> float:
    values = pd.Series(target)
    if values.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(values, probability))


def metric_row(
    *,
    model: str,
    target: pd.Series,
    probability: np.ndarray,
    average_probability_movement: float,
    record_type: str = "fold",
    subgroup: str = "all_games",
) -> dict:
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    return {
        "record_type": record_type,
        "subgroup": subgroup,
        "model": model,
        "games": int(len(target)),
        "small_sample": bool(len(target) < 20),
        "roc_auc": _safe_roc_auc(target, probability),
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(
            log_loss(target, probability, labels=[0, 1])
        ),
        "accuracy": float(accuracy_score(target, predicted)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "expected_calibration_error": expected_calibration_error(
            target, probability
        ),
        "average_probability_movement": float(
            average_probability_movement
        ),
    }


def _base_probability(
    model,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return symmetric_model_probability(
            model,
            frame[feature_columns],
            _reverse_base_frame(frame, feature_columns),
        )


def _fit_fold_components(
    train: pd.DataFrame,
    base_template,
    base_features: list[str],
    model_cache: dict | None = None,
):
    cache_key = tuple(sorted(train["SEASON"].astype(str).unique()))
    if model_cache is not None and cache_key in model_cache:
        return model_cache[cache_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_model = clone(base_template).fit(
            train[base_features],
            train["TEAM_A_WON"].astype(int),
        )
        advanced_model = _fit_advanced_model(train)
    result = (base_model, advanced_model)
    if model_cache is not None:
        model_cache[cache_key] = result
    return result


def _fit_fold_blend(
    train: pd.DataFrame,
    base_template,
    base_features: list[str],
    model_cache: dict | None = None,
):
    predictions = []
    seasons = sorted(train["SEASON"].astype(str).unique())
    for validation_season in seasons:
        inner_train = train[
            train["SEASON"].astype(str).ne(validation_season)
        ]
        validation = train[
            train["SEASON"].astype(str).eq(validation_season)
        ]
        if (
            inner_train["SEASON"].nunique() < 2
            or validation.empty
            or inner_train["TEAM_A_WON"].nunique() < 2
        ):
            continue
        base_model, advanced_model = _fit_fold_components(
            inner_train,
            base_template,
            base_features,
            model_cache=model_cache,
        )
        predictions.append(
            pd.DataFrame(
                {
                    "base": _base_probability(
                        base_model, validation, base_features
                    ),
                    "advanced": _advanced_probability(
                        advanced_model, validation
                    ),
                    "target": validation["TEAM_A_WON"]
                    .astype(int)
                    .to_numpy(),
                }
            )
        )
    if not predictions:
        raise ValueError(
            "Could not create season-held-out blend predictions."
        )
    out_of_fold = pd.concat(predictions, ignore_index=True)
    return _fit_blend(
        [out_of_fold["base"], out_of_fold["advanced"]],
        out_of_fold["target"],
    )


def _subgroup_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    base_probability = frame["base_probability"]
    movement = frame["advanced_probability_delta"].abs()
    masks: dict[str, pd.Series] = {}
    for round_name in sorted(frame["playoff_round"].dropna().unique()):
        masks[f"playoff_round={round_name}"] = frame[
            "playoff_round"
        ].eq(round_name)
    for game_number in sorted(
        pd.to_numeric(
            frame["game_number_audit"], errors="coerce"
        ).dropna().unique()
    ):
        masks[f"game_number={int(game_number)}"] = pd.to_numeric(
            frame["game_number_audit"], errors="coerce"
        ).eq(game_number)
    masks.update(
        {
            "team_a_favorite": base_probability.ge(0.5),
            "team_a_underdog": base_probability.lt(0.5),
            "close_prediction_45_55": base_probability.between(
                0.45, 0.55, inclusive="both"
            ),
            "high_confidence_over_65": np.maximum(
                base_probability, 1 - base_probability
            ).gt(0.65),
            "advanced_movement_over_2_pct": movement.gt(0.02),
        }
    )
    return masks


def _evaluate_subgroups(
    prediction_frame: pd.DataFrame,
) -> list[dict]:
    rows = []
    for subgroup, mask in _subgroup_masks(prediction_frame).items():
        group = prediction_frame[mask]
        if group.empty:
            continue
        movement = float(
            group["advanced_probability_delta"].abs().mean()
        )
        for model, column in (
            ("base_only", "base_probability"),
            (
                "base_plus_team_advanced",
                "base_plus_team_advanced_probability",
            ),
        ):
            rows.append(
                metric_row(
                    model=model,
                    target=group["TEAM_A_WON"],
                    probability=group[column].to_numpy(),
                    average_probability_movement=movement,
                    record_type="subgroup",
                    subgroup=subgroup,
                )
            )
    return rows


def evaluate_validation_fold(
    merged: pd.DataFrame,
    *,
    validation_design: str,
    fold: str,
    train_seasons: list[str],
    test_season: str,
    base_template,
    base_features: list[str],
    model_cache: dict | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    train = merged[merged["SEASON"].isin(train_seasons)].copy()
    test = merged[merged["SEASON"].eq(test_season)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty validation fold: {fold}")

    blend = _fit_fold_blend(
        train,
        base_template,
        base_features,
        model_cache=model_cache,
    )
    base_model, advanced_model = _fit_fold_components(
        train,
        base_template,
        base_features,
        model_cache=model_cache,
    )
    base_probability = _base_probability(
        base_model, test, base_features
    )
    advanced_probability = _advanced_probability(
        advanced_model, test
    )
    combined_probability = _blend_probability(
        blend, [base_probability, advanced_probability]
    )
    movement = float(np.mean(np.abs(combined_probability - base_probability)))

    predictions = test[
        [
            "SEASON",
            "GAME_ID",
            "GAME_DATE",
            "TEAM_A",
            "TEAM_B",
            "TEAM_A_WON",
            "game_number_audit",
        ]
    ].copy()
    predictions["validation_design"] = validation_design
    predictions["fold"] = fold
    predictions["test_season"] = test_season
    predictions["playoff_round"] = predictions["GAME_ID"].map(
        playoff_round_from_game_id
    )
    predictions["base_probability"] = base_probability
    predictions[
        "base_plus_team_advanced_probability"
    ] = combined_probability
    predictions["advanced_probability_delta"] = (
        combined_probability - base_probability
    )
    target = predictions["TEAM_A_WON"]
    predictions["base_brier_loss"] = (
        base_probability - target
    ) ** 2
    predictions["advanced_brier_loss"] = (
        combined_probability - target
    ) ** 2
    predictions["brier_improvement"] = (
        predictions["base_brier_loss"]
        - predictions["advanced_brier_loss"]
    )

    rows = [
        metric_row(
            model="base_only",
            target=target,
            probability=base_probability,
            average_probability_movement=movement,
        ),
        metric_row(
            model="base_plus_team_advanced",
            target=target,
            probability=combined_probability,
            average_probability_movement=movement,
        ),
        *_evaluate_subgroups(predictions),
    ]
    metadata = {
        "validation_design": validation_design,
        "fold": fold,
        "train_seasons": ",".join(train_seasons),
        "test_season": test_season,
        "train_games": int(len(train)),
        "test_games": int(len(test)),
    }
    rows = [{**metadata, **row} for row in rows]
    return rows, predictions


def summarize_validation(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subgroup_values = (
        sorted(
            validation.loc[
                validation["record_type"].eq("subgroup"), "subgroup"
            ].unique()
        )
        if "subgroup" in validation
        else []
    )
    scopes = [
        (
            "all_games",
            "all_games",
            validation[validation["record_type"].eq("fold")],
        ),
        *[
            (
                "subgroup",
                subgroup,
                validation[
                    validation["record_type"].eq("subgroup")
                    & validation["subgroup"].eq(subgroup)
                ],
            )
            for subgroup in subgroup_values
        ],
    ]
    for scope, subgroup, records in scopes:
        pivot = records.pivot_table(
            index=["validation_design", "fold", "test_season"],
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
        for design, group in pivot.groupby(
            level="validation_design"
        ):
            flat = group.droplevel("validation_design")
            deltas = pd.DataFrame(
                {
                    metric: (
                        flat[(metric, "base_plus_team_advanced")]
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
            )
            brier_gains = -deltas["brier_score"]
            positive_brier = brier_gains.clip(lower=0)
            best_index = deltas["brier_score"].idxmin()
            worst_index = deltas["brier_score"].idxmax()
            rows.append(
                {
                    "scope": scope,
                    "subgroup": subgroup,
                    "validation_design": design,
                    "folds": int(len(deltas)),
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
                                "base_plus_team_advanced",
                            )
                        ].mean()
                    ),
                    "best_fold": str(best_index[0]),
                    "best_fold_test_season": str(best_index[1]),
                    "best_fold_brier_delta": float(
                        deltas.loc[best_index, "brier_score"]
                    ),
                    "worst_fold": str(worst_index[0]),
                    "worst_fold_test_season": str(worst_index[1]),
                    "worst_fold_brier_delta": float(
                        deltas.loc[worst_index, "brier_score"]
                    ),
                    "largest_share_of_positive_brier_gain": (
                        float(
                            positive_brier.max()
                            / positive_brier.sum()
                        )
                        if positive_brier.sum() > 0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def recommendation_from_summary(summary: pd.DataFrame) -> str:
    overall = summary[summary["scope"].eq("all_games")]
    if overall.empty:
        return "RESEARCH-ONLY: no validation folds were available."
    total_folds = int(overall["folds"].sum())
    brier_wins = int(overall["brier_improved_folds"].sum())
    log_wins = int(overall["log_loss_improved_folds"].sum())
    roc_wins = int(overall["roc_auc_improved_folds"].sum())
    probability_stable = (
        brier_wins > total_folds / 2
        or log_wins > total_folds / 2
    )
    ranking_not_consistently_damaged = (
        roc_wins >= total_folds / 2
        or overall["average_roc_auc_delta"].mean() >= -0.002
    )
    gains_not_concentrated = bool(
        overall["largest_share_of_positive_brier_gain"].max() <= 0.5
    )
    if (
        probability_stable
        and ranking_not_consistently_damaged
        and gains_not_concentrated
    ):
        return (
            "DEPLOY CANDIDATE: Base + Advanced improved probability quality "
            "across most folds without consistent ROC-AUC damage. Production "
            "consideration still requires a separate locked validation review."
        )
    return (
        "RESEARCH-ONLY: the advanced-feature gain is unstable, too small, "
        "concentrated in too few folds, or accompanied by recurring ROC-AUC "
        "damage."
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
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    recommendation: str,
    path: Path,
) -> None:
    fold_metrics = validation[
        validation["record_type"].eq("fold")
    ][
        [
            "validation_design",
            "test_season",
            "model",
            "games",
            "roc_auc",
            "brier_score",
            "log_loss",
            "accuracy",
            "f1",
            "expected_calibration_error",
            "average_probability_movement",
        ]
    ]
    lines = [
        "# Team Advanced Broader Validation",
        "",
        "Offline validation audit only. Production models and application behavior were unchanged.",
        "",
        "Base + Advanced uses the existing leakage-safe V3 feature builder. Every fold fits model clones and blend coefficients only from that fold's training seasons.",
        "",
        "## Fold Metrics",
        "",
        _markdown_table(fold_metrics),
        "",
        "## Stability Summary",
        "",
        _markdown_table(summary[summary["scope"].eq("all_games")]),
        "",
        "## Subgroup Summary",
        "",
        _markdown_table(summary[summary["scope"].eq("subgroup")]),
        "",
        "## Decision",
        "",
        recommendation,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_broader_validation(
    *,
    team_rows_path: str | Path = DEFAULT_TEAM_ROWS_PATH,
    base_training_path: str | Path = DEFAULT_BASE_TRAINING_PATH,
    series_context_path: str | Path = DEFAULT_SERIES_CONTEXT_PATH,
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    base = pd.read_csv(base_training_path)
    team_rows = pd.read_csv(team_rows_path)
    context = pd.read_csv(series_context_path)
    advanced = build_advanced_dataset(base, team_rows)
    merged = _merge_inputs(base, advanced, context)

    artifact = load_model(base_model_path)
    entry = get_model_entry_for_mode(
        artifact, PREDICTION_MODE_CURRENT
    )
    base_template = entry["pipeline"]
    base_features = list(entry["feature_columns"])

    validation_rows = []
    prediction_frames = []
    model_cache = {}
    for fold_config in validation_folds(ALL_SEASONS):
        print(
            "Evaluating "
            f"{fold_config['validation_design']}: "
            f"{fold_config['test_season']}"
        )
        rows, predictions = evaluate_validation_fold(
            merged,
            base_template=base_template,
            base_features=base_features,
            model_cache=model_cache,
            **fold_config,
        )
        validation_rows.extend(rows)
        prediction_frames.append(predictions)

    validation = pd.DataFrame(validation_rows)
    summary = summarize_validation(validation)
    recommendation = recommendation_from_summary(summary)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    examples = pd.concat(
        [
            predictions.nlargest(20, "brier_improvement").assign(
                example_type="advanced_helped"
            ),
            predictions.nsmallest(20, "brier_improvement").assign(
                example_type="advanced_hurt"
            ),
        ],
        ignore_index=True,
    )
    example_columns = [
        "example_type",
        "validation_design",
        "fold",
        "test_season",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_A",
        "TEAM_B",
        "TEAM_A_WON",
        "playoff_round",
        "game_number_audit",
        "base_probability",
        "base_plus_team_advanced_probability",
        "advanced_probability_delta",
        "brier_improvement",
    ]
    examples = examples[example_columns]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(
        output_dir / "team_advanced_broader_validation.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "team_advanced_broader_validation_summary.csv",
        index=False,
    )
    examples.to_csv(
        output_dir / "team_advanced_broader_validation_examples.csv",
        index=False,
    )
    _write_doc(validation, summary, recommendation, Path(doc_path))
    return validation, summary, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Broader season validation for V3 advanced deltas"
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
    validation, summary, recommendation = run_broader_validation(
        team_rows_path=args.team_rows_path,
        base_training_path=args.base_training_path,
        series_context_path=args.series_context_path,
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        doc_path=args.doc_path,
    )
    print(
        validation[validation["record_type"].eq("fold")].to_string(
            index=False
        )
    )
    print(summary.to_string(index=False))
    print(recommendation)


if __name__ == "__main__":
    main()
