from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

DEFAULT_PREDICTIONS_PATH = Path(
    "data/processed/momentum_state_experiment_predictions.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/processed")
MIN_ROC_GAMES = 30
MODEL_COLUMNS = {
    "base_only": "base_probability",
    "base_plus_series": "base_plus_series_probability",
    "base_plus_momentum": "base_plus_momentum_probability",
    "base_plus_series_plus_momentum": (
        "base_plus_series_plus_momentum_probability"
    ),
}


def orient_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create one focal-team row for each side of every held-out game."""
    rows = []
    for _, game in predictions.iterrows():
        for side in ("a", "b"):
            is_a = side == "a"
            focal_wins = int(
                game["team_a_series_wins_audit"]
                if is_a
                else game["team_b_series_wins_audit"]
            )
            opponent_wins = int(
                game["team_b_series_wins_audit"]
                if is_a
                else game["team_a_series_wins_audit"]
            )
            previous_winner = int(game["previous_game_winner_audit"])
            previous_probability_a = float(
                game["previous_expected_probability_a_audit"]
            )
            previous_margin_a = float(game["previous_margin_audit"])
            previous_residual_a = float(game["previous_residual_margin_a_audit"])
            row = {
                "SEASON": game["SEASON"],
                "GAME_ID": str(game["GAME_ID"]),
                "GAME_DATE": game["GAME_DATE"],
                "focal_team": game["TEAM_A"] if is_a else game["TEAM_B"],
                "opponent": game["TEAM_B"] if is_a else game["TEAM_A"],
                "target": int(game["TEAM_A_WON"]) if is_a else 1 - int(game["TEAM_A_WON"]),
                "game_number": int(game["game_number_audit"]),
                "series_wins": focal_wins,
                "opponent_series_wins": opponent_wins,
                "is_home": (
                    int(game["team_a_home_audit"])
                    if is_a
                    else 1 - int(game["team_a_home_audit"])
                ),
                "focal_seed": float(game["seed_A"] if is_a else game["seed_B"]),
                "opponent_seed": float(game["seed_B"] if is_a else game["seed_A"]),
                "won_previous_game": int(
                    previous_winner == (1 if is_a else -1)
                ),
                "won_previous_two_games": int(
                    game[
                        "team_a_won_previous_two_audit"
                        if is_a
                        else "team_b_won_previous_two_audit"
                    ]
                ),
                "previous_margin": previous_margin_a if is_a else -previous_margin_a,
                "previous_absolute_margin": abs(previous_margin_a),
                "previous_expected_probability": (
                    previous_probability_a if is_a else 1 - previous_probability_a
                ),
                "previous_residual_margin": (
                    previous_residual_a if is_a else -previous_residual_a
                ),
                "last_two_positive_residuals": int(
                    game[
                        "team_a_last_two_positive_residuals_audit"
                        if is_a
                        else "team_b_last_two_positive_residuals_audit"
                    ]
                ),
                "momentum_state": float(
                    game[
                        "team_a_momentum_state_audit"
                        if is_a
                        else "team_b_momentum_state_audit"
                    ]
                ),
                "home_court_swing": int(game["home_court_swing_audit"]),
            }
            for model_name, column in MODEL_COLUMNS.items():
                probability = float(game[column])
                row[model_name] = probability if is_a else 1 - probability
            rows.append(row)
    return pd.DataFrame(rows)


def subgroup_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    has_previous = frame["game_number"] > 1
    return {
        "game_6_or_7": frame["game_number"].isin([6, 7]),
        "elimination_game": frame["opponent_series_wins"].eq(3),
        "closeout_game": frame["series_wins"].eq(3),
        "team_trailing_series": frame["series_wins"] < frame["opponent_series_wins"],
        "team_leading_series": frame["series_wins"] > frame["opponent_series_wins"],
        "team_down_3_1_or_3_2": (
            frame["opponent_series_wins"].eq(3)
            & frame["series_wins"].isin([1, 2])
        ),
        "team_up_3_1_or_3_2": (
            frame["series_wins"].eq(3)
            & frame["opponent_series_wins"].isin([1, 2])
        ),
        "team_won_previous_game": has_previous & frame["won_previous_game"].eq(1),
        "team_won_previous_2_games": frame["won_previous_two_games"].eq(1),
        "positive_residual_margin_previous_game": (
            has_previous & frame["previous_residual_margin"].gt(0)
        ),
        "positive_residual_margin_last_2_games": (
            frame["last_two_positive_residuals"].eq(1)
        ),
        "won_previous_as_lower_base_probability_team": (
            has_previous
            & frame["won_previous_game"].eq(1)
            & frame["previous_expected_probability"].lt(0.5)
        ),
        "previous_game_margin_at_least_10": (
            has_previous & frame["previous_absolute_margin"].ge(10)
        ),
        "previous_game_margin_at_most_5": (
            has_previous & frame["previous_absolute_margin"].le(5)
        ),
        "home_court_swing_game": has_previous & frame["home_court_swing"].eq(1),
        "lower_seed_with_positive_momentum": (
            frame["focal_seed"].gt(frame["opponent_seed"])
            & frame["momentum_state"].gt(0)
        ),
        "higher_seed_with_negative_momentum": (
            frame["focal_seed"].lt(frame["opponent_seed"])
            & frame["momentum_state"].lt(0)
        ),
    }


def _safe_roc(
    target: pd.Series,
    probability: pd.Series,
    unique_games: int,
) -> tuple[float, str]:
    class_counts = target.value_counts()
    if unique_games < MIN_ROC_GAMES:
        return np.nan, f"too_small_for_roc_n_lt_{MIN_ROC_GAMES}"
    if len(class_counts) < 2 or class_counts.min() < 5:
        return np.nan, "too_few_outcomes_for_roc"
    return float(roc_auc_score(target, probability)), ""


def audit_subgroups(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oriented = orient_predictions(predictions)
    audit_rows = []
    example_rows = []
    for subgroup, mask in subgroup_masks(oriented).items():
        group = oriented[mask].copy()
        if group.empty:
            continue
        unique_games = int(
            group[["SEASON", "GAME_ID"]].drop_duplicates().shape[0]
        )
        base_brier = brier_score_loss(group["target"], group["base_only"])
        base_log_loss = log_loss(group["target"], group["base_only"], labels=[0, 1])
        series_brier = brier_score_loss(
            group["target"], group["base_plus_series"]
        )
        series_log_loss = log_loss(
            group["target"], group["base_plus_series"], labels=[0, 1]
        )
        for model_name in MODEL_COLUMNS:
            probability = group[model_name]
            roc_auc, warning = _safe_roc(
                group["target"], probability, unique_games
            )
            brier = float(brier_score_loss(group["target"], probability))
            loss = float(log_loss(group["target"], probability, labels=[0, 1]))
            if model_name == "base_plus_momentum":
                no_momentum_brier = base_brier
                no_momentum_log_loss = base_log_loss
            elif model_name == "base_plus_series_plus_momentum":
                no_momentum_brier = series_brier
                no_momentum_log_loss = series_log_loss
            else:
                no_momentum_brier = brier
                no_momentum_log_loss = loss
            audit_rows.append(
                {
                    "subgroup": subgroup,
                    "model": model_name,
                    "n_games": unique_games,
                    "n_team_observations": int(len(group)),
                    "brier_score": brier,
                    "log_loss": loss,
                    "roc_auc": roc_auc,
                    "average_probability_movement_vs_base": float(
                        (probability - group["base_only"]).abs().mean()
                    ),
                    "brier_improvement_vs_base": float(base_brier - brier),
                    "log_loss_improvement_vs_base": float(base_log_loss - loss),
                    "improves_brier_or_log_loss": bool(
                        brier < base_brier or loss < base_log_loss
                    ),
                    "brier_improvement_vs_no_momentum": float(
                        no_momentum_brier - brier
                    ),
                    "log_loss_improvement_vs_no_momentum": float(
                        no_momentum_log_loss - loss
                    ),
                    "momentum_improves_brier_or_log_loss": bool(
                        brier < no_momentum_brier
                        or loss < no_momentum_log_loss
                    ),
                    "small_sample_warning": warning,
                }
            )

        group["full_absolute_error"] = (
            group["base_plus_series_plus_momentum"] - group["target"]
        ).abs()
        group["base_absolute_error"] = (group["base_only"] - group["target"]).abs()
        group["error_change_vs_base"] = (
            group["full_absolute_error"] - group["base_absolute_error"]
        )
        for outcome, selected in (
            ("helped", group.nsmallest(3, "error_change_vs_base")),
            ("hurt", group.nlargest(3, "error_change_vs_base")),
        ):
            for _, row in selected.iterrows():
                example_rows.append(
                    {
                        "subgroup": subgroup,
                        "effect": outcome,
                        "season": row["SEASON"],
                        "game_id": row["GAME_ID"],
                        "game_date": row["GAME_DATE"],
                        "focal_team": row["focal_team"],
                        "opponent": row["opponent"],
                        "target": int(row["target"]),
                        "base_probability": float(row["base_only"]),
                        "base_plus_series_plus_momentum_probability": float(
                            row["base_plus_series_plus_momentum"]
                        ),
                        "absolute_error_change_vs_base": float(
                            row["error_change_vs_base"]
                        ),
                    }
                )

    audit = pd.DataFrame(audit_rows)
    examples = pd.DataFrame(example_rows)
    momentum_rows = audit[
        audit["model"].isin(
            ["base_plus_momentum", "base_plus_series_plus_momentum"]
        )
    ].copy()
    robust = momentum_rows[
        momentum_rows["n_games"].ge(MIN_ROC_GAMES)
        & momentum_rows["brier_improvement_vs_no_momentum"].gt(0)
        & momentum_rows["log_loss_improvement_vs_no_momentum"].gt(0)
    ]
    related_groups = {
        "late_series": {"game_6_or_7", "elimination_game"},
        "trailing": {
            "team_trailing_series",
            "team_down_3_1_or_3_2",
            "won_previous_as_lower_base_probability_team",
        },
        "residual": {
            "positive_residual_margin_previous_game",
            "positive_residual_margin_last_2_games",
        },
    }
    robust_related = [
        family
        for family, names in related_groups.items()
        if len(set(robust["subgroup"]) & names) >= 2
    ]
    if robust_related:
        recommendation = (
            "4. worth building a gated model: momentum improved both Brier and log "
            f"loss across related robust subgroup families: {', '.join(robust_related)}."
        )
    elif not robust.empty:
        recommendation = (
            "3. useful in specific contexts but not production-ready: isolated "
            "subgroups improved, but the pattern was not robust across related contexts."
        )
    elif (momentum_rows["momentum_improves_brier_or_log_loss"]).any():
        recommendation = (
            "2. useful only as explanation: a few small or one-metric subgroup gains "
            "exist, but none are robust enough for gated prediction."
        )
    else:
        recommendation = (
            "1. no value anywhere: the momentum layer did not improve Brier or log "
            "loss in any audited subgroup."
        )
    return audit, examples, recommendation


def run_subgroup_audit(
    *,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    predictions = pd.read_csv(predictions_path)
    predictions = predictions[
        predictions["SEASON"].isin(["2023-24", "2024-25"])
    ].copy()
    audit, examples, recommendation = audit_subgroups(predictions)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "momentum_subgroup_audit.csv", index=False)
    examples.to_csv(output_dir / "momentum_subgroup_examples.csv", index=False)
    print(audit.to_string(index=False))
    print(recommendation)
    return audit, examples, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit playoff momentum subgroups")
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_subgroup_audit(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
