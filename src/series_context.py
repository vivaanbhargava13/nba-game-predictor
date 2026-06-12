from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import PREDICTION_MODE_CURRENT, get_model_entry_for_mode, load_model
from .nba_data import load_playoff_games, season_range


DEFAULT_BASE_MODEL_PATH = Path("models/playoff_predictor.joblib")
DEFAULT_SERIES_CONTEXT_MODEL_PATH = Path("models/series_context_model.joblib")
DEFAULT_SERIES_CONTEXT_DATA_PATH = Path("data/processed/series_context_training.csv")
DEFAULT_SERIES_CONTEXT_METRICS_PATH = Path("data/processed/series_context_model_comparison.csv")
SERIES_CONTEXT_SCHEMA_VERSION = "2026-06-11-series-context-v1"
SERIES_CONTEXT_FEATURE_COLUMNS = [
    "game_number",
    "series_score_diff",
    "team_a_series_wins",
    "team_b_series_wins",
    "team_a_elimination_game",
    "team_b_elimination_game",
    "team_a_home",
    "team_a_trailing_series",
    "team_a_leading_series",
    "previous_game_winner",
    "previous_game_margin",
    "team_a_recent_series_wins",
    "team_b_recent_series_wins",
    "team_a_road_wins_in_series",
    "team_b_road_wins_in_series",
    "team_a_home_losses_in_series",
    "team_b_home_losses_in_series",
]
BLEND_FEATURE_COLUMNS = ["base_model_logit_probability", "series_context_logit_probability"]
RECENT_SERIES_WINDOW = 3


def _series_score_text(team_a: str, team_b: str, team_a_wins: int, team_b_wins: int) -> str:
    if team_a_wins == team_b_wins:
        return f"Series tied {team_a_wins}-{team_b_wins}"
    if team_a_wins > team_b_wins:
        return f"{team_a} leads {team_a_wins}-{team_b_wins}"
    return f"{team_b} leads {team_b_wins}-{team_a_wins}"


def _completed_game_result(game: dict, team_a: str, team_b: str) -> tuple[str | None, float | None]:
    winner = str(game.get("winner") or game.get("winner_abbr") or "").upper()
    margin = game.get("margin")
    if winner not in {team_a, team_b}:
        away = str(game.get("away_abbr") or "").upper()
        home = str(game.get("home_abbr") or "").upper()
        away_score = game.get("away_score")
        home_score = game.get("home_score")
        if away in {team_a, team_b} and home in {team_a, team_b} and away_score is not None and home_score is not None:
            away_score = float(away_score)
            home_score = float(home_score)
            winner = away if away_score > home_score else home
            margin = abs(away_score - home_score)
    if winner not in {team_a, team_b}:
        return None, None
    return winner, None if margin is None else float(margin)


def resolve_pregame_series_state_for_card(
    *,
    current_series_wins: dict[str, int],
    current_completed_games: list[dict] | None,
    requested_game_number: int,
    requested_away_team: str,
    requested_home_team: str,
    if_necessary: bool,
    series_teams: tuple[str, str] | list[str],
) -> dict:
    """Resolve the pregame state required for a scheduled playoff card to exist."""
    away = str(requested_away_team).upper()
    home = str(requested_home_team).upper()
    teams = tuple(str(team).upper() for team in series_teams)
    if len(teams) != 2 or set(teams) != {away, home}:
        teams = (away, home)

    wins = {team: max(0, int(current_series_wins.get(team, 0) or 0)) for team in teams}
    requested = min(7, max(1, int(requested_game_number)))
    next_unplayed = sum(wins.values()) + 1
    assumed_prior_winners: list[str] = []
    context_available = True
    conditional_context = False

    if requested < next_unplayed or max(wins.values()) >= 4:
        context_available = False
    elif requested > next_unplayed:
        steps = requested - next_unplayed
        possible_paths: list[tuple[list[str], dict[str, int]]] = []

        def enumerate_paths(path: list[str], path_wins: dict[str, int]) -> None:
            if len(path) == steps:
                possible_paths.append((path.copy(), path_wins.copy()))
                return
            for winner in teams:
                updated = path_wins.copy()
                updated[winner] += 1
                if updated[winner] >= 4:
                    continue
                enumerate_paths([*path, winner], updated)

        enumerate_paths([], wins)
        if len(possible_paths) == 1:
            assumed_prior_winners, wins = possible_paths[0]
            conditional_context = True
        else:
            context_available = False

    previous_game_winner = None
    previous_game_margin = None
    if context_available and assumed_prior_winners:
        previous_game_winner = assumed_prior_winners[-1]
    elif context_available and current_completed_games:
        previous_game_winner, previous_game_margin = _completed_game_result(
            current_completed_games[-1],
            away,
            home,
        )

    away_wins = wins.get(away, 0)
    home_wins = wins.get(home, 0)
    score_text = _series_score_text(away, home, away_wins, home_wins)
    if not context_available:
        note = "Playoff context pending prior result."
    elif conditional_context:
        if len(assumed_prior_winners) == 1:
            assumption = f"{assumed_prior_winners[0]} wins Game {next_unplayed}"
        else:
            game_numbers = " and ".join(str(number) for number in range(next_unplayed, requested))
            winner = assumed_prior_winners[0]
            if all(team == winner for team in assumed_prior_winners):
                assumption = f"{winner} wins Games {game_numbers}"
            else:
                assumption = ", ".join(
                    f"{winner_name} wins Game {game_number}"
                    for game_number, winner_name in zip(
                        range(next_unplayed, requested),
                        assumed_prior_winners,
                    )
                )
        note = (
            f"Conditional context: assumes {assumption}; the pregame state is "
            f"{score_text} before Game {requested}."
        )
    else:
        note = f"Current series context: {score_text} before Game {requested}."

    previous_game_winner_side = None
    if previous_game_winner == away:
        previous_game_winner_side = "team_a"
    elif previous_game_winner == home:
        previous_game_winner_side = "team_b"

    return {
        "resolved_game_number": requested,
        "resolved_team_a_series_wins": away_wins,
        "resolved_team_b_series_wins": home_wins,
        "resolved_series_score_text": score_text,
        "conditional_context": conditional_context,
        "context_available": context_available,
        "assumed_prior_winners": assumed_prior_winners,
        "previous_game_winner": previous_game_winner,
        "previous_game_winner_side": previous_game_winner_side,
        "previous_game_margin": previous_game_margin,
        "context_note": note,
        "if_necessary": bool(if_necessary),
    }


def resolve_completed_game_pregame_state_for_card(
    *,
    postgame_series_wins: dict[str, int],
    completed_games: list[dict] | None,
    requested_game_number: int,
    requested_away_team: str,
    requested_home_team: str,
    game_id: str = "",
    away_score: int | float | None = None,
    home_score: int | float | None = None,
) -> dict:
    """Reconstruct a completed game's series state immediately before tipoff."""
    away = str(requested_away_team).upper()
    home = str(requested_home_team).upper()
    requested = min(7, max(1, int(requested_game_number)))
    postgame_wins = {
        away: max(0, int(postgame_series_wins.get(away, 0) or 0)),
        home: max(0, int(postgame_series_wins.get(home, 0) or 0)),
    }

    loaded_game = {
        "game_id": str(game_id),
        "away_abbr": away,
        "home_abbr": home,
        "away_score": away_score,
        "home_score": home_score,
    }
    loaded_winner, _loaded_margin = _completed_game_result(loaded_game, away, home)
    pregame_wins = postgame_wins.copy()
    context_available = loaded_winner in pregame_wins and pregame_wins[loaded_winner] > 0
    if context_available:
        pregame_wins[loaded_winner] -= 1
        context_available = sum(pregame_wins.values()) == requested - 1

    prior_games: list[tuple[int, dict]] = []
    for index, game in enumerate(completed_games or []):
        candidate_id = str(game.get("game_id") or "")
        candidate_number = game.get("game_number")
        if game_id and candidate_id == str(game_id):
            continue
        if candidate_number is not None and int(candidate_number) >= requested:
            continue
        prior_games.append((index, game))
    prior_games.sort(
        key=lambda item: (
            int(item[1].get("game_number"))
            if item[1].get("game_number") is not None
            else requested,
            str(item[1].get("game_datetime") or item[1].get("game_date") or ""),
            item[0],
        )
    )

    previous_game_winner = None
    previous_game_margin = None
    if prior_games:
        previous_game_winner, previous_game_margin = _completed_game_result(
            prior_games[-1][1],
            away,
            home,
        )

    away_wins = pregame_wins[away]
    home_wins = pregame_wins[home]
    pregame_text = _series_score_text(away, home, away_wins, home_wins)
    postgame_text = _series_score_text(
        away,
        home,
        postgame_wins[away],
        postgame_wins[home],
    )
    previous_game_winner_side = None
    if previous_game_winner == away:
        previous_game_winner_side = "team_a"
    elif previous_game_winner == home:
        previous_game_winner_side = "team_b"

    return {
        "resolved_game_number": requested,
        "resolved_team_a_series_wins": away_wins,
        "resolved_team_b_series_wins": home_wins,
        "resolved_series_score_text": pregame_text,
        "pregame_series_state": pregame_text,
        "postgame_series_state": postgame_text,
        "displayed_series_state": postgame_text,
        "conditional_context": False,
        "context_available": context_available,
        "assumed_prior_winners": [],
        "previous_game_winner": previous_game_winner,
        "previous_game_winner_side": previous_game_winner_side,
        "previous_game_margin": previous_game_margin,
        "context_note": (
            f"Replay prediction for Game {requested} using the pregame series state: {pregame_text}."
            if context_available
            else "Completed-game pregame context unavailable."
        ),
        "completed_game_replay": True,
        "loaded_game_winner": loaded_winner,
    }


def _logit(values) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def _metrics(name: str, y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | str]:
    predicted = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def _game_records_for_pair(
    playoff_games: pd.DataFrame,
    team_a_id: int,
    team_b_id: int,
    before_date: pd.Timestamp | str | None = None,
) -> list[dict]:
    if playoff_games.empty:
        return []

    games = playoff_games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    if before_date is not None:
        games = games[games["GAME_DATE"] < pd.to_datetime(before_date)]

    records: list[dict] = []
    for game_id, group in games.groupby("GAME_ID", sort=False):
        team_ids = set(group["TEAM_ID"].astype(int))
        if team_ids != {int(team_a_id), int(team_b_id)} or len(group) != 2:
            continue
        team_a = group[group["TEAM_ID"].astype(int).eq(int(team_a_id))].iloc[0]
        team_b = group[group["TEAM_ID"].astype(int).eq(int(team_b_id))].iloc[0]
        records.append(
            {
                "game_id": str(game_id),
                "game_date": pd.Timestamp(team_a["GAME_DATE"]),
                "team_a_home": int("vs." in str(team_a["MATCHUP"])),
                "team_a_won": int(str(team_a["WL"]) == "W"),
                "team_a_margin": float(team_a["PTS"]) - float(team_b["PTS"]),
            }
        )
    return sorted(records, key=lambda row: (row["game_date"], row["game_id"]))


def build_series_state_features(
    *,
    game_number: int,
    team_a_series_wins: int,
    team_b_series_wins: int,
    team_a_home: bool,
    prior_games: list[dict] | None = None,
) -> dict[str, float]:
    prior_games = prior_games or []
    recent_games = prior_games[-RECENT_SERIES_WINDOW:]
    previous = prior_games[-1] if prior_games else None
    return {
        "game_number": float(game_number),
        "series_score_diff": float(team_a_series_wins - team_b_series_wins),
        "team_a_series_wins": float(team_a_series_wins),
        "team_b_series_wins": float(team_b_series_wins),
        "team_a_elimination_game": float(int(team_b_series_wins == 3)),
        "team_b_elimination_game": float(int(team_a_series_wins == 3)),
        "team_a_home": float(int(team_a_home)),
        "team_a_trailing_series": float(int(team_a_series_wins < team_b_series_wins)),
        "team_a_leading_series": float(int(team_a_series_wins > team_b_series_wins)),
        "previous_game_winner": 0.0 if previous is None else float(1 if previous["team_a_won"] else -1),
        "previous_game_margin": 0.0 if previous is None else float(previous["team_a_margin"]),
        "team_a_recent_series_wins": float(sum(int(game["team_a_won"]) for game in recent_games)),
        "team_b_recent_series_wins": float(sum(1 - int(game["team_a_won"]) for game in recent_games)),
        "team_a_road_wins_in_series": float(
            sum(int(game["team_a_won"] and not game["team_a_home"]) for game in prior_games)
        ),
        "team_b_road_wins_in_series": float(
            sum(int(not game["team_a_won"] and game["team_a_home"]) for game in prior_games)
        ),
        "team_a_home_losses_in_series": float(
            sum(int(not game["team_a_won"] and game["team_a_home"]) for game in prior_games)
        ),
        "team_b_home_losses_in_series": float(
            sum(int(game["team_a_won"] and not game["team_a_home"]) for game in prior_games)
        ),
    }


def reverse_series_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reversed_frame = frame.copy()
    reversed_frame["series_score_diff"] = -frame["series_score_diff"]
    reversed_frame["team_a_series_wins"] = frame["team_b_series_wins"]
    reversed_frame["team_b_series_wins"] = frame["team_a_series_wins"]
    reversed_frame["team_a_elimination_game"] = frame["team_b_elimination_game"]
    reversed_frame["team_b_elimination_game"] = frame["team_a_elimination_game"]
    reversed_frame["team_a_home"] = 1 - frame["team_a_home"]
    reversed_frame["team_a_trailing_series"] = frame["team_a_leading_series"]
    reversed_frame["team_a_leading_series"] = frame["team_a_trailing_series"]
    reversed_frame["previous_game_winner"] = -frame["previous_game_winner"]
    reversed_frame["previous_game_margin"] = -frame["previous_game_margin"]
    reversed_frame["team_a_recent_series_wins"] = frame["team_b_recent_series_wins"]
    reversed_frame["team_b_recent_series_wins"] = frame["team_a_recent_series_wins"]
    reversed_frame["team_a_road_wins_in_series"] = frame["team_b_road_wins_in_series"]
    reversed_frame["team_b_road_wins_in_series"] = frame["team_a_road_wins_in_series"]
    reversed_frame["team_a_home_losses_in_series"] = frame["team_b_home_losses_in_series"]
    reversed_frame["team_b_home_losses_in_series"] = frame["team_a_home_losses_in_series"]
    return reversed_frame


def build_series_context_dataset(
    seasons: list[str],
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        games = load_playoff_games(season, cache_dir)
        if games.empty:
            continue
        games = games.copy()
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
        paired_games = []
        for game_id, group in games.groupby("GAME_ID", sort=False):
            if len(group) != 2:
                continue
            ordered = group.sort_values("TEAM_ID").reset_index(drop=True)
            team_a = ordered.iloc[0]
            team_b = ordered.iloc[1]
            paired_games.append(
                {
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
        paired = pd.DataFrame(paired_games)
        if paired.empty:
            continue
        paired["PAIR"] = paired.apply(
            lambda row: tuple(sorted((int(row["TEAM_A_ID"]), int(row["TEAM_B_ID"])))),
            axis=1,
        )
        for _pair, series in paired.groupby("PAIR", sort=False):
            series = series.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
            history: list[dict] = []
            team_a_wins = 0
            team_b_wins = 0
            for game_index, game in series.iterrows():
                state = build_series_state_features(
                    game_number=game_index + 1,
                    team_a_series_wins=team_a_wins,
                    team_b_series_wins=team_b_wins,
                    team_a_home=bool(game["TEAM_A_HOME"]),
                    prior_games=history,
                )
                rows.append(
                    {
                        "SERIES_CONTEXT_SCHEMA_VERSION": SERIES_CONTEXT_SCHEMA_VERSION,
                        "SEASON": season,
                        "GAME_ID": str(game["GAME_ID"]),
                        "GAME_DATE": game["GAME_DATE"],
                        "TEAM_A_ID": int(game["TEAM_A_ID"]),
                        "TEAM_A": game["TEAM_A"],
                        "TEAM_B_ID": int(game["TEAM_B_ID"]),
                        "TEAM_B": game["TEAM_B"],
                        "TEAM_A_WON": int(game["TEAM_A_WON"]),
                        **state,
                    }
                )
                history.append(
                    {
                        "team_a_home": int(game["TEAM_A_HOME"]),
                        "team_a_won": int(game["TEAM_A_WON"]),
                        "team_a_margin": float(game["TEAM_A_MARGIN"]),
                    }
                )
                team_a_wins += int(game["TEAM_A_WON"])
                team_b_wins += 1 - int(game["TEAM_A_WON"])
    return pd.DataFrame(rows)


def build_runtime_series_context(
    *,
    season: str,
    team_a_id: int,
    team_b_id: int,
    prediction_date: pd.Timestamp | str,
    home_team_id: int,
    game_number: int,
    team_a_series_wins: int,
    team_b_series_wins: int,
    cache_dir: str | Path = "data/raw",
    team_a_abbr: str | None = None,
    team_b_abbr: str | None = None,
    resolved_context: dict | None = None,
) -> dict[str, float]:
    games = load_playoff_games(season, cache_dir)
    prior_games = _game_records_for_pair(games, team_a_id, team_b_id, prediction_date)
    features = build_series_state_features(
        game_number=game_number,
        team_a_series_wins=team_a_series_wins,
        team_b_series_wins=team_b_series_wins,
        team_a_home=int(team_a_id) == int(home_team_id),
        prior_games=prior_games,
    )
    if not resolved_context:
        return features

    previous_winner = str(resolved_context.get("previous_game_winner") or "").upper()
    if previous_winner and team_a_abbr and previous_winner == str(team_a_abbr).upper():
        features["previous_game_winner"] = 1.0
    elif previous_winner and team_b_abbr and previous_winner == str(team_b_abbr).upper():
        features["previous_game_winner"] = -1.0

    if resolved_context.get("conditional_context"):
        features["previous_game_margin"] = np.nan
        assumed = [str(winner).upper() for winner in resolved_context.get("assumed_prior_winners") or []]
        actual_recent = prior_games[-max(0, RECENT_SERIES_WINDOW - len(assumed)) :]
        recent_winners = [
            str(team_a_abbr).upper() if game["team_a_won"] else str(team_b_abbr).upper()
            for game in actual_recent
        ] + assumed[-RECENT_SERIES_WINDOW:]
        features["team_a_recent_series_wins"] = float(
            sum(winner == str(team_a_abbr).upper() for winner in recent_winners)
        )
        features["team_b_recent_series_wins"] = float(
            sum(winner == str(team_b_abbr).upper() for winner in recent_winners)
        )
    return features


def _context_estimator() -> CalibratedClassifierCV:
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)


def _augment_context(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    original = frame[SERIES_CONTEXT_FEATURE_COLUMNS].reset_index(drop=True)
    reversed_frame = reverse_series_context_frame(original)
    return (
        pd.concat([original, reversed_frame], ignore_index=True),
        pd.concat([target.reset_index(drop=True), 1 - target.reset_index(drop=True)], ignore_index=True),
    )


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
            reversed_frame[column] = -frame[column]
    if "home_team_A" in reversed_frame:
        reversed_frame["home_team_A"] = 1 - frame["home_team_A"]
    if "higher_seed_A" in reversed_frame:
        if {"seed_A", "seed_B"}.issubset(frame.columns):
            reversed_frame["higher_seed_A"] = (
                pd.to_numeric(frame["seed_B"], errors="coerce")
                < pd.to_numeric(frame["seed_A"], errors="coerce")
            ).astype(float)
        else:
            reversed_frame["higher_seed_A"] = 1 - frame["higher_seed_A"]
    return reversed_frame


def symmetric_model_probability(model, frame: pd.DataFrame, reverse_frame: pd.DataFrame) -> np.ndarray:
    direct = model.predict_proba(frame)[..., 1]
    reverse_complement = 1 - model.predict_proba(reverse_frame)[..., 1]
    return (np.asarray(direct, dtype=float) + np.asarray(reverse_complement, dtype=float)) / 2


def context_model_probability(model, frame: pd.DataFrame) -> np.ndarray:
    context = frame[SERIES_CONTEXT_FEATURE_COLUMNS]
    return symmetric_model_probability(model, context, reverse_series_context_frame(context))


def blended_probability(blend_model, base_probability, context_probability) -> np.ndarray:
    base_logit = _logit(base_probability)
    context_logit = _logit(context_probability)
    forward = np.column_stack([base_logit, context_logit])
    reverse = -forward
    direct = blend_model.predict_proba(forward)[:, 1]
    reverse_complement = 1 - blend_model.predict_proba(reverse)[:, 1]
    return (direct + reverse_complement) / 2


def _merge_base_and_context(base_frame: pd.DataFrame, context_frame: pd.DataFrame) -> pd.DataFrame:
    base = base_frame.copy()
    context = context_frame.copy()
    base["GAME_ID"] = base["GAME_ID"].astype(str)
    context["GAME_ID"] = context["GAME_ID"].astype(str)
    base = base.drop(
        columns=[column for column in SERIES_CONTEXT_FEATURE_COLUMNS if column in base.columns],
        errors="ignore",
    )
    return base.merge(
        context[
            [
                "SEASON",
                "GAME_ID",
                *SERIES_CONTEXT_FEATURE_COLUMNS,
            ]
        ],
        on=["SEASON", "GAME_ID"],
        how="inner",
        validate="one_to_one",
    )


def _fit_context_model(frame: pd.DataFrame):
    x, y = _augment_context(frame, frame["TEAM_A_WON"].astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _context_estimator().fit(x, y)


def _context_coefficients(model: CalibratedClassifierCV) -> dict[str, float]:
    coefficients = []
    for calibrated_classifier in model.calibrated_classifiers_:
        estimator = calibrated_classifier.estimator
        classifier = estimator.named_steps["classifier"]
        coefficients.append(classifier.coef_[0].astype(float))
    average = np.mean(np.vstack(coefficients), axis=0)
    return dict(zip(SERIES_CONTEXT_FEATURE_COLUMNS, average))


def train_series_context_model(
    *,
    base_training_path: str | Path = "data/processed/training_matchups_regular_season.csv",
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    cache_dir: str | Path = "data/raw",
    model_path: str | Path = DEFAULT_SERIES_CONTEXT_MODEL_PATH,
    dataset_path: str | Path = DEFAULT_SERIES_CONTEXT_DATA_PATH,
    metrics_path: str | Path = DEFAULT_SERIES_CONTEXT_METRICS_PATH,
    context_train_seasons: list[str] | None = None,
    blend_train_seasons: list[str] | None = None,
    test_seasons: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    context_train_seasons = context_train_seasons or season_range("2015-16", "2022-23")
    blend_train_seasons = blend_train_seasons or [
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
    ]
    test_seasons = test_seasons or ["2023-24", "2024-25"]
    all_seasons = sorted(set(context_train_seasons + test_seasons))

    context_frame = build_series_context_dataset(all_seasons, cache_dir)
    base_frame = pd.read_csv(base_training_path)
    merged = _merge_base_and_context(base_frame, context_frame)
    if merged.empty:
        raise ValueError("No matching base and series-context training rows were found.")

    base_bundle = load_model(base_model_path)
    base_entry = get_model_entry_for_mode(base_bundle, PREDICTION_MODE_CURRENT)
    base_pipeline = base_entry["pipeline"]
    base_features = list(base_entry["feature_columns"])

    blend_rows = []
    for season in blend_train_seasons:
        fold_train_seasons = [candidate for candidate in context_train_seasons if candidate != season]
        train = merged[merged["SEASON"].isin(fold_train_seasons)]
        validation = merged[merged["SEASON"].eq(season)]
        if train.empty or validation.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fold_base = clone(base_pipeline).fit(train[base_features], train["TEAM_A_WON"].astype(int))
        fold_context = _fit_context_model(train)
        base_probability = symmetric_model_probability(
            fold_base,
            validation[base_features],
            _reverse_base_frame(validation, base_features),
        )
        context_probability = context_model_probability(fold_context, validation)
        blend_rows.append(
            pd.DataFrame(
                {
                    "base_probability": base_probability,
                    "context_probability": context_probability,
                    "target": validation["TEAM_A_WON"].astype(int).to_numpy(),
                }
            )
        )
    if not blend_rows:
        raise ValueError("Could not create out-of-season blend training predictions.")
    blend_training = pd.concat(blend_rows, ignore_index=True)
    blend_x = np.column_stack(
        [
            _logit(blend_training["base_probability"]),
            _logit(blend_training["context_probability"]),
        ]
    )
    blend_y = blend_training["target"].astype(int).to_numpy()
    blend_x = np.vstack([blend_x, -blend_x])
    blend_y = np.concatenate([blend_y, 1 - blend_y])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        blend_model = LogisticRegression(
            fit_intercept=False,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            solver="liblinear",
        ).fit(blend_x, blend_y)

    context_train = merged[merged["SEASON"].isin(context_train_seasons)]
    test = merged[merged["SEASON"].isin(test_seasons)]
    context_model = _fit_context_model(context_train)
    base_probability = symmetric_model_probability(
        base_pipeline,
        test[base_features],
        _reverse_base_frame(test, base_features),
    )
    context_probability = context_model_probability(context_model, test)
    blend_probability = blended_probability(blend_model, base_probability, context_probability)
    metrics = pd.DataFrame(
        [
            _metrics("base_only", test["TEAM_A_WON"], base_probability),
            _metrics("series_context_only", test["TEAM_A_WON"], context_probability),
            _metrics("blended", test["TEAM_A_WON"], blend_probability),
        ]
    )
    metrics["train_seasons"] = ",".join(context_train_seasons)
    metrics["blend_train_seasons"] = ",".join(blend_train_seasons)
    metrics["test_seasons"] = ",".join(test_seasons)

    artifact = {
        "schema_version": SERIES_CONTEXT_SCHEMA_VERSION,
        "feature_columns": SERIES_CONTEXT_FEATURE_COLUMNS,
        "context_model": context_model,
        "context_standardized_coefficients": _context_coefficients(context_model),
        "blend_model": blend_model,
        "blend_feature_columns": BLEND_FEATURE_COLUMNS,
        "blend_coefficients": dict(zip(BLEND_FEATURE_COLUMNS, blend_model.coef_[0].astype(float))),
        "context_train_seasons": context_train_seasons,
        "blend_train_seasons": blend_train_seasons,
        "test_seasons": test_seasons,
        "base_model_path": str(base_model_path),
        "base_feature_columns": base_features,
        "metrics": metrics.to_dict(orient="records"),
        "recent_series_window": RECENT_SERIES_WINDOW,
    }

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    context_frame.to_csv(dataset_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return artifact, metrics


def load_series_context_model(path: str | Path = DEFAULT_SERIES_CONTEXT_MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_series_context_probability(artifact: dict, features: dict[str, float]) -> float:
    frame = pd.DataFrame([features], columns=artifact["feature_columns"])
    return float(context_model_probability(artifact["context_model"], frame)[0])


def apply_series_context_probability(
    *,
    base_probability: float,
    season: str,
    team_a_id: int,
    team_b_id: int,
    prediction_date: pd.Timestamp | str,
    home_team_id: int,
    game_number: int,
    team_a_series_wins: int,
    team_b_series_wins: int,
    cache_dir: str | Path = "data/raw",
    model_path: str | Path = DEFAULT_SERIES_CONTEXT_MODEL_PATH,
    team_a_abbr: str | None = None,
    team_b_abbr: str | None = None,
    resolved_context: dict | None = None,
) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        return {
            "probability": float(base_probability),
            "base_probability": float(base_probability),
            "context_probability": None,
            "applied": False,
            "note": "Series-context model unavailable; using base game probability.",
            "error": "artifact_missing",
            "features": None,
        }
    try:
        artifact = load_series_context_model(model_path)
        features = build_runtime_series_context(
            season=season,
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            prediction_date=prediction_date,
            home_team_id=home_team_id,
            game_number=game_number,
            team_a_series_wins=team_a_series_wins,
            team_b_series_wins=team_b_series_wins,
            cache_dir=cache_dir,
            team_a_abbr=team_a_abbr,
            team_b_abbr=team_b_abbr,
            resolved_context=resolved_context,
        )
        context_probability = predict_series_context_probability(artifact, features)
        probability = float(
            blended_probability(
                artifact["blend_model"],
                np.array([base_probability]),
                np.array([context_probability]),
            )[0]
        )
    except Exception as exc:
        return {
            "probability": float(base_probability),
            "base_probability": float(base_probability),
            "context_probability": None,
            "applied": False,
            "note": "Series-context model unavailable; using base game probability.",
            "error": str(exc),
            "features": None,
        }
    return {
        "probability": probability,
        "base_probability": float(base_probability),
        "context_probability": context_probability,
        "applied": True,
        "note": "Game probability blends the base model with learned historical playoff series context.",
        "error": None,
        "features": features,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the historical playoff series-context model")
    parser.add_argument("--base-training-path", default="data/processed/training_matchups_regular_season.csv")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--model-path", default=DEFAULT_SERIES_CONTEXT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_SERIES_CONTEXT_DATA_PATH)
    parser.add_argument("--metrics-path", default=DEFAULT_SERIES_CONTEXT_METRICS_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact, metrics = train_series_context_model(
        base_training_path=args.base_training_path,
        base_model_path=args.base_model_path,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        metrics_path=args.metrics_path,
    )
    print(metrics.to_string(index=False))
    print(f"Blend coefficients: {artifact['blend_coefficients']}")
    print(f"Saved series-context model to {args.model_path}")


if __name__ == "__main__":
    main()
