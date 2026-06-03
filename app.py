from __future__ import annotations

import os
import json
import html
import queue
import re
import threading
import time
from contextlib import contextmanager
from datetime import date
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("reports/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from src.model import (
    DIFF_COLUMNS,
    PREDICTION_MODE_CURRENT,
    PREDICTION_MODE_PLAYOFF,
    SERIES_CONTEXT_FEATURES,
    extract_feature_importances,
    get_model_entry_for_mode,
    load_model,
)
from src.nba_data import FEATURE_COLUMNS, load_team_stats
from src.live_games import load_live_games, nba_season_from_date
from src.predictor import DEFAULT_CACHE_DIR, DEFAULT_MODEL_PATH, _predict_probability, validate_series_score


SEASON_OPTIONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2025, 2014, -1)]
DEFAULT_SEASON = "2024-25"
DEFAULT_SEASON_TYPE = "Regular Season"
PROCESSED_DIR = Path("data/processed")
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "feature_importances.csv"
MODEL_COMPARISON_PATH = PROCESSED_DIR / "model_comparison.csv"
MODEL_CALIBRATION_PATH = PROCESSED_DIR / "model_calibration.csv"
PREDICTION_EXPLANATIONS_PATH = PROCESSED_DIR / "prediction_explanations.csv"
PREDICTION_CHATS_PATH = PROCESSED_DIR / "prediction_chats.csv"
CHAT_PROVIDER_GEMINI = "Gemini"
CHAT_PROVIDER_OPENAI = "OpenAI"
CHAT_PROVIDER_FALLBACK = "fallback"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-lite"
GEMINI_QUOTA_FALLBACK_NOTICE = "Gemini quota unavailable; using fallback explanation."
TEAM_A_COLOR = "#1F8A70"
TEAM_B_COLOR = "#D97706"
INK = "#172033"
MUTED = "#667085"
SURFACE = "#FFFFFF"
LINE = "#D9DEE7"
NBA_TEAM_COLORS = {
    "ATL": ("#E03A3E", "#C1D32F"),
    "BKN": ("#000000", "#777777"),
    "BOS": ("#007A33", "#BA9653"),
    "CHA": ("#1D1160", "#00788C"),
    "CHI": ("#CE1141", "#000000"),
    "CLE": ("#860038", "#FDBB30"),
    "DAL": ("#00538C", "#B8C4CA"),
    "DEN": ("#0E2240", "#FEC524"),
    "DET": ("#1D42BA", "#C8102E"),
    "GSW": ("#1D428A", "#FFC72C"),
    "HOU": ("#CE1141", "#000000"),
    "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#1D428A"),
    "LAL": ("#552583", "#FDB927"),
    "MEM": ("#5D76A9", "#12173F"),
    "MIA": ("#98002E", "#F9A01B"),
    "MIL": ("#00471B", "#EEE1C6"),
    "MIN": ("#0C2340", "#78BE20"),
    "NOP": ("#0C2340", "#C8102E"),
    "NYK": ("#F58426", "#006BB6"),
    "OKC": ("#007AC1", "#EF3B24"),
    "ORL": ("#0077C0", "#C4CED4"),
    "PHI": ("#006BB6", "#ED174C"),
    "PHX": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#000000"),
    "SAC": ("#5A2D81", "#63727A"),
    "SAS": ("#C4CED4", "#000000"),
    "TOR": ("#CE1141", "#000000"),
    "UTA": ("#002B5C", "#F9A01B"),
    "WAS": ("#002B5C", "#E31837"),
}


def perf_diagnostics_enabled() -> bool:
    raw_value = os.getenv("PLAYOFF_PREDICTOR_PERF") or os.getenv("STREAMLIT_PERF") or os.getenv("DEBUG_PERF") or ""
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def perf_timer(phase: str):
    if not perf_diagnostics_enabled():
        yield
        return

    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        print(f"[perf] {phase}: {elapsed_ms:.1f} ms")


def _path_version(path: str | Path) -> tuple[int, int]:
    try:
        stat = Path(path).stat()
        return int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return 0, 0


def model_artifact_version(model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple[int, int]:
    return _path_version(model_path)


@st.cache_resource
def cached_model(model_path: str, model_version: tuple[int, int] | None = None):
    del model_version
    with perf_timer("model artifact load"):
        return load_model(model_path)


@st.cache_data(show_spinner=False)
def cached_team_stats(season: str, season_type: str) -> pd.DataFrame:
    with perf_timer("processed data load"):
        return load_team_stats(season, cache_dir=DEFAULT_CACHE_DIR, season_type=season_type)


@st.cache_data(ttl=120, show_spinner=False)
def cached_live_games() -> dict:
    with perf_timer("live/latest/upcoming games fetch"):
        return load_live_games()


def safe_live_games_payload(fetcher=cached_live_games, timeout_seconds: float = 1.5) -> dict:
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def fetch_in_background() -> None:
        try:
            result_queue.put(fetcher())
        except Exception as exc:
            result_queue.put(exc)

    thread = threading.Thread(target=fetch_in_background, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return {"latest": [], "upcoming": [], "error": "Live games unavailable right now."}

    try:
        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            raise result
        payload = result
        return payload if isinstance(payload, dict) else {"latest": [], "upcoming": [], "error": "Live games unavailable"}
    except Exception as exc:
        return {"latest": [], "upcoming": [], "error": str(exc)}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = hex_color.strip().lstrip("#")
    if len(cleaned) != 6:
        return (31, 138, 112)
    return tuple(int(cleaned[index : index + 2], 16) for index in (0, 2, 4))


def readable_text_color(background_hex: str) -> str:
    red, green, blue = _hex_to_rgb(background_hex)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    return "#172033" if luminance > 0.62 else "#FFFFFF"


def _relative_luminance(hex_color: str) -> float:
    red, green, blue = [channel / 255 for channel in _hex_to_rgb(hex_color)]
    values = []
    for channel in (red, green, blue):
        values.append(channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def _contrast_ratio(color_a: str, color_b: str = "#FFFFFF") -> float:
    lum_a = _relative_luminance(color_a)
    lum_b = _relative_luminance(color_b)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def get_team_colors(abbreviation: str) -> dict[str, str]:
    primary, secondary = NBA_TEAM_COLORS.get(str(abbreviation).upper(), (TEAM_A_COLOR, TEAM_B_COLOR))
    text_accent = primary if _contrast_ratio(primary, "#FFFFFF") >= 3.0 else secondary
    if _contrast_ratio(text_accent, "#FFFFFF") < 3.0:
        text_accent = INK
    return {
        "primary": primary,
        "secondary": secondary,
        "text": readable_text_color(primary),
        "accent_text": text_accent,
    }


def probability_chart(team_a_label: str, team_b_label: str, team_a_probability: float):
    chart_data = pd.DataFrame(
        {
            "Team": [team_a_label, team_b_label],
            "Win Probability": [team_a_probability, 1 - team_a_probability],
        }
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    sns.barplot(
        data=chart_data,
        x="Team",
        y="Win Probability",
        hue="Team",
        palette=[TEAM_A_COLOR, TEAM_B_COLOR],
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Win Probability")
    ax.set_title("Win Probability", color=INK, fontsize=15, weight="bold", pad=14)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", color="#E5E7EB", linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)

    for patch, probability in zip(ax.patches, chart_data["Win Probability"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.025,
            f"{probability:.1%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=INK,
        )

    fig.tight_layout()
    return fig


def load_feature_importances(
    model_bundle: dict,
    feature_columns: list[str],
    prediction_context_mode: str | None = None,
) -> pd.DataFrame:
    model_name = str(model_bundle.get("metrics", {}).get("model", "Current Model"))
    if FEATURE_IMPORTANCE_PATH.exists():
        importances = _read_processed_csv(FEATURE_IMPORTANCE_PATH)
        if {"feature", "importance"}.issubset(importances.columns):
            if prediction_context_mode and "prediction_context_mode" in importances.columns:
                importances = importances[importances["prediction_context_mode"].eq(prediction_context_mode)]
            importances = importances[importances["feature"].isin(feature_columns)].copy()
            importances["importance"] = pd.to_numeric(importances["importance"], errors="coerce")
            if not importances.empty and importances["importance"].notna().any():
                return importances.sort_values("importance", ascending=False).reset_index(drop=True)

    return extract_feature_importances(model_bundle["pipeline"], model_name, feature_columns)


def all_saved_feature_columns(model_bundle: dict) -> list[str]:
    columns: list[str] = []
    for key in ["feature_columns"]:
        columns.extend(model_bundle.get(key, []) or [])
    for group_key in ["production_models", "models"]:
        for entry in (model_bundle.get(group_key) or {}).values():
            columns.extend(entry.get("feature_columns", []) or [])
    return list(dict.fromkeys(columns))


def _abbr_from_label(label: str) -> str:
    if "(" in label and label.endswith(")"):
        return label.rsplit("(", 1)[1][:-1]
    return label


def valid_team_a_series_win_options(game_number: int) -> list[int]:
    expected_total = int(game_number) - 1
    minimum = max(0, expected_total - 3)
    maximum = min(3, expected_total)
    return list(range(minimum, maximum + 1))


def series_status_text(
    selected_abbr: str,
    opponent_abbr: str,
    selected_wins: int,
    opponent_wins: int,
) -> tuple[str, str]:
    selected_wins = int(selected_wins)
    opponent_wins = int(opponent_wins)
    if selected_wins >= 4 or opponent_wins >= 4:
        if selected_wins > opponent_wins:
            return f"{selected_abbr} wins {selected_wins}-{opponent_wins}", selected_abbr
        if opponent_wins > selected_wins:
            return f"{opponent_abbr} wins {opponent_wins}-{selected_wins}", opponent_abbr
    if selected_wins == opponent_wins:
        return f"Series tied {selected_wins}-{opponent_wins}", "Tied"
    if selected_wins > opponent_wins:
        return f"{selected_abbr} leads {selected_wins}-{opponent_wins}", selected_abbr
    return f"{opponent_abbr} leads {opponent_wins}-{selected_wins}", opponent_abbr


def feature_importance_chart(importances: pd.DataFrame):
    top_importances = importances.dropna(subset=["importance"]).head(10).copy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#FFFFFF")
    if top_importances.empty:
        ax.text(0.5, 0.5, "Feature importances unavailable", ha="center", va="center")
        ax.axis("off")
        return fig

    sns.barplot(
        data=top_importances,
        x="importance",
        y="feature",
        hue="feature",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title("Top 10 Feature Importances", color=INK, fontsize=15, weight="bold", pad=14)
    ax.grid(axis="x", color="#E5E7EB", linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


NEUTRAL_FEATURE_VALUES = {
    "home_team_A": 0.5,
    "higher_seed_A": 0.5,
    "game_number": 1.0,
    "elimination_game": 0.0,
    "series_score_diff": 0.0,
}

SEMANTIC_SERIES_FEATURES = {"series_score_diff"}


def _neutral_feature_value(feature: str) -> float:
    return NEUTRAL_FEATURE_VALUES.get(feature, 0.0)


def _feature_probability_contributions(
    pipeline,
    features: dict[str, float],
    feature_columns: list[str],
    full_probability: float,
) -> dict[str, float]:
    contributions: dict[str, float] = {}
    base_row = {column: features.get(column) for column in feature_columns}
    base_frame = pd.DataFrame([base_row], columns=feature_columns)
    base_probability = float(pipeline.predict_proba(base_frame)[0, 1])
    for feature in feature_columns:
        neutralized = dict(base_row)
        neutralized[feature] = _neutral_feature_value(feature)
        neutral_frame = pd.DataFrame([neutralized], columns=feature_columns)
        neutral_probability = float(pipeline.predict_proba(neutral_frame)[0, 1])
        contributions[feature] = base_probability - neutral_probability
    return contributions


def game_win_prediction_features(
    features: dict[str, float],
    prediction_context_mode: str,
) -> dict[str, float]:
    """Return feature values used for game-win prediction.

    The actual series score is semantic context for the series outcome and chat/debug.
    It should not act as a learned single-game win factor.
    """
    game_features = dict(features)
    if prediction_context_mode == PREDICTION_MODE_PLAYOFF:
        game_features["series_score_diff"] = 0.0
        game_features["game_number"] = float(NEUTRAL_FEATURE_VALUES["game_number"])
        game_features["elimination_game"] = float(NEUTRAL_FEATURE_VALUES["elimination_game"])
    return game_features


def predict_probability_from_features(pipeline, features: dict[str, float], feature_columns: list[str]) -> float:
    feature_frame = pd.DataFrame([features], columns=feature_columns)
    return float(pipeline.predict_proba(feature_frame)[0, 1])


@lru_cache(maxsize=None)
def _series_probability_from_score(game_win_probability: float, selected_wins: int, opponent_wins: int) -> float:
    if selected_wins >= 4:
        return 1.0
    if opponent_wins >= 4:
        return 0.0
    return (
        game_win_probability * _series_probability_from_score(game_win_probability, selected_wins + 1, opponent_wins)
        + (1.0 - game_win_probability)
        * _series_probability_from_score(game_win_probability, selected_wins, opponent_wins + 1)
    )


def simulate_best_of_seven_series_probability(
    game_win_probability: float,
    selected_wins: int,
    opponent_wins: int,
) -> float:
    """Estimate selected team's series win probability from current score.

    Uses the current single-game win probability as a simple constant baseline for
    each remaining game in the best-of-7 series.
    """
    p = max(0.0, min(1.0, float(game_win_probability)))
    return float(_series_probability_from_score(round(p, 6), int(selected_wins), int(opponent_wins)))


def apply_semantic_series_factor_direction(factors: pd.DataFrame) -> pd.DataFrame:
    """Make semantic series-score direction deterministic if it appears in a table."""
    if factors.empty or "feature" not in factors.columns or "pushes_toward" not in factors.columns:
        return factors
    adjusted = factors.copy()
    mask = adjusted["feature"].eq("series_score_diff")
    if not mask.any():
        return adjusted

    def direction(value) -> str:
        numeric = float(value or 0.0)
        if numeric > 0:
            return "Team A"
        if numeric < 0:
            return "Team B"
        return "Tied"

    adjusted.loc[mask, "pushes_toward"] = adjusted.loc[mask, "value"].apply(direction)
    return adjusted


def _semantic_feature_direction(feature: str, value) -> str | None:
    numeric = float(value or 0.0)
    if feature == "seed_difference":
        if numeric > 0:
            return "Team A"
        if numeric < 0:
            return "Team B"
        return "Tied"
    if feature == "higher_seed_A":
        if numeric >= 1.0:
            return "Team A"
        if numeric <= 0.0:
            return "Team B"
        return "Tied"
    if feature == "series_score_diff":
        if numeric > 0:
            return "Team A"
        if numeric < 0:
            return "Team B"
        return "Tied"
    return None


def apply_semantic_factor_direction(factors: pd.DataFrame) -> pd.DataFrame:
    """Use basketball semantics for deterministic context features shown to users."""
    if factors.empty or "feature" not in factors.columns or "pushes_toward" not in factors.columns:
        return factors
    adjusted = factors.copy()
    for index, row in adjusted.iterrows():
        direction = _semantic_feature_direction(str(row.get("feature")), row.get("value", 0.0))
        if direction is not None:
            adjusted.at[index, "pushes_toward"] = direction
    return adjusted


def _local_factor_table_impl(
    features: dict[str, float],
    importances: pd.DataFrame,
    feature_columns: list[str],
    pipeline=None,
    full_probability: float | None = None,
) -> pd.DataFrame:
    values = pd.DataFrame(
        {
            "feature": feature_columns,
            "value": [features.get(column) for column in feature_columns],
        }
    )
    global_importances = importances[["feature", "importance"]].rename(
        columns={"importance": "global_importance"}
    )
    factors = values.merge(global_importances, on="feature", how="left")
    factors["global_importance"] = pd.to_numeric(factors["global_importance"], errors="coerce")
    factors["importance"] = factors["global_importance"]
    factors["value"] = pd.to_numeric(factors["value"], errors="coerce").fillna(0.0)
    if pipeline is not None and full_probability is not None:
        contributions = _feature_probability_contributions(pipeline, features, feature_columns, full_probability)
        factors["signed_contribution"] = factors["feature"].map(contributions).fillna(0.0)
    else:
        factors["signed_contribution"] = factors["value"] * factors["global_importance"].fillna(0.0)
    factors["local_effect"] = factors["signed_contribution"]
    factors["model_delta_direction"] = factors["signed_contribution"].apply(lambda value: "Team A" if value >= 0 else "Team B")
    factors["pushes_toward"] = factors["model_delta_direction"]
    factors["abs_contribution"] = factors["signed_contribution"].abs()
    factors = apply_semantic_factor_direction(factors)
    return factors.sort_values(
        ["global_importance", "abs_contribution"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def local_factor_table(*args, **kwargs) -> pd.DataFrame:
    with perf_timer("explanation factor generation"):
        return _local_factor_table_impl(*args, **kwargs)


def factor_chart(
    factors: pd.DataFrame,
    team_a_label: str,
    team_b_label: str,
    team_a_color: str = TEAM_A_COLOR,
    team_b_color: str = TEAM_B_COLOR,
):
    factors = filter_noninformative_factors(factors)
    top_factors = factors.head(10).copy()
    top_factors["direction"] = top_factors["pushes_toward"].map(
        {"Team A": team_a_label, "Team B": team_b_label}
    ).fillna(team_a_label)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#FFFFFF")
    sns.barplot(
        data=top_factors,
        x="signed_contribution",
        y="feature",
        hue="direction",
        palette=[team_a_color, team_b_color],
        ax=ax,
    )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Local contribution heuristic")
    ax.set_ylabel("")
    ax.set_title("Top Factors Toward Each Team", color=INK, fontsize=15, weight="bold", pad=14)
    ax.grid(axis="x", color="#E5E7EB", linewidth=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def filter_noninformative_factors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    importance_column = "global_importance" if "global_importance" in df.columns else "importance"
    contribution_column = "local_effect" if "local_effect" in df.columns else "signed_contribution"
    if importance_column not in df.columns or contribution_column not in df.columns:
        return df.copy()

    importance = pd.to_numeric(df[importance_column], errors="coerce")
    contribution = pd.to_numeric(df[contribution_column], errors="coerce")
    noninformative = (importance.isna() | importance.eq(0.0)) & (
        contribution.isna() | contribution.eq(0.0)
    )
    return df.loc[~noninformative].copy()


def display_factor_table(
    factors: pd.DataFrame,
    team_a_abbreviation: str,
    team_b_abbreviation: str,
    hide_empty_rows: bool = True,
) -> pd.DataFrame:
    source = factors.copy()
    if "global_importance" not in source.columns and "importance" in source.columns:
        source["global_importance"] = source["importance"]
    if "local_effect" not in source.columns and "signed_contribution" in source.columns:
        source["local_effect"] = source["signed_contribution"]
    if hide_empty_rows:
        source = filter_noninformative_factors(source)
    columns = ["feature", "value", "global_importance", "local_effect", "pushes_toward"]
    view = source.head(10)[[column for column in columns if column in source.columns]].copy()
    direction_labels = {"Team A": team_a_abbreviation, "Team B": team_b_abbreviation}
    if "pushes_toward" in view.columns:
        view["pushes_toward"] = view["pushes_toward"].replace(direction_labels)
    if "global_importance" in view.columns:
        view["global_importance"] = view["global_importance"].where(
            pd.notna(view["global_importance"]), "unavailable"
        )
    return view.rename(
        columns={
            "feature": "feature",
            "value": "value",
            "global_importance": "global_importance",
            "local_effect": "local_effect",
            "pushes_toward": "favors",
        }
    )


def display_feature_frame(
    feature_frame: pd.DataFrame,
    features: dict[str, float],
    team_a_abbreviation: str,
    team_b_abbreviation: str,
) -> pd.DataFrame:
    view = feature_frame.copy()
    view = view.drop(columns=[column for column in SEMANTIC_SERIES_FEATURES if column in view.columns])
    if "home_team_A" in view.columns:
        view = view.rename(columns={"home_team_A": "home_team"})
        view["home_team"] = team_a_abbreviation if float(features.get("home_team_A", 0) or 0) == 1.0 else team_b_abbreviation
    return view


def display_home_debug_table(
    rows: list[dict],
    team_a_abbreviation: str,
    team_b_abbreviation: str,
) -> pd.DataFrame:
    view = pd.DataFrame(rows).copy()
    if view.empty:
        return view
    if "team_A_probability" in view.columns:
        view = view.rename(columns={"team_A_probability": f"{team_a_abbreviation}_probability"})
    if "home_team_A" in view.columns:
        view = view.rename(columns={"home_team_A": "home_team"})
        view["home_team"] = view["home_team"].apply(
            lambda value: team_a_abbreviation
            if pd.notna(value) and float(value) == 1.0
            else (team_b_abbreviation if pd.notna(value) else None)
        )
    return view


def filter_explanation_features_for_mode(
    factors: pd.DataFrame,
    importances: pd.DataFrame,
    prediction_context_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded_features = set(SEMANTIC_SERIES_FEATURES)
    excluded_features.update(SERIES_CONTEXT_FEATURES)
    return (
        factors[~factors["feature"].isin(excluded_features)].reset_index(drop=True),
        importances[~importances["feature"].isin(excluded_features)].reset_index(drop=True),
    )


def inject_dashboard_css() -> None:
    st.markdown(
        f"""
        <style>
            header,
            [data-testid="stHeader"] {{
                visibility: hidden !important;
                height: 0 !important;
                min-height: 0 !important;
                background: transparent !important;
            }}
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"] {{
                display: none;
            }}
            [data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) {{
                display: flex !important;
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 0 !important;
                height: 0 !important;
                background: transparent !important;
                overflow: visible !important;
                pointer-events: none;
            }}
            [data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) > * {{
                display: flex !important;
            }}
            [data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) [data-testid="stToolbarActions"],
            [data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) [data-testid="stAppDeployButton"],
            [data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainMenu"] {{
                display: none !important;
            }}
            [data-testid="stExpandSidebarButton"],
            [data-testid="stExpandSidebarButton"] * {{
                display: inline-flex !important;
                visibility: visible !important;
                opacity: 1 !important;
                pointer-events: auto !important;
            }}
            [data-testid="stExpandSidebarButton"] {{
                position: fixed !important;
                top: 0.75rem !important;
                left: 0.75rem !important;
                width: 2rem !important;
                height: 2rem !important;
                min-width: 2rem !important;
                min-height: 2rem !important;
                align-items: center !important;
                justify-content: center !important;
                z-index: 1000000 !important;
                background: #FFFFFF !important;
                color: var(--ink) !important;
                border: 1px solid var(--line) !important;
                border-radius: 6px !important;
                box-shadow: 0 4px 14px rgba(23, 32, 51, 0.12) !important;
            }}
            [data-testid="stExpandSidebarButton"] svg,
            [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
                color: var(--ink) !important;
                fill: var(--ink) !important;
            }}
            [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
                width: 24px !important;
                height: 24px !important;
            }}
            :root {{
                --ink: {INK};
                --muted: {MUTED};
                --line: {LINE};
                --surface: {SURFACE};
                --surface-soft: #F7F8FA;
                --team-a: {TEAM_A_COLOR};
                --team-b: {TEAM_B_COLOR};
            }}
            .stApp {{
                background:
                    linear-gradient(180deg, #F5F7FA 0%, #EEF2F6 100%);
                color: var(--ink);
            }}
            [data-testid="stSidebar"] {{
                background: #FFFFFF;
                border-right: 1px solid var(--line);
            }}
            [data-testid="stSidebarContent"] {{
                padding-top: 0.35rem !important;
            }}
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] label {{
                color: var(--ink);
            }}
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div[role="radiogroup"] label,
            [data-testid="stSidebar"] div[role="radiogroup"] label * {{
                color: var(--ink) !important;
            }}
            [data-testid="stSidebar"] div[role="radio"] {{
                color: var(--ink) !important;
                background: transparent !important;
            }}
            .block-container {{
                max-width: 1240px;
                padding-top: 1.45rem;
                padding-bottom: 1rem;
            }}
            .sidebar-compact-separator {{
                height: 1px;
                background: #E6EAF0;
                margin: 0.25rem 0 0.35rem;
            }}
            .sidebar-inline-label {{
                color: var(--ink);
                font-size: 0.9rem;
                font-weight: 800;
                line-height: 2.35rem;
                white-space: nowrap;
            }}
            .sidebar-inline-label.right {{
                text-align: right;
            }}
            .sidebar-inline-label.center {{
                text-align: center;
            }}
            .sidebar-context-note {{
                color: var(--muted);
                font-size: 0.83rem;
                margin: 0.35rem 0 0.5rem;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] {{
                margin-bottom: 0.15rem;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
                min-height: 2.15rem;
                background: #FFFFFF !important;
                color: var(--ink) !important;
                border-color: #D9DEE7 !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] span,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] div {{
                color: var(--ink) !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] [aria-disabled="true"],
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] [aria-disabled="true"] * {{
                background: #F7F8FA !important;
                color: var(--ink) !important;
                opacity: 1 !important;
                -webkit-text-fill-color: var(--ink) !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] svg {{
                color: var(--ink) !important;
                fill: var(--ink) !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] {{
                width: 58px !important;
                min-width: 58px !important;
                max-width: 58px !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] {{
                width: 56px !important;
                min-width: 56px !important;
                max-width: 56px !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] > div,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] > div {{
                min-height: 2rem !important;
                height: 2rem !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
                border-radius: 7px !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] > div > div:first-child,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] > div > div:first-child {{
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
                padding-left: 0.2rem !important;
                padding-right: 0 !important;
                min-width: 0 !important;
                flex: 1 1 auto !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] > div > div:first-child > div,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] > div > div:first-child > div {{
                width: 100% !important;
                min-width: 0 !important;
                text-align: center !important;
                justify-content: center !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] > div > div:first-child > div:first-child,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] > div > div:first-child > div:first-child {{
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
                height: 100% !important;
                min-width: 0 !important;
                font-size: 0.86rem !important;
                line-height: 1 !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] input,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] input {{
                caret-color: transparent !important;
                color: transparent !important;
                width: 0 !important;
                min-width: 0 !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] > div > div:last-child,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] > div > div:last-child {{
                width: 20px !important;
                min-width: 20px !important;
                flex: 0 0 20px !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Game number"]) [data-baseweb="select"] svg,
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="series wins"]) [data-baseweb="select"] svg {{
                width: 18px !important;
                height: 18px !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Series Score"]) [data-baseweb="select"] > div {{
                min-height: 2.35rem !important;
                height: 2.35rem !important;
                border-radius: 8px !important;
                background: #FFFFFF !important;
                border: 1px solid #D9DEE7 !important;
                color: var(--ink) !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Series Score"]) [data-baseweb="select"] > div > div:first-child {{
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
                font-weight: 850 !important;
                letter-spacing: 0 !important;
                white-space: nowrap !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stSelectbox"]:has([aria-label*="Series Score"]) [data-baseweb="select"] input {{
                caret-color: transparent !important;
                color: transparent !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stNumberInput"]:has(input[aria-label*="series wins"]) {{
                margin-bottom: 0 !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stNumberInput"]:has(input[aria-label*="series wins"]) > div {{
                width: 52px !important;
                min-width: 52px !important;
                max-width: 52px !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stNumberInput"]:has(input[aria-label*="series wins"]) input {{
                height: 2rem !important;
                min-height: 2rem !important;
                padding: 0 !important;
                text-align: center !important;
                background: #FFFFFF !important;
                color: var(--ink) !important;
                -webkit-text-fill-color: var(--ink) !important;
                border: 1px solid #D9DEE7 !important;
                border-radius: 7px !important;
                font-size: 0.88rem !important;
                font-weight: 750 !important;
                opacity: 1 !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stNumberInput"]:has(input[aria-label*="series wins"]) input:disabled {{
                background: #F7F8FA !important;
                color: var(--ink) !important;
                -webkit-text-fill-color: var(--ink) !important;
                opacity: 1 !important;
            }}
            [data-testid="stSidebar"] div[data-testid="stNumberInput"]:has(input[aria-label*="series wins"]) button {{
                display: none !important;
            }}
            .dashboard-title {{
                font-size: 2.15rem;
                line-height: 1.15;
                font-weight: 800;
                color: var(--ink);
                margin: 0;
            }}
            .dashboard-subtitle {{
                color: var(--muted);
                font-size: 1rem;
                margin-top: 0.45rem;
                margin-bottom: 1.1rem;
            }}
            .dashboard-header {{
                display: grid;
                grid-template-columns: minmax(0, 1fr) minmax(240px, 320px);
                gap: 1rem;
                align-items: center;
                margin-bottom: 1.1rem;
            }}
            .dashboard-header .dashboard-subtitle {{
                margin-bottom: 0;
            }}
            .model-status-card {{
                position: relative;
                overflow: hidden;
                background: linear-gradient(180deg, rgba(249, 115, 22, 0.035), rgba(100, 116, 139, 0.025) 48%, var(--surface) 100%);
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 0.9rem 0.95rem 0.82rem;
                box-shadow: 0 12px 30px rgba(23, 32, 51, 0.07);
                align-self: center;
                min-width: 0;
            }}
            .model-status-card::before {{
                content: "";
                position: absolute;
                inset: 0 0 auto;
                height: 3px;
                background: linear-gradient(90deg, #F97316 0%, #94A3B8 72%, #64748B 100%);
            }}
            .model-status-top {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.85rem;
                margin-bottom: 0.6rem;
            }}
            .model-status-title {{
                display: inline-flex;
                align-items: center;
                gap: 0.42rem;
                color: var(--ink);
                font-size: 0.8rem;
                font-weight: 900;
                line-height: 1;
                text-transform: uppercase;
                letter-spacing: 0.02em;
            }}
            .model-status-title::before {{
                content: "";
                width: 0.48rem;
                height: 0.48rem;
                border-radius: 999px;
                background: #F97316;
                box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.14);
                flex: 0 0 auto;
            }}
            .model-status-badges {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.46rem;
                margin-bottom: 0.55rem;
            }}
            .model-status-pill {{
                display: inline-flex;
                align-items: center;
                border: 1px solid rgba(23, 32, 51, 0.1);
                border-radius: 999px;
                color: var(--ink);
                font-size: 0.72rem;
                font-weight: 800;
                line-height: 1;
                padding: 0.32rem 0.58rem;
                white-space: nowrap;
            }}
            .model-status-pill[data-tone="warm"] {{
                background: rgba(249, 115, 22, 0.14);
                border-color: rgba(249, 115, 22, 0.3);
                color: #9A4A0F;
            }}
            .model-status-pill[data-tone="cool"] {{
                background: rgba(59, 130, 246, 0.11);
                border-color: rgba(59, 130, 246, 0.24);
                color: #1E4F8A;
            }}
            .model-status-limitations {{
                color: #667085;
                font-size: 0.69rem;
                line-height: 1.3;
            }}
            .model-details-button {{
                appearance: none;
                display: inline-flex;
                align-items: center;
                flex: 0 0 auto;
                border: 1px solid rgba(100, 116, 139, 0.22);
                border-radius: 999px;
                padding: 0.25rem 0.58rem;
                background: rgba(255, 255, 255, 0.72);
                color: #475467;
                cursor: pointer;
                font-family: inherit;
                font-size: 0.68rem;
                font-weight: 800;
                line-height: 1;
                transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, box-shadow 160ms ease;
            }}
            .model-details-button:hover {{
                background: #FFFFFF;
                border-color: rgba(249, 115, 22, 0.34);
                color: var(--ink);
                box-shadow: 0 4px 12px rgba(23, 32, 51, 0.08);
            }}
            .model-details-popover {{
                width: min(840px, calc(100vw - 2rem));
                max-height: min(82vh, 760px);
                overflow: auto;
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 16px;
                padding: 1.05rem;
                background:
                    radial-gradient(circle at top left, rgba(249, 115, 22, 0.18), transparent 30%),
                    linear-gradient(145deg, #0B1220 0%, #111827 56%, #172033 100%);
                color: #F8FAFC;
                box-shadow: 0 26px 64px rgba(2, 6, 23, 0.42);
                font-size: 0.84rem;
                line-height: 1.45;
            }}
            .model-details-title {{
                margin: 0 0 0.9rem;
                color: #F8FAFC;
                font-size: 1.15rem;
                line-height: 1.2;
                font-weight: 900;
            }}
            .model-details-grid {{
                display: grid;
                grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
                gap: 0.8rem;
                align-items: start;
            }}
            .model-details-stack {{
                display: grid;
                gap: 0.8rem;
            }}
            .model-details-section {{
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 12px;
                padding: 0.82rem;
                background: rgba(15, 23, 42, 0.76);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }}
            .model-details-section-primary {{
                padding: 0;
                background: transparent;
                border: 0;
                box-shadow: none;
            }}
            .model-details-section-title {{
                color: #CBD5E1;
                font-size: 0.7rem;
                font-weight: 900;
                letter-spacing: 0.04em;
                line-height: 1;
                margin-bottom: 0.62rem;
                text-transform: uppercase;
            }}
            .model-facts-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.58rem;
            }}
            .model-fact-row {{
                display: grid;
                gap: 0.22rem;
                min-width: 0;
                padding: 0.58rem;
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 10px;
                background: rgba(30, 41, 59, 0.48);
            }}
            .model-detail-label {{
                color: #94A3B8;
                font-weight: 800;
            }}
            .model-detail-value {{
                color: #F8FAFC;
                font-weight: 800;
                overflow-wrap: normal;
                white-space: nowrap;
            }}
            .model-fact-row .model-detail-value {{
                white-space: normal;
            }}
            .model-metrics-dashboard {{
                display: grid;
                grid-template-columns: minmax(13rem, 0.92fr) minmax(0, 1.08fr);
                gap: 0.8rem;
                align-items: stretch;
            }}
            .model-hero-metric {{
                display: grid;
                align-content: center;
                min-height: 9rem;
                padding: 1rem;
                border: 1px solid rgba(249, 115, 22, 0.32);
                border-radius: 14px;
                background:
                    linear-gradient(145deg, rgba(249, 115, 22, 0.24), rgba(59, 130, 246, 0.08)),
                    rgba(15, 23, 42, 0.86);
            }}
            .model-hero-label {{
                color: #FDBA74;
                font-size: 0.72rem;
                font-weight: 900;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }}
            .model-hero-value {{
                color: #FFFFFF;
                font-size: 3rem;
                font-weight: 950;
                line-height: 1;
                margin: 0.3rem 0 0.22rem;
                white-space: nowrap;
                font-variant-numeric: tabular-nums;
            }}
            .model-hero-subtitle {{
                color: #CBD5E1;
                font-size: 0.78rem;
                font-weight: 750;
            }}
            .model-metrics-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.55rem;
            }}
            .model-metric-card {{
                display: grid;
                gap: 0.2rem;
                min-width: 0;
                padding: 0.68rem;
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 10px;
                background: rgba(30, 41, 59, 0.56);
            }}
            .model-feature-list {{
                display: grid;
                gap: 0.52rem;
            }}
            .model-feature-row {{
                display: grid;
                grid-template-columns: minmax(7.5rem, 1fr) 4.2rem;
                gap: 0.7rem;
                align-items: center;
            }}
            .model-feature-label {{
                color: #F8FAFC;
                font-weight: 800;
                overflow-wrap: anywhere;
            }}
            .model-feature-value {{
                color: #CBD5E1;
                font-variant-numeric: tabular-nums;
                text-align: right;
                white-space: nowrap;
            }}
            .model-feature-track {{
                grid-column: 1 / -1;
                height: 0.38rem;
                border-radius: 999px;
                background: rgba(148, 163, 184, 0.18);
                overflow: hidden;
            }}
            .model-feature-bar {{
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, #F97316, #3B82F6);
            }}
            .model-calibration-table {{
                display: grid;
                gap: 0.34rem;
            }}
            .model-calibration-row {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.5rem;
                align-items: center;
                color: #E2E8F0;
                font-size: 0.78rem;
            }}
            .model-calibration-row[data-head="true"] {{
                color: #94A3B8;
                font-size: 0.68rem;
                font-weight: 900;
                text-transform: uppercase;
            }}
            .model-calibration-row span {{
                white-space: nowrap;
                font-variant-numeric: tabular-nums;
            }}
            .model-detail-note {{
                color: #94A3B8;
                font-size: 0.78rem;
                line-height: 1.4;
                margin: 0;
            }}
            .model-calibration-copy {{
                color: #CBD5E1;
                font-size: 0.78rem;
                line-height: 1.45;
                margin: 0 0 0.42rem;
            }}
            @media (max-width: 720px) {{
                .model-details-grid,
                .model-metrics-dashboard,
                .model-facts-grid,
                .model-metrics-grid {{
                    grid-template-columns: 1fr;
                }}
                .model-feature-row,
                .model-calibration-row {{
                    grid-template-columns: minmax(0, 1fr) auto auto;
                }}
                .model-hero-metric {{
                    min-height: 7rem;
                }}
                .model-hero-value {{
                    font-size: 2.45rem;
                }}
            }}
            @media (max-width: 720px) {{
                .dashboard-header {{
                    grid-template-columns: 1fr;
                }}
                .model-status-card {{
                    width: 100%;
                }}
            }}
            .live-games-board {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin: 0.25rem 0 1rem;
            }}
            .live-games-section {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem;
                box-shadow: 0 8px 24px rgba(23, 32, 51, 0.045);
            }}
            .live-games-heading {{
                color: var(--ink);
                font-size: 0.9rem;
                font-weight: 900;
                margin-bottom: 0.55rem;
            }}
            .live-games-row {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.55rem;
            }}
            .live-game-card {{
                border: 1px solid #E5E9F0;
                border-radius: 8px;
                padding: 0.62rem 0.55rem;
                text-align: center;
                background: #FBFCFE;
                min-height: 116px;
            }}
            .live-game-teams {{
                display: flex;
                justify-content: center;
                align-items: baseline;
                gap: 0.35rem;
                font-weight: 950;
                font-size: 1.05rem;
                letter-spacing: 0;
                white-space: nowrap;
            }}
            .live-game-vs {{
                color: var(--muted);
                font-size: 0.75rem;
                font-weight: 850;
            }}
            .live-game-date {{
                color: var(--muted);
                font-size: 0.76rem;
                margin-top: 0.3rem;
                min-height: 1.1rem;
            }}
            .live-game-score,
            .live-game-prediction {{
                color: var(--ink);
                font-size: 0.86rem;
                margin-top: 0.4rem;
                line-height: 1.25;
            }}
            .live-game-note {{
                color: var(--muted);
                font-size: 0.78rem;
                margin-top: 0.42rem;
            }}
            .live-game-card strong {{
                color: var(--ink);
                font-weight: 950;
            }}
            .live-game-load-note {{
                color: var(--muted);
                font-size: 0.72rem;
                text-align: center;
                margin: 0.25rem 0 0.15rem;
            }}
            div[data-testid="stButton"] > button[kind="secondary"] {{
                background: #FFFFFF !important;
                color: var(--ink) !important;
                border: 1px solid #D9DEE7 !important;
                border-radius: 7px !important;
                box-shadow: none !important;
                min-height: 2.1rem !important;
                padding: 0.35rem 0.55rem !important;
                font-size: 0.78rem !important;
                line-height: 1.1 !important;
                font-weight: 800 !important;
            }}
            div[data-testid="stButton"] > button[kind="secondary"]:hover {{
                border-color: color-mix(in srgb, var(--team-a) 45%, #D9DEE7) !important;
                background: #F8FAFC !important;
                color: var(--ink) !important;
            }}
            div[data-testid="stButton"] > button[kind="secondary"] p {{
                color: var(--ink) !important;
                font-size: 0.78rem !important;
                line-height: 1.1 !important;
                white-space: nowrap !important;
            }}
            @media (max-width: 980px) {{
                .live-games-board {{
                    grid-template-columns: 1fr;
                }}
            }}
            @media (max-width: 720px) {{
                .live-games-row {{
                    grid-template-columns: 1fr;
                }}
            }}
            .panel {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1.05rem 1.15rem;
                box-shadow: 0 8px 24px rgba(23, 32, 51, 0.05);
                margin-bottom: 1rem;
            }}
            .panel-title {{
                color: var(--ink);
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
            }}
            .panel-note {{
                color: var(--muted);
                font-size: 0.88rem;
                margin-bottom: 0.65rem;
            }}
            .matchup-strip {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.35rem;
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1.35rem 1rem;
                margin-bottom: 1rem;
                text-align: center;
            }}
            .matchup-title {{
                color: var(--ink);
                font-size: 3rem;
                line-height: 1;
                font-weight: 900;
                letter-spacing: 0;
            }}
            .team-gradient-text {{
                color: var(--fallback-color);
                background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
                -webkit-background-clip: text;
                background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 0 rgba(0, 0, 0, 0);
            }}
            .matchup-subtitle {{
                color: var(--muted);
                font-size: 1rem;
                font-weight: 650;
            }}
            .summary-card {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem 1.15rem;
                margin-bottom: 1rem;
            }}
            .summary-main {{
                color: var(--ink);
                font-size: 1.35rem;
                font-weight: 850;
                margin-bottom: 0.25rem;
            }}
            .confidence-pill {{
                display: inline-block;
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 0.2rem 0.65rem;
                color: var(--ink);
                background: var(--surface-soft);
                font-size: 0.86rem;
                font-weight: 800;
            }}
            .prob-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-bottom: 1rem;
            }}
            .prob-card {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem;
            }}
            .prob-label {{
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .prob-value {{
                font-size: 2rem;
                line-height: 1.1;
                font-weight: 900;
                margin-top: 0.2rem;
            }}
            .prob-home {{
                color: var(--muted);
                font-size: 0.88rem;
                margin-top: 0.3rem;
            }}
            .prob-meter {{
                width: 100%;
                height: 24px;
                display: flex;
                overflow: hidden;
                border-radius: 6px;
                border: 1px solid var(--line);
                background: #FFFFFF;
                margin-top: 0.7rem;
            }}
            .prob-meter-a {{
                background: var(--team-a);
            }}
            .prob-meter-b {{
                background: var(--team-b);
            }}
            .meter-labels {{
                display: flex;
                justify-content: space-between;
                color: var(--muted);
                font-size: 0.84rem;
                margin-top: 0.45rem;
            }}
            .saved-note {{
                color: var(--muted);
                font-size: 0.84rem;
                padding-top: 0.35rem;
            }}
            div[data-testid="stMetric"] {{
                background: #FFFFFF;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem 1rem;
            }}
            div[data-testid="stMetricLabel"] p {{
                color: var(--muted);
                font-weight: 700;
            }}
            div[data-testid="stMetricValue"] {{
                color: var(--ink);
            }}
            .stButton > button {{
                width: 100%;
                border-radius: 6px;
                min-height: 2.8rem;
                font-weight: 800;
            }}
            h2, h3 {{
                color: var(--ink);
            }}
            div[data-testid="stExpander"] {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
            }}
            div[data-testid="stDataFrame"],
            div[data-testid="stDataFrame"] p,
            div[data-testid="stDataFrame"] span,
            div[data-testid="stTable"],
            div[data-testid="stTable"] p,
            div[data-testid="stTable"] span {{
                color: var(--ink);
            }}
            div[data-testid="stExpander"] summary,
            div[data-testid="stExpander"] summary p,
            div[data-testid="stExpander"] summary span {{
                color: var(--ink) !important;
                font-weight: 800;
            }}
            div[data-testid="stExpander"] svg,
            [data-testid="stSidebar"] svg,
            button svg {{
                visibility: visible !important;
                display: inline-block !important;
            }}
            [data-testid="stSidebarCollapseButton"],
            [data-testid="stSidebarCollapseButton"] *,
            [data-testid="collapsedControl"],
            [data-testid="collapsedControl"] *,
            [data-testid="stExpandSidebarButton"],
            [data-testid="stExpandSidebarButton"] *,
            [data-testid="stExpander"] summary svg,
            [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{
                visibility: visible !important;
                opacity: 1 !important;
            }}
            .chat-panel {{
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem;
                box-shadow: 0 8px 24px rgba(23, 32, 51, 0.05);
                height: 560px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}
            .chat-panel-title {{
                color: var(--ink);
                font-size: 1.05rem;
                font-weight: 850;
                margin-bottom: 0.25rem;
            }}
            .chat-panel-note {{
                color: var(--muted);
                font-size: 0.86rem;
                margin-bottom: 0.55rem;
            }}
            .chat-history {{
                height: 470px;
                overflow-y: auto;
                padding: 0.35rem 0.25rem 0.35rem 0;
                border-top: 1px solid #EEF1F5;
            }}
            .chat-stream {{
                display: flex;
                flex-direction: column;
                gap: 0.55rem;
            }}
            .chat-row {{
                display: flex;
                width: 100%;
            }}
            .chat-row.assistant {{
                justify-content: flex-start;
            }}
            .chat-row.user {{
                justify-content: flex-end;
            }}
            .chat-bubble {{
                max-width: 94%;
                border-radius: 8px;
                padding: 0.58rem 0.72rem;
                line-height: 1.35;
                font-size: 0.88rem;
                border: 1px solid var(--line);
                box-shadow: 0 4px 14px rgba(23, 32, 51, 0.05);
                overflow-wrap: anywhere;
            }}
            .chat-bubble.assistant {{
                background: #FFFFFF;
                color: var(--ink);
            }}
            .chat-bubble.user {{
                background: var(--team-a);
                color: var(--team-a-text);
                border-color: var(--team-a);
            }}
            .chat-bubble p,
            .chat-bubble li,
            .chat-bubble strong {{
                color: inherit !important;
            }}
            .chat-bubble p {{
                margin: 0.12rem 0;
            }}
            .chat-bubble p:has(strong) {{
                margin-top: 0.48rem;
                margin-bottom: 0.12rem;
            }}
            .chat-bubble p:first-child {{
                margin-top: 0;
            }}
            .chat-bubble p + ul,
            .chat-bubble p + ol {{
                margin-top: 0.12rem;
            }}
            .chat-bubble ul,
            .chat-bubble ol {{
                margin-top: 0.12rem;
                margin-bottom: 0.35rem;
                padding-left: 1rem;
            }}
            .chat-bubble li {{
                margin-bottom: 0.16rem;
            }}
            .chat-input-card {{
                background: var(--surface);
                border: 0;
                border-top: 1px solid #EEF1F5;
                border-radius: 0;
                padding: 0.55rem 0 0;
                margin-top: 0;
                box-shadow: none;
            }}
            .chat-input-card div[data-testid="stForm"] {{
                border: 0;
                padding: 0;
            }}
            .chat-input-card div[data-testid="stHorizontalBlock"] {{
                align-items: center;
                gap: 0.45rem;
            }}
            .chat-input-card div[data-testid="stTextInput"] {{
                margin-bottom: 0;
            }}
            .chat-input-card input,
            .chat-input-card input::placeholder,
            div[data-testid="stTextInput"] input,
            div[data-testid="stTextInput"] input::placeholder {{
                color: var(--ink) !important;
                caret-color: var(--ink) !important;
                opacity: 1 !important;
            }}
            .chat-input-card input,
            div[data-testid="stTextInput"] input {{
                background: #FFFFFF !important;
                border: 1px solid var(--line) !important;
                height: 2.35rem !important;
                min-height: 2.35rem !important;
                font-size: 0.9rem !important;
            }}
            div[data-testid="stForm"] button {{
                min-height: 2.35rem !important;
                height: 2.35rem !important;
                min-width: 4.8rem !important;
                margin-top: 0 !important;
                border-radius: 6px !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                white-space: nowrap !important;
                color: var(--team-a-text) !important;
                background: var(--team-a) !important;
                border: 1px solid var(--team-a) !important;
                transform: translateY(-0.05rem);
                font-size: 0.9rem !important;
                font-weight: 800 !important;
            }}
            @media (max-width: 720px) {{
                .prob-grid {{
                    grid-template-columns: 1fr;
                }}
                .matchup-title {{
                    font-size: 2.25rem;
                }}
                .chat-panel {{
                    height: 520px;
                }}
                .chat-history {{
                    height: 430px;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_team_css(team_a_abbreviation: str, team_b_abbreviation: str) -> tuple[dict[str, str], dict[str, str]]:
    team_a_colors = get_team_colors(team_a_abbreviation)
    team_b_colors = get_team_colors(team_b_abbreviation)
    st.markdown(
        f"""
        <style>
            :root {{
                --team-a: {team_a_colors["primary"]};
                --team-a-secondary: {team_a_colors["secondary"]};
                --team-a-text: {team_a_colors["text"]};
                --team-a-accent-text: {team_a_colors["accent_text"]};
                --team-b: {team_b_colors["primary"]};
                --team-b-secondary: {team_b_colors["secondary"]};
                --team-b-text: {team_b_colors["text"]};
                --team-b-accent-text: {team_b_colors["accent_text"]};
            }}
            .matchup-strip {{
                border-color: color-mix(in srgb, var(--team-a) 35%, var(--line));
                box-shadow: inset 4px 0 0 var(--team-a), inset -4px 0 0 var(--team-b), 0 8px 24px rgba(23, 32, 51, 0.05);
            }}
            .summary-card,
            .chat-panel {{
                border-color: color-mix(in srgb, var(--team-a) 28%, var(--line));
            }}
            div[data-testid="stForm"] button {{
                background: var(--team-a) !important;
                border-color: var(--team-a) !important;
                color: var(--team-a-text) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return team_a_colors, team_b_colors


MODEL_DETAIL_FACTS = [
    ("Model", "Calibrated RF"),
    ("Method", "Order-symmetric probabilities"),
    ("Purpose", "Pure basketball-stat matchup/series forecast"),
    ("Excludes", "odds, injuries, lineup/news"),
]

MODEL_DETAIL_METRICS = [
    ("Accuracy", "accuracy"),
    ("ROC AUC", "roc_auc"),
    ("Brier Score", "brier_score"),
    ("Log Loss", "log_loss"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("ECE", "expected_calibration_error"),
]


@st.cache_data(show_spinner=False)
def _cached_processed_csv(path: str, path_version: tuple[int, int]) -> pd.DataFrame:
    del path_version
    with perf_timer(f"processed data load {Path(path).name}"):
        return pd.read_csv(path)


def _read_processed_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return _cached_processed_csv(str(path), _path_version(path)).copy()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _format_model_detail_number(value) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _select_model_detail_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    selected = frame.copy()
    if "model" in selected.columns:
        model_rows = selected[selected["model"].astype(str).str.contains("Random Forest", case=False, na=False)]
        if not model_rows.empty:
            selected = model_rows
    if "prediction_context_mode" in selected.columns:
        mode_rows = selected[selected["prediction_context_mode"].eq(PREDICTION_MODE_CURRENT)]
        if not mode_rows.empty:
            selected = mode_rows
    if "calibration_method" in selected.columns:
        calibrated_rows = selected[~selected["calibration_method"].astype(str).str.lower().eq("raw")]
        if not calibrated_rows.empty:
            selected = calibrated_rows
    if "feature_set" in selected.columns:
        feature_rows = selected[selected["feature_set"].astype(str).str.contains("corrected_signs", na=False)]
        if not feature_rows.empty:
            selected = feature_rows
    return selected


def _best_model_detail_row(frame: pd.DataFrame) -> pd.Series | None:
    selected = _select_model_detail_rows(frame)
    if selected.empty:
        return None

    sort_columns = [column for column in ["roc_auc", "brier_score", "log_loss"] if column in selected.columns]
    if sort_columns:
        selected = selected.assign(
            roc_auc=pd.to_numeric(selected.get("roc_auc"), errors="coerce"),
            brier_score=pd.to_numeric(selected.get("brier_score"), errors="coerce"),
            log_loss=pd.to_numeric(selected.get("log_loss"), errors="coerce"),
        )
        selected = selected.sort_values(
            sort_columns,
            ascending=[False if column == "roc_auc" else True for column in sort_columns],
            na_position="last",
        )
    return selected.iloc[0]


def _render_model_facts_html() -> str:
    rows = []
    for label, value in MODEL_DETAIL_FACTS:
        rows.append(
            '<div class="model-fact-row">'
            f'<span class="model-detail-label">{html.escape(label)}</span>'
            f'<span class="model-detail-value">{html.escape(value)}</span>'
            "</div>"
        )
    return "".join(rows)


def _render_model_metrics_html() -> str:
    calibration = _read_processed_csv(MODEL_CALIBRATION_PATH)
    row = _best_model_detail_row(calibration)
    if row is None:
        comparison = _read_processed_csv(MODEL_COMPARISON_PATH)
        row = _best_model_detail_row(comparison)
    if row is None:
        return '<p class="model-detail-note">Validation metrics are unavailable right now.</p>'

    hero_value = _format_model_detail_number(row.get("roc_auc"))
    if not hero_value:
        hero_label, hero_column = MODEL_DETAIL_METRICS[0]
        hero_value = _format_model_detail_number(row.get(hero_column)) or "Unavailable"
    else:
        hero_label = "ROC AUC"

    supporting_rows = []
    for label, column in MODEL_DETAIL_METRICS:
        if label == hero_label:
            continue
        value = _format_model_detail_number(row.get(column))
        if not value:
            value = "Unavailable"
        supporting_rows.append(
            '<div class="model-metric-card">'
            f'<span class="model-detail-label">{html.escape(label)}</span>'
            f'<span class="model-detail-value">{html.escape(value)}</span>'
            "</div>"
        )
    return (
        '<div class="model-metrics-dashboard">'
        '<div class="model-hero-metric">'
        f'<span class="model-hero-label">{html.escape(hero_label)}</span>'
        f'<span class="model-hero-value">{html.escape(hero_value)}</span>'
        '<span class="model-hero-subtitle">Validation performance</span>'
        "</div>"
        f'<div class="model-metrics-grid">{"".join(supporting_rows)}</div>'
        "</div>"
    )


def _load_model_detail_importances() -> pd.DataFrame:
    def artifact_importances() -> pd.DataFrame:
        try:
            model_bundle = cached_model(str(DEFAULT_MODEL_PATH), model_artifact_version(DEFAULT_MODEL_PATH))
            feature_columns = all_saved_feature_columns(model_bundle)
            return load_feature_importances(model_bundle, feature_columns, PREDICTION_MODE_CURRENT)
        except Exception:
            return pd.DataFrame()

    importances = pd.DataFrame()
    if FEATURE_IMPORTANCE_PATH.exists():
        importances = _read_processed_csv(FEATURE_IMPORTANCE_PATH)
        if not {"feature", "importance"}.issubset(importances.columns):
            importances = pd.DataFrame()
    if importances.empty:
        importances = artifact_importances()
    if importances.empty:
        return pd.DataFrame()

    if "prediction_context_mode" in importances.columns:
        mode_rows = importances[importances["prediction_context_mode"].eq(PREDICTION_MODE_CURRENT)]
        if not mode_rows.empty:
            importances = mode_rows
    if "model" in importances.columns:
        model_rows = importances[importances["model"].astype(str).str.contains("Random Forest", case=False, na=False)]
        if not model_rows.empty:
            importances = model_rows
    importances = importances.copy()
    importances["importance"] = pd.to_numeric(importances["importance"], errors="coerce")
    importances = importances.dropna(subset=["importance"])
    if importances.empty and FEATURE_IMPORTANCE_PATH.exists():
        importances = artifact_importances()
        if importances.empty:
            return pd.DataFrame()
        importances = importances.copy()
        importances["importance"] = pd.to_numeric(importances["importance"], errors="coerce")
        importances = importances.dropna(subset=["importance"])
    return importances.sort_values("importance", ascending=False).head(6).reset_index(drop=True)


def _render_feature_importance_html() -> str:
    importances = _load_model_detail_importances()
    if importances.empty:
        return '<p class="model-detail-note">Feature importance data is unavailable for this model.</p>'

    max_importance = max(float(importances["importance"].max()), 0.000001)
    rows = []
    for _, row in importances.iterrows():
        feature = html.escape(basketball_feature_label(str(row["feature"])).capitalize())
        importance = float(row["importance"])
        width = max(0.0, min(100.0, (importance / max_importance) * 100))
        rows.append(
            '<div class="model-feature-row">'
            f'<span class="model-feature-label">{feature}</span>'
            f'<span class="model-feature-value">{importance:.3f}</span>'
            '<span class="model-feature-track">'
            f'<span class="model-feature-bar" style="width: {width:.1f}%"></span>'
            "</span>"
            "</div>"
        )
    return f'<div class="model-feature-list">{"".join(rows)}</div>'


def _render_calibration_html() -> str:
    calibration = _read_processed_csv(MODEL_CALIBRATION_PATH)
    selected = _select_model_detail_rows(calibration)
    lower_column = "bin_lower" if "bin_lower" in selected.columns else "bin_low"
    required_columns = {"bin_count", lower_column, "bin_upper", "mean_predicted_probability", "observed_win_rate"}
    if selected.empty or not required_columns.issubset(selected.columns):
        return '<p class="model-detail-note">Calibration bins are unavailable right now.</p>'

    selected = selected.copy()
    selected["bin_count"] = pd.to_numeric(selected["bin_count"], errors="coerce").fillna(0)
    selected[lower_column] = pd.to_numeric(selected[lower_column], errors="coerce")
    selected = selected[selected["bin_count"] > 0].sort_values(lower_column).head(4)
    if selected.empty:
        return '<p class="model-detail-note">Calibration bins are unavailable right now.</p>'

    rows = [
        '<p class="model-calibration-copy">'
        "Predicted probabilities are compared with observed win rates across bins."
        "</p>",
        (
            '<div class="model-calibration-row" data-head="true">'
            "<span>Predicted</span><span>Observed</span><span>Count</span></div>"
        ),
    ]
    for _, row in selected.iterrows():
        predicted = _format_model_detail_number(row.get("mean_predicted_probability"))
        observed = _format_model_detail_number(row.get("observed_win_rate"))
        rows.append(
            '<div class="model-calibration-row">'
            f'<span>{html.escape(predicted)}</span>'
            f'<span>{html.escape(observed)}</span>'
            f'<span>{int(row["bin_count"])}</span>'
            "</div>"
        )
    return f'<div class="model-calibration-table">{"".join(rows)}</div>'


def _render_model_details_popover() -> str:
    return f"""
        <div id="model-details-popover" class="model-details-popover" popover>
            <h2 class="model-details-title">About this model</h2>
            <div class="model-details-stack">
                <section class="model-details-section model-details-section-primary">
                    {_render_model_metrics_html()}
                </section>
                <section class="model-details-section">
                    <div class="model-details-section-title">Model facts</div>
                    <div class="model-facts-grid">
                        {_render_model_facts_html()}
                    </div>
                </section>
                <div class="model-details-grid">
                    <section class="model-details-section">
                        <div class="model-details-section-title">Top feature importances</div>
                        {_render_feature_importance_html()}
                    </section>
                    <section class="model-details-section">
                        <div class="model-details-section-title">Calibration</div>
                        {_render_calibration_html()}
                    </section>
                </div>
            </div>
        </div>
    """


def render_header() -> None:
    model_status_card_html = f"""
        <div class="dashboard-header">
            <div>
                <h1 class="dashboard-title">Game Predictor Dashboard</h1>
                <div class="dashboard-subtitle">
                    Compare teams, inspect win probabilities, and review the model factors behind each prediction.
                </div>
            </div>
            <div class="model-status-card">
                <div class="model-status-top">
                    <div class="model-status-title">MODEL STATUS</div>
                    <button type="button" class="model-details-button" popovertarget="model-details-popover">Details</button>
                </div>
                <div class="model-status-badges">
                    <span class="model-status-pill" data-tone="warm">Calibrated RF</span>
                    <span class="model-status-pill" data-tone="cool">Order-symmetric</span>
                </div>
                <div class="model-status-limitations">No odds • injuries • lineup/news</div>
            </div>
    </div>
    """
    st.markdown(model_status_card_html, unsafe_allow_html=True)
    st.markdown(_render_model_details_popover(), unsafe_allow_html=True)


def _team_gradient_span(abbreviation: str) -> str:
    colors = get_team_colors(abbreviation)
    safe_abbr = html.escape(str(abbreviation))
    return (
        '<span class="team-gradient-text" '
        f'style="--gradient-start:{colors["primary"]}; --gradient-end:{colors["secondary"]}; '
        f'--fallback-color:{colors["accent_text"]};">{safe_abbr}</span>'
    )


def _score_text(game: dict) -> str:
    away_score = game.get("away_score")
    home_score = game.get("home_score")
    if away_score is None or home_score is None:
        return "Final score unavailable"
    away_bold = int(away_score) > int(home_score)
    home_bold = int(home_score) > int(away_score)
    away = f"<strong>{away_score}</strong>" if away_bold else str(away_score)
    home = f"<strong>{home_score}</strong>" if home_bold else str(home_score)
    return f"{html.escape(str(game.get('away_abbr', '')))} {away} - {home} {html.escape(str(game.get('home_abbr', '')))}"


def _int_or_none(value) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _derived_series_status(game: dict, away_abbr: str, home_abbr: str) -> str:
    series_status = str(game.get("series_status") or "").strip()
    if series_status:
        return series_status

    has_playoff_marker = bool(game.get("round") or game.get("series_label") or game.get("game_number"))
    if not has_playoff_marker:
        return ""

    away_wins = _int_or_none(game.get("away_series_wins"))
    home_wins = _int_or_none(game.get("home_series_wins"))
    game_number = _int_or_none(game.get("game_number"))
    if away_wins is None and home_wins is None and game_number == 1:
        away_wins = 0
        home_wins = 0

    if away_wins is None or home_wins is None:
        return ""
    if away_wins == home_wins:
        return f"Series tied {away_wins}-{home_wins}"

    leader_abbr = away_abbr if away_wins > home_wins else home_abbr
    leader_wins = max(away_wins, home_wins)
    trailer_wins = min(away_wins, home_wins)
    verb = "wins" if leader_wins >= 4 else "leads"
    return f"{leader_abbr} {verb} {leader_wins}-{trailer_wins}"


def _live_series_status_text(game: dict) -> str:
    away_status_wins = game.get("away_series_wins")
    home_status_wins = game.get("home_series_wins")
    if away_status_wins is not None and home_status_wins is not None:
        away_wins = int(away_status_wins)
        home_wins = int(home_status_wins)
        if max(away_wins, home_wins) >= 4:
            winner = str(game.get("away_abbr") if away_wins > home_wins else game.get("home_abbr"))
            return f"{winner} wins {max(away_wins, home_wins)}-{min(away_wins, home_wins)}"

    series_status = str(game.get("series_status") or "").strip()
    match = re.search(
        r"\b([A-Z]{2,4})\s+(?:wins|leads)(?:\s+(?:the\s+)?series)?\s+(4)\s*[-–]\s*(\d+)",
        series_status,
        re.IGNORECASE,
    )
    if match:
        winner = match.group(1).upper()
        return f"{winner} wins 4-{int(match.group(3))}"
    return series_status


def live_game_context_lines(game: dict) -> list[str]:
    lines: list[str] = []
    series_status = _live_series_status_text(game)
    if series_status:
        lines.append(series_status)
    if bool(game.get("if_necessary")):
        lines.append("IF NECESSARY")
    return lines


def _plain_score_text(game: dict) -> str:
    away_score = game.get("away_score")
    home_score = game.get("home_score")
    if away_score is None or home_score is None:
        return "Final score unavailable"
    away_abbr = str(game.get("away_abbr") or "")
    home_abbr = str(game.get("home_abbr") or "")
    return f"{away_abbr} {away_score} - {home_score} {home_abbr}"


def live_game_prediction_context(game: dict) -> dict:
    game_date = live_game_date(game)
    away_series_wins = game.get("away_series_wins")
    home_series_wins = game.get("home_series_wins")
    has_series_context = (
        bool(game.get("series_status"))
        or bool(game.get("round"))
        or bool(game.get("series_label"))
        or bool(game.get("if_necessary"))
        or game.get("game_number") is not None
        or away_series_wins is not None
        or home_series_wins is not None
    )
    game_number = int(game.get("game_number") or 1)
    game_number = min(7, max(1, game_number))
    team_a_series_wins = int(away_series_wins or 0)
    team_b_series_wins = int(home_series_wins or 0)
    if game_number == 7:
        team_a_series_wins = 3
        team_b_series_wins = 3
    elif has_series_context:
        expected_total = game_number - 1
        if team_a_series_wins + team_b_series_wins != expected_total:
            team_b_series_wins = max(0, expected_total - team_a_series_wins)

    return {
        "team_a": str(game.get("away_abbr") or ""),
        "team_b": str(game.get("home_abbr") or ""),
        "season": str(game.get("season") or nba_season_from_date(game_date)),
        "prediction_date": game_date.isoformat(),
        "home_team": "team2",
        "feature_season_type": DEFAULT_SEASON_TYPE,
        "prediction_context_mode": PREDICTION_MODE_PLAYOFF if has_series_context else PREDICTION_MODE_CURRENT,
        "game_number": game_number,
        "team_a_series_wins": team_a_series_wins,
        "team_b_series_wins": team_b_series_wins,
    }


def _compute_matchup_prediction_impl(
    *,
    team_a: str,
    team_b: str,
    season: str,
    prediction_date: str,
    home_team: str,
    feature_season_type: str,
    prediction_context_mode: str,
    game_number: int = 1,
    team_a_series_wins: int = 0,
    team_b_series_wins: int = 0,
    predictor=_predict_probability,
) -> dict:
    with perf_timer("selected matchup prediction"):
        probability_direct, features, team_a_stats, team_b_stats, home_team_id = predictor(
            team_a=team_a,
            team_b=team_b,
            season=season,
            prediction_date=prediction_date,
            home_team=home_team,
            cache_dir=DEFAULT_CACHE_DIR,
            feature_season_type=feature_season_type,
            model_path=DEFAULT_MODEL_PATH,
            debug=False,
            prediction_context_mode=prediction_context_mode,
            game_number=int(game_number),
            team_a_series_wins=int(team_a_series_wins),
            team_b_series_wins=int(team_b_series_wins),
        )
        reverse_home_team = {"team1": "team2", "team2": "team1"}.get(home_team, home_team)
        probability_reverse, _, _, _, _ = predictor(
            team_a=team_b,
            team_b=team_a,
            season=season,
            prediction_date=prediction_date,
            home_team=reverse_home_team,
            cache_dir=DEFAULT_CACHE_DIR,
            feature_season_type=feature_season_type,
            model_path=DEFAULT_MODEL_PATH,
            debug=False,
            prediction_context_mode=prediction_context_mode,
            game_number=int(game_number),
            team_a_series_wins=int(team_b_series_wins),
            team_b_series_wins=int(team_a_series_wins),
        )
        probability_reverse_complement = 1 - float(probability_reverse)
        probability = (float(probability_direct) + probability_reverse_complement) / 2
        game_features = game_win_prediction_features(features, prediction_context_mode)
        series_probability = None
        if prediction_context_mode == PREDICTION_MODE_PLAYOFF:
            series_probability = simulate_best_of_seven_series_probability(
                probability,
                int(team_a_series_wins),
                int(team_b_series_wins),
            )
        return {
            "team_a_probability": float(probability),
            "team_b_probability": float(1 - probability),
            "p_direct": float(probability_direct),
            "p_reverse_complement": float(probability_reverse_complement),
            "p_symmetric_final": float(probability),
            "team_a_series_probability": series_probability,
            "team_b_series_probability": None if series_probability is None else float(1 - series_probability),
            "features": features,
            "game_features": game_features,
            "team_a_stats": team_a_stats,
            "team_b_stats": team_b_stats,
            "home_team_id": home_team_id,
            "prediction_context": {
                "season": season,
                "prediction_date": prediction_date,
                "home_team": home_team,
                "feature_season_type": feature_season_type,
                "prediction_context_mode": prediction_context_mode,
                "game_number": int(game_number),
                "team_a_series_wins": int(team_a_series_wins),
                "team_b_series_wins": int(team_b_series_wins),
            },
            "freshly_computed": True,
        }


@st.cache_data(show_spinner=False)
def _cached_default_matchup_prediction(
    team_a: str,
    team_b: str,
    season: str,
    prediction_date: str,
    home_team: str,
    feature_season_type: str,
    prediction_context_mode: str,
    game_number: int,
    team_a_series_wins: int,
    team_b_series_wins: int,
    model_version: tuple[int, int],
) -> dict:
    del model_version
    return _compute_matchup_prediction_impl(
        team_a=team_a,
        team_b=team_b,
        season=season,
        prediction_date=prediction_date,
        home_team=home_team,
        feature_season_type=feature_season_type,
        prediction_context_mode=prediction_context_mode,
        game_number=game_number,
        team_a_series_wins=team_a_series_wins,
        team_b_series_wins=team_b_series_wins,
        predictor=_predict_probability,
    )


def compute_matchup_prediction(
    *,
    team_a: str,
    team_b: str,
    season: str,
    prediction_date: str,
    home_team: str,
    feature_season_type: str,
    prediction_context_mode: str,
    game_number: int = 1,
    team_a_series_wins: int = 0,
    team_b_series_wins: int = 0,
    predictor=_predict_probability,
) -> dict:
    if predictor is _predict_probability:
        return _cached_default_matchup_prediction(
            team_a,
            team_b,
            season,
            prediction_date,
            home_team,
            feature_season_type,
            prediction_context_mode,
            int(game_number),
            int(team_a_series_wins),
            int(team_b_series_wins),
            model_artifact_version(DEFAULT_MODEL_PATH),
        )
    return _compute_matchup_prediction_impl(
        team_a=team_a,
        team_b=team_b,
        season=season,
        prediction_date=prediction_date,
        home_team=home_team,
        feature_season_type=feature_season_type,
        prediction_context_mode=prediction_context_mode,
        game_number=game_number,
        team_a_series_wins=team_a_series_wins,
        team_b_series_wins=team_b_series_wins,
        predictor=predictor,
    )


def compute_live_game_prediction(game: dict, model_available: bool, predictor=_predict_probability) -> dict:
    if not model_available:
        raise RuntimeError("Model artifact is unavailable.")
    context = live_game_prediction_context(game)
    return compute_matchup_prediction(
        **context,
        predictor=predictor,
    )


def upcoming_prediction_cache_key(
    game: dict,
    model_available: bool = True,
    model_version: tuple[int, int] | None = None,
) -> tuple:
    context = live_game_prediction_context(game)
    return (
        bool(model_available),
        str(context.get("season")),
        str(context.get("prediction_date")),
        str(context.get("team_a")),
        str(context.get("team_b")),
        str(context.get("home_team")),
        int(context.get("game_number", 1)),
        int(context.get("team_a_series_wins", 0)),
        int(context.get("team_b_series_wins", 0)),
        str(context.get("prediction_context_mode")),
        str(game.get("game_id") or ""),
        str(game.get("game_date") or ""),
        str(game.get("away_abbr") or ""),
        str(game.get("home_abbr") or ""),
        model_version if model_version is not None else model_artifact_version(DEFAULT_MODEL_PATH),
    )


@st.cache_data(show_spinner=False)
def _cached_live_card_prediction(game: dict, model_available: bool, cache_key: tuple) -> dict:
    del cache_key
    with perf_timer("upcoming card prediction computation"):
        return compute_live_game_prediction(game, model_available=model_available, predictor=_predict_probability)


def _format_matchup_probability_line(
    prefix: str,
    away_abbr: str,
    home_abbr: str,
    away_probability: float,
) -> str:
    away_pct = f"{away_probability:.0%}"
    home_pct = f"{1 - away_probability:.0%}"
    if away_probability >= 0.5:
        away_pct = f"<strong>{away_pct}</strong>"
    else:
        home_pct = f"<strong>{home_pct}</strong>"
    prefix_text = f"{prefix}: " if prefix else ""
    return f"{prefix_text}{html.escape(away_abbr)} {away_pct} - {home_pct} {html.escape(home_abbr)}"


def _upcoming_prediction_result_with_payload(
    game: dict,
    model_available: bool,
    predictor=_predict_probability,
) -> tuple[str, str | None, dict | None]:
    away_abbr = str(game.get("away_abbr", ""))
    home_abbr = str(game.get("home_abbr", ""))
    try:
        if predictor is _predict_probability:
            result = _cached_live_card_prediction(
                game,
                model_available,
                upcoming_prediction_cache_key(game, model_available, model_artifact_version(DEFAULT_MODEL_PATH)),
            )
        else:
            with perf_timer("upcoming card prediction computation"):
                result = compute_live_game_prediction(game, model_available=model_available, predictor=predictor)
    except Exception as exc:
        return "Prediction unavailable", f"{away_abbr} @ {home_abbr}: {exc}", None

    prediction = _format_matchup_probability_line(
        "",
        away_abbr,
        home_abbr,
        float(result["team_a_probability"]),
    )
    return prediction, None, result


def _upcoming_prediction_result(
    game: dict,
    model_available: bool,
    predictor=_predict_probability,
) -> tuple[str, str | None]:
    prediction, debug, _result = _upcoming_prediction_result_with_payload(game, model_available, predictor)
    return prediction, debug


def _upcoming_prediction(game: dict, model_available: bool) -> str:
    prediction, _debug = _upcoming_prediction_result(game, model_available)
    return prediction


def _plain_upcoming_prediction(prediction_html: str) -> str:
    return (
        prediction_html.replace("<strong>", "")
        .replace("</strong>", "")
        .replace("<br>", "\n")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def _latest_game_card_html(game: dict) -> str:
    context = "".join(f'<div class="live-game-note">{html.escape(line)}</div>' for line in live_game_context_lines(game))
    return (
        '<div class="live-game-card">'
        f'<div class="live-game-teams">{_team_gradient_span(game.get("away_abbr", ""))}'
        f'<span class="live-game-vs">@</span>{_team_gradient_span(game.get("home_abbr", ""))}</div>'
        f'<div class="live-game-date">{html.escape(str(game.get("time_label") or game.get("date_label") or ""))}</div>'
        f"{context}"
        f'<div class="live-game-score">{_score_text(game)}</div>'
        "</div>"
    )


def _upcoming_game_card_html_from_prediction(game: dict, prediction_html: str) -> str:
    context = "".join(f'<div class="live-game-note">{html.escape(line)}</div>' for line in live_game_context_lines(game))
    return (
        '<div class="live-game-card">'
        f'<div class="live-game-teams">{_team_gradient_span(game.get("away_abbr", ""))}'
        f'<span class="live-game-vs">@</span>{_team_gradient_span(game.get("home_abbr", ""))}</div>'
        f'<div class="live-game-date">{html.escape(str(game.get("time_label") or game.get("date_label") or ""))}</div>'
        f"{context}"
        f'<div class="live-game-prediction">{prediction_html}</div>'
        "</div>"
    )


def _upcoming_game_card_html(game: dict, model_available: bool) -> str:
    prediction, _debug = _upcoming_prediction_result(game, model_available)
    return _upcoming_game_card_html_from_prediction(game, prediction)


def build_live_games_topbar_html(live_games: dict, model_available: bool) -> str:
    live_games = live_games if isinstance(live_games, dict) else {"latest": [], "upcoming": []}
    latest_cards = [_latest_game_card_html(game) for game in reversed((live_games.get("latest") or [])[:3])]
    upcoming_cards = [_upcoming_game_card_html(game, model_available) for game in (live_games.get("upcoming") or [])[:3]]
    if not latest_cards:
        latest_cards = ['<div class="live-game-card"><div class="live-game-note">Latest games unavailable</div></div>']
    if not upcoming_cards:
        upcoming_cards = ['<div class="live-game-card"><div class="live-game-note">Upcoming games unavailable</div></div>']

    return (
        '<div class="live-games-board">'
        '<section class="live-games-section">'
        '<div class="live-games-heading">Latest Games</div>'
        f'<div class="live-games-row">{"".join(latest_cards)}</div>'
        "</section>"
        '<section class="live-games-section">'
        '<div class="live-games-heading">Upcoming Games</div>'
        f'<div class="live-games-row">{"".join(upcoming_cards)}</div>'
        "</section>"
        "</div>"
    )


def team_label_for_abbr(labels: list[str], abbreviation: str) -> str:
    suffix = f"({abbreviation})"
    for label in labels:
        if label.endswith(suffix):
            return label
    raise ValueError(f"Team abbreviation {abbreviation} is not available in the current team list.")


def live_game_date(game: dict) -> date:
    raw_value = game.get("game_datetime") or game.get("game_date")
    if not raw_value:
        return date.today()
    timestamp = pd.to_datetime(raw_value, errors="coerce")
    if pd.isna(timestamp):
        return date.today()
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    else:
        timestamp = timestamp.tz_convert("America/New_York")
    return timestamp.tz_convert("America/New_York").date()


def live_game_selection_state(game: dict, labels: list[str]) -> dict:
    away_abbr = str(game.get("away_abbr") or "")
    home_abbr = str(game.get("home_abbr") or "")
    away_label = team_label_for_abbr(labels, away_abbr)
    home_label = team_label_for_abbr(labels, home_abbr)
    game_date = live_game_date(game)
    season = str(game.get("season") or nba_season_from_date(game_date))
    if season not in SEASON_OPTIONS:
        season = DEFAULT_SEASON

    away_series_wins = game.get("away_series_wins")
    home_series_wins = game.get("home_series_wins")
    has_series_context = (
        bool(game.get("series_status"))
        or bool(game.get("round"))
        or bool(game.get("series_label"))
        or bool(game.get("if_necessary"))
        or game.get("game_number") is not None
        or away_series_wins is not None
        or home_series_wins is not None
    )
    game_number = int(game.get("game_number") or 1)
    game_number = min(7, max(1, game_number))
    team_a_series_wins = int(away_series_wins or 0)
    team_b_series_wins = int(home_series_wins or 0)
    if game_number != 7:
        team_a_series_wins = min(team_a_series_wins, max(0, game_number - 1))
        team_b_series_wins = min(team_b_series_wins, max(0, game_number - 1))

    return {
        "selected_team_label": away_label,
        "opponent_team_label": home_label,
        "home_team_label": home_label,
        "prediction_date": game_date,
        "season": season,
        "season_type": DEFAULT_SEASON_TYPE,
        "prediction_context_mode": PREDICTION_MODE_PLAYOFF if has_series_context else PREDICTION_MODE_CURRENT,
        "game_number": game_number,
        "team_a_series_wins": team_a_series_wins,
        "team_b_series_wins": team_b_series_wins,
        "raw_live_game_series_label": str(game.get("series_status") or ""),
    }


def live_game_payload(game: dict) -> dict:
    game_date = live_game_date(game)
    away_series_wins = game.get("away_series_wins")
    home_series_wins = game.get("home_series_wins")
    game_number = game.get("game_number")
    away_abbr = str(game.get("away_abbr") or "")
    home_abbr = str(game.get("home_abbr") or "")
    has_playoff_marker = bool(game.get("round") or game.get("series_label") or game_number)
    if (
        has_playoff_marker
        and _int_or_none(game_number) == 1
        and _int_or_none(away_series_wins) is None
        and _int_or_none(home_series_wins) is None
    ):
        away_series_wins = 0
        home_series_wins = 0
    return {
        "game_id": str(game.get("game_id") or ""),
        "away_abbr": away_abbr,
        "home_abbr": home_abbr,
        "game_datetime": game.get("game_datetime"),
        "game_date": game_date.isoformat(),
        "game_time": str(game.get("time_label") or game.get("date_label") or ""),
        "season": nba_season_from_date(game_date),
        "series_status": _derived_series_status(game, away_abbr, home_abbr),
        "round": game.get("round"),
        "series_label": game.get("series_label"),
        "game_number": None if game_number is None else int(game_number),
        "away_series_wins": None if away_series_wins is None else int(away_series_wins),
        "home_series_wins": None if home_series_wins is None else int(home_series_wins),
        "if_necessary": bool(game.get("if_necessary")),
        "away_score": game.get("away_score"),
        "home_score": game.get("home_score"),
        "date_label": str(game.get("date_label") or ""),
        "time_label": str(game.get("time_label") or game.get("date_label") or ""),
    }


def live_card_key(section: str, payload: dict, index: int) -> str:
    safe_game_id = "".join(character if character.isalnum() else "_" for character in str(payload.get("game_id") or "unknown"))
    return f"live_card_{section}_{safe_game_id}_{index}"


def apply_live_game_selection(game: dict, labels: list[str]) -> None:
    try:
        st.session_state.update(live_game_selection_state(dict(game), labels))
        st.session_state.pop("prediction_payload", None)
        st.session_state.pop("prediction_context", None)
        st.session_state.pop("prediction_chat_messages", None)
        st.session_state.pop("live_game_selection_error", None)
    except Exception as exc:
        st.session_state["live_game_selection_error"] = str(exc)


def live_game_button_label(game: dict, *, is_upcoming: bool, model_available: bool) -> tuple[str, str | None]:
    away_abbr = str(game.get("away_abbr") or "")
    home_abbr = str(game.get("home_abbr") or "")
    lines = [
        f"{away_abbr} @ {home_abbr}",
        str(game.get("time_label") or game.get("date_label") or ""),
        *live_game_context_lines(game),
    ]
    debug = None
    if is_upcoming:
        prediction_html, debug = _upcoming_prediction_result(game, model_available)
        lines.append(_plain_upcoming_prediction(prediction_html))
    else:
        lines.append(_plain_score_text(game))
    return "\n".join(line for line in lines if line), debug


def live_game_render_payloads(live_games: dict, section: str) -> list[dict]:
    games = (live_games.get(section) or [])[:3]
    if section == "latest":
        games = list(reversed(games))
    return [live_game_payload(game) for game in games]


def render_live_card_button(payload: dict, labels: list[str], section: str, index: int) -> None:
    away_abbr = str(payload.get("away_abbr") or "")
    home_abbr = str(payload.get("home_abbr") or "")
    st.markdown('<div class="live-game-load-note">Click to load this matchup</div>', unsafe_allow_html=True)
    st.button(
        f"Load {away_abbr} @ {home_abbr}",
        key=live_card_key(section, payload, index),
        on_click=apply_live_game_selection,
        args=(dict(payload), labels),
        width="stretch",
    )


def render_live_games_topbar(live_games: dict, model_available: bool, labels: list[str]) -> None:
    live_games = live_games if isinstance(live_games, dict) else {"latest": [], "upcoming": []}
    st.markdown('<div class="live-games-board streamlit-live-games">', unsafe_allow_html=True)
    latest_section, upcoming_section = st.columns(2, gap="medium")
    debug_rows: list[dict[str, str]] = []

    with latest_section:
        st.markdown('<div class="live-games-heading">Latest Games</div>', unsafe_allow_html=True)
        latest_games = live_game_render_payloads(live_games, "latest")
        if not latest_games:
            st.markdown('<div class="live-game-note">Latest games unavailable</div>', unsafe_allow_html=True)
        else:
            latest_cols = st.columns(len(latest_games), gap="small")
            for index, payload in enumerate(latest_games):
                with latest_cols[index]:
                    st.markdown(_latest_game_card_html(payload), unsafe_allow_html=True)
                    render_live_card_button(payload, labels, "latest", index)

    with upcoming_section:
        st.markdown('<div class="live-games-heading">Upcoming Games</div>', unsafe_allow_html=True)
        upcoming_games = live_game_render_payloads(live_games, "upcoming")
        if not upcoming_games:
            st.markdown('<div class="live-game-note">Upcoming games unavailable</div>', unsafe_allow_html=True)
        else:
            upcoming_cols = st.columns(len(upcoming_games), gap="small")
            for index, payload in enumerate(upcoming_games):
                prediction, debug, computed = _upcoming_prediction_result_with_payload(payload, model_available)
                prediction_debug = {
                    "game_id": str(payload.get("game_id") or ""),
                    "game": f"{payload.get('away_abbr')} @ {payload.get('home_abbr')}",
                    "card_prediction_context": json.dumps(live_game_prediction_context(payload), sort_keys=True),
                    "card_game_probability": None,
                    "card_series_probability": None,
                    "p_direct": None,
                    "p_reverse_complement": None,
                    "p_symmetric_final": None,
                    "freshly_computed": False,
                    "reason": debug or "",
                }
                if computed:
                    prediction_debug["card_game_probability"] = computed.get("team_a_probability")
                    prediction_debug["card_series_probability"] = computed.get("team_a_series_probability")
                    prediction_debug["p_direct"] = computed.get("p_direct")
                    prediction_debug["p_reverse_complement"] = computed.get("p_reverse_complement")
                    prediction_debug["p_symmetric_final"] = computed.get("p_symmetric_final")
                    prediction_debug["freshly_computed"] = bool(computed.get("freshly_computed"))
                debug_rows.append(prediction_debug)
                if debug:
                    debug_rows[-1]["reason"] = debug
                with upcoming_cols[index]:
                    st.markdown(_upcoming_game_card_html_from_prediction(payload, prediction), unsafe_allow_html=True)
                    render_live_card_button(payload, labels, "upcoming", index)

    st.markdown("</div>", unsafe_allow_html=True)
    selection_error = st.session_state.get("live_game_selection_error")
    if selection_error:
        debug_rows.append({"game_id": "", "game": "Live card selection", "reason": selection_error})
    st.session_state["live_game_prediction_debug"] = debug_rows


def render_live_games_topbar_safe(model_available: bool, labels: list[str]) -> None:
    try:
        live_games = safe_live_games_payload()
        if not live_games.get("latest") and not live_games.get("upcoming"):
            st.session_state["live_game_prediction_debug"] = []
            st.markdown(
                """
                <div class="panel-note" style="margin:0.15rem 0 0.75rem;">
                    Live games unavailable right now.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        render_live_games_topbar(live_games, model_available=model_available, labels=labels)
    except Exception:
        st.session_state["live_game_prediction_debug"] = []
        st.markdown(
            """
            <div class="panel-note" style="margin:0.15rem 0 0.75rem;">
                Live games unavailable right now.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_matchup_strip(
    team_a: pd.Series,
    team_b: pd.Series,
    home_team_label: str,
    team_a_label: str,
    team_b_label: str,
    team_a_colors: dict[str, str],
    team_b_colors: dict[str, str],
) -> None:
    team_a_home = "Home" if home_team_label == team_a_label else "Away"
    team_b_home = "Home" if home_team_label == team_b_label else "Away"
    st.markdown(
        f"""
            <div class="matchup-strip">
            <div class="matchup-title">
                <span class="team-gradient-text" style="--gradient-start:{team_a_colors['primary']}; --gradient-end:{team_a_colors['secondary']}; --fallback-color:{team_a_colors['accent_text']};">{team_a['TEAM_ABBREVIATION']}</span>
                <span style="color:{MUTED}; font-size:1.6rem;">vs</span>
                <span class="team-gradient-text" style="--gradient-start:{team_b_colors['primary']}; --gradient-end:{team_b_colors['secondary']}; --fallback-color:{team_b_colors['accent_text']};">{team_b['TEAM_ABBREVIATION']}</span>
            </div>
            <div class="matchup-subtitle">
                {team_a['TEAM_NAME']} ({team_a_home}) &nbsp;|&nbsp; {team_b['TEAM_NAME']} ({team_b_home})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_summary(
    team_a_display: str,
    team_b_display: str,
    team_a_probability: float,
    home_team: str,
    title: str = "Game Win Probability",
) -> None:
    team_b_probability = 1 - team_a_probability
    a_width = max(0, min(100, team_a_probability * 100))
    b_width = 100 - a_width
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="prob-meter">
                <div class="prob-meter-a" style="width:{a_width:.2f}%"></div>
                <div class="prob-meter-b" style="width:{b_width:.2f}%"></div>
            </div>
            <div class="meter-labels">
                <span>{team_a_display}: {team_a_probability:.1%}</span>
                <span>{team_b_display}: {team_b_probability:.1%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_series_probability_summary(
    team_a_display: str,
    team_b_display: str,
    team_a_series_probability: float,
) -> None:
    render_probability_summary(
        team_a_display,
        team_b_display,
        team_a_series_probability,
        home_team="",
        title="Series Win Probability",
    )


def confidence_label(probability_gap: float) -> str:
    if probability_gap < 0.10:
        return "Low"
    if probability_gap < 0.20:
        return "Medium"
    return "High"


def basketball_feature_label(feature: str) -> str:
    labels = {
        "home_team_A": "home court",
        "home_win_pct_diff": "home/away record edge",
        "away_win_pct_diff": "opposite home/away split",
        "home_advantage_diff": "home-court advantage edge",
        "team_A_home_advantage": "selected team's home edge",
        "team_B_home_advantage": "opponent home edge",
        "clipped_home_win_pct_diff": "clipped home/away record edge",
        "clipped_away_win_pct_diff": "clipped opposite home/away split",
        "seed_difference": "playoff seed edge",
        "higher_seed_A": "higher seed flag",
        "efg_pct_diff": "effective field goal percentage",
        "ts_pct_diff": "true shooting efficiency",
        "game_number": "series game number",
        "elo_diff": "pre-game Elo rating",
        "OFF_RATING_DIFF": "offensive rating",
        "DEF_RATING_DIFF": "defensive rating",
        "NET_RATING_DIFF": "net rating",
        "W_PCT_DIFF": "season win percentage",
        "PLUS_MINUS_DIFF": "point differential",
        "PACE_DIFF": "pace",
        "season_h2h_win_pct_diff": "season head-to-head results",
        "season_h2h_margin_diff": "head-to-head scoring margin",
        "h2h_win_pct_diff": "head-to-head win percentage",
        "h2h_margin_diff": "head-to-head scoring margin",
        "h2h_net_rating_diff": "head-to-head net rating",
        "h2h_efg_pct_diff": "head-to-head shooting efficiency",
        "h2h_ts_pct_diff": "head-to-head true shooting",
        "h2h_turnover_pct_diff": "head-to-head turnover edge",
        "h2h_reb_pct_diff": "head-to-head rebounding",
        "series_score_diff": "current series score",
        "last_5_win_pct_diff": "last five games",
        "last_10_win_pct_diff": "last ten games",
        "last_5_net_rating_diff": "last five net rating",
        "last_10_net_rating_diff": "last ten net rating",
        "last_5_point_diff_diff": "last five point margin",
        "last_10_point_diff_diff": "last ten point margin",
        "top_3_ppg_diff": "top-three scorer production",
        "top_5_ppg_diff": "top-five scorer production",
        "rest_days_diff": "rest advantage",
        "weighted_recent_win_pct_diff": "weighted recent wins",
        "weighted_recent_net_rating_diff": "weighted recent net rating",
        "weighted_recent_ts_pct_diff": "weighted recent true shooting",
        "weighted_recent_def_rating_diff": "weighted recent defense",
        "three_point_offense_vs_defense_diff": "three-point matchup",
        "paint_scoring_vs_paint_defense_diff": "two-point scoring matchup",
        "offensive_rebound_vs_defensive_rebound_diff": "rebounding matchup",
        "turnover_creation_vs_turnover_rate_diff": "turnover matchup",
        "ft_rate_vs_foul_rate_diff": "free throw matchup",
        "top_1_ppg_diff": "top scorer production",
        "top_1_mpg_diff": "top scorer minutes",
        "top_1_ts_pct_diff": "top scorer efficiency",
        "top_3_ts_pct_diff": "top-three scorer efficiency",
    }
    return labels.get(feature, feature.replace("_", " ").replace("DIFF", "difference").lower())


FEATURE_SYNONYMS = {
    "PLUS_MINUS_DIFF": ["point differential", "point diff", "scoring margin", "margin"],
    "NET_RATING_DIFF": ["net rating", "net"],
    "OFF_RATING_DIFF": ["offensive rating", "offense", "off rating"],
    "DEF_RATING_DIFF": ["defensive rating", "defense", "def rating"],
    "W_PCT_DIFF": ["win percentage", "win pct", "record"],
    "efg_pct_diff": ["efg", "effective field goal", "shooting efficiency"],
    "ts_pct_diff": ["ts", "true shooting"],
    "PACE_DIFF": ["pace", "tempo"],
    "elo_diff": ["elo"],
    "seed_difference": ["seed", "seeding", "seed difference"],
    "home_win_pct_diff": ["home split", "home/away", "home court", "home-court"],
    "home_advantage_diff": ["home advantage", "home court", "home-court"],
    "season_h2h_win_pct_diff": ["head to head", "h2h"],
    "h2h_win_pct_diff": ["head to head", "h2h"],
    "last_5_win_pct_diff": ["last 5", "last five", "recent form"],
    "last_10_win_pct_diff": ["last 10", "last ten"],
    "top_3_ppg_diff": ["top 3", "star power", "scorers"],
    "rest_days_diff": ["rest", "fatigue"],
    "three_point_offense_vs_defense_diff": ["three point", "3 point", "three-point"],
    "weighted_recent_net_rating_diff": ["weighted recent", "recent net"],
}


FEATURE_MEANINGS = {
    "PLUS_MINUS_DIFF": "Point differential measures how much a team outscored opponents overall, so it captures not just wins, but how convincing those wins were. Higher is better.",
    "NET_RATING_DIFF": "Net rating is scoring margin per 100 possessions. Higher is better because it means a team is winning the possession-by-possession math.",
    "OFF_RATING_DIFF": "Offensive rating is points scored per 100 possessions. Higher is better because it means the team produces more efficiently after adjusting for pace.",
    "DEF_RATING_DIFF": "Defensive rating is points allowed per 100 possessions. Lower is better because the team is giving up fewer points.",
    "W_PCT_DIFF": "Win percentage is how often a team won its games. Higher is better, though the model weighs it alongside margin and efficiency.",
    "efg_pct_diff": "eFG% is shooting efficiency adjusted for the extra value of three-pointers. Higher is better.",
    "ts_pct_diff": "True shooting percentage measures overall scoring efficiency, including twos, threes, and free throws. Higher is better.",
    "PACE_DIFF": "Pace is possessions per game. It is not strictly better or worse by itself; it describes tempo and style fit.",
    "elo_diff": "Elo is a rolling team-strength rating that updates after games. Higher is better.",
    "seed_difference": "Seed difference captures playoff seeding context. Lower NBA seeds are better, so a smaller seed number is an advantage.",
    "home_win_pct_diff": "Home/away split advantage compares the home team's home record against the opponent's road record.",
    "home_advantage_diff": "Home advantage compares the selected home team's home record to its own overall record, with the input clipped so it cannot overpower the prediction.",
    "season_h2h_win_pct_diff": "Head-to-head win percentage looks at how these teams did against each other earlier in the season.",
    "h2h_win_pct_diff": "Head-to-head win percentage looks at how these teams did against each other earlier in the season.",
    "h2h_margin_diff": "Head-to-head margin captures scoring margin in prior meetings between these teams.",
    "three_point_offense_vs_defense_diff": "The three-point matchup compares each team's three-point attack against the opponent's three-point prevention.",
    "weighted_recent_net_rating_diff": "Weighted recent net rating gives more weight to the latest games while still using only games before the prediction date.",
    "last_5_win_pct_diff": "Last-five win percentage captures short-term form entering the matchup.",
    "last_10_win_pct_diff": "Last-ten win percentage captures a broader recent-form window.",
    "top_3_ppg_diff": "Top-three scorer production estimates how much high-end scoring punch each team has.",
    "rest_days_diff": "Rest-days difference captures whether one team has had more time off before the game.",
    "series_score_diff": "Series score is selected team wins minus opponent wins before this game. It is semantic series context, not a normal game-win factor: positive means selected team leads, negative means opponent leads, and zero means tied.",
}


def team_chat_name(context: dict, side: str) -> str:
    team = context["team_a"] if side == "Team A" else context["team_b"]
    suffix = " at home" if "(Home)" in team["display"] else ""
    return f"{team['name']} ({team['abbreviation']}){suffix}"


def other_side(side: str) -> str:
    return "Team B" if side == "Team A" else "Team A"


CHAT_FEATURE_LABELS = {
    "clipped_home_win_pct_diff": "home/away record edge",
    "clipped_away_win_pct_diff": "road/home split edge",
    "W_PCT_DIFF": "season win percentage",
    "PLUS_MINUS_DIFF": "point differential",
    "NET_RATING_DIFF": "net rating",
    "OFF_RATING_DIFF": "offensive rating",
    "DEF_RATING_DIFF": "defensive rating",
    "PACE_DIFF": "pace",
    "seed_difference": "seeding",
    "higher_seed_A": "higher seed",
}

CHAT_FEATURE_MEANINGS = {
    "clipped_home_win_pct_diff": "Compares how the home team has performed at home against the opponent's road record.",
    "clipped_away_win_pct_diff": "Compares how the away team has performed on the road against the opponent's home record.",
    "W_PCT_DIFF": "Captures the season-long win-rate gap between the teams.",
    "PLUS_MINUS_DIFF": "Captures how much teams have outscored or been outscored by opponents overall.",
    "NET_RATING_DIFF": "Measures scoring margin per 100 possessions.",
    "OFF_RATING_DIFF": "Measures points scored per 100 possessions.",
    "DEF_RATING_DIFF": "Measures points allowed per 100 possessions; lower is better, but direction comes from model contribution.",
    "PACE_DIFF": "Measures possessions per game and style tempo.",
    "seed_difference": "Captures the playoff seeding gap.",
    "higher_seed_A": "Indicates whether the selected team is the higher seed.",
}

CHAT_MODEL_LIMITATIONS = [
    "The model uses the current matchup and model features only.",
    "It does not include injuries, player availability, lineup news, trades, betting odds, or external reports.",
    "Top factors explain the model's feature evidence, not a complete basketball scouting report.",
]


def chat_feature_label(feature: str) -> str:
    return CHAT_FEATURE_LABELS.get(feature, basketball_feature_label(feature))


def chat_feature_meaning(feature: str) -> str:
    return CHAT_FEATURE_MEANINGS.get(
        feature,
        f"{chat_feature_label(feature).capitalize()} is one of the model inputs for this matchup.",
    )


def clean_chat_factor(row: dict | pd.Series, context: dict | None = None) -> dict:
    feature = str(row.get("feature", ""))
    side = factor_helped_side(row)
    favors = side
    if context is not None and side is not None:
        favors = team_chat_name(context, side)
    return {
        "feature": feature,
        "pushes_toward": side,
        "signed_contribution": row.get("signed_contribution"),
        "raw_feature": feature,
        "friendly_label": chat_feature_label(feature),
        "value": row.get("value"),
        "global_importance": row.get("global_importance"),
        "local_effect": row.get("signed_contribution"),
        "favors": favors,
        "short_basketball_meaning": chat_feature_meaning(feature),
    }


def clean_chat_factors(factors: pd.DataFrame, context: dict | None = None) -> list[dict]:
    if factors is None or factors.empty:
        return []
    return [clean_chat_factor(row, context) for row in factors.to_dict(orient="records")]


def factor_helped_side(row: dict | pd.Series) -> str | None:
    """Return model-provided direction. Do not infer direction from raw feature signs."""
    side = str(row.get("pushes_toward", "")).strip()
    if side in {"Team A", "Team B"}:
        return side
    return None


def factor_basketball_sentence(row: dict | pd.Series, context: dict) -> str:
    feature = str(row["feature"])
    side = factor_helped_side(row) or "Team A"
    team = team_chat_name(context, side)

    actual_series_status = str(context.get("series_status_text") or "").strip()
    series_score_value = float(context.get("series_score_diff") or 0.0)
    if series_score_value > 0:
        deterministic_series_direction = team_chat_name(context, "Team A")
    elif series_score_value < 0:
        deterministic_series_direction = team_chat_name(context, "Team B")
    else:
        deterministic_series_direction = "neither team"
    series_description = (
        f"The series score context points toward {deterministic_series_direction}. "
        f"The actual series status is {actual_series_status}."
        if actual_series_status
        else f"The series score context points toward {deterministic_series_direction}."
    )
    descriptions = {
        "OFF_RATING_DIFF": f"Offensive rating helped {team} in this model read. Offensive rating is points scored per 100 possessions.",
        "DEF_RATING_DIFF": f"Defensive rating helped {team} in this model read. Defensive rating is points allowed per 100 possessions, but this direction comes from the model contribution rather than a raw-value rule.",
        "NET_RATING_DIFF": f"Net rating helped {team} in this model read. Net rating captures scoring margin per 100 possessions.",
        "W_PCT_DIFF": f"Season win percentage helped {team} in this model read.",
        "PLUS_MINUS_DIFF": f"Point differential helped {team} in this model read. It captures how convincingly a team has outscored opponents overall.",
        "PACE_DIFF": f"Pace helped {team} in this model read. Pace measures possessions per game and can reflect style fit.",
        "efg_pct_diff": f"eFG% helped {team} in this model read. eFG% measures shooting efficiency with extra credit for made threes.",
        "ts_pct_diff": f"True shooting helped {team} in this model read. TS% includes twos, threes, and free throws.",
        "elo_diff": f"Elo helped {team} in this model read. Elo is the rolling team-strength signal.",
        "seed_difference": f"Playoff seeding helped {team} in this model read.",
        "higher_seed_A": f"The higher-seed flag helped {team} in this model read.",
        "home_team_A": f"Home-court status helped {team} in this model read.",
        "home_advantage_diff": f"The selected home team's own home-court edge helped {team} in this model read.",
        "home_win_pct_diff": f"The home/away split helped {team} in this model read, comparing the home team's home record against the away team's road record.",
        "away_win_pct_diff": f"The opposite home/away split helped {team} in this model read.",
        "season_h2h_win_pct_diff": f"Same-season head-to-head results helped {team} in this model read.",
        "season_h2h_margin_diff": f"Head-to-head scoring margin helped {team} in this model read.",
        "series_score_diff": series_description,
        "last_5_win_pct_diff": f"Recent form over the last five games helped {team} in this model read.",
        "last_10_win_pct_diff": f"Recent form over the last ten games helped {team} in this model read.",
        "last_5_net_rating_diff": f"Short-term net rating over the last five games helped {team} in this model read.",
        "last_10_net_rating_diff": f"Short-term net rating over the last ten games helped {team} in this model read.",
        "last_5_point_diff_diff": f"Recent point margin over the last five games helped {team} in this model read.",
        "last_10_point_diff_diff": f"Recent point margin over the last ten games helped {team} in this model read.",
        "top_3_ppg_diff": f"Top-end scoring production from leading players helped {team} in this model read.",
        "top_5_ppg_diff": f"Broader scorer production across the top five players helped {team} in this model read.",
        "rest_days_diff": f"The rest situation helped {team} in this model read.",
    }
    return descriptions.get(feature, f"{basketball_feature_label(feature).capitalize()} helped {team} in this model read.")


def series_context_sentence(context: dict) -> str:
    if context.get("prediction_context_mode") != PREDICTION_MODE_PLAYOFF:
        return ""
    game_number = int(context.get("user_series_context", {}).get("game_number", 1) or 1)
    status = str(context.get("series_status_text") or "").strip()
    home_team_text = str(context.get("home_team", ""))
    team_a_abbr = str(context["team_a"]["abbreviation"])
    team_b_abbr = str(context["team_b"]["abbreviation"])
    if team_a_abbr in home_team_text:
        home_side = "Team A"
    elif team_b_abbr in home_team_text:
        home_side = "Team B"
    else:
        home_side = "Team A" if "(Home)" in str(context.get("team_a", {}).get("display", "")) else "Team B"
    away_side = other_side(home_side)
    away = context["team_a"]["abbreviation"] if away_side == "Team A" else context["team_b"]["abbreviation"]
    home = context["team_a"]["abbreviation"] if home_side == "Team A" else context["team_b"]["abbreviation"]
    if not status:
        return f"Series context: unavailable. This prediction is for Game {game_number}: {away} away at {home} home."
    return f"Series context: {status}. This prediction is for Game {game_number}: {away} away at {home} home."


def text_contradicts_series_status(text: str, context: dict) -> bool:
    status = str(context.get("series_status_text") or "")
    if not status:
        return False
    lowered = str(text or "").lower()
    team_a = context["team_a"]
    team_b = context["team_b"]
    team_phrases = {
        team_a["abbreviation"]: [
            str(team_a["abbreviation"]).lower(),
            str(team_a["name"]).lower(),
            str(team_a["name"]).split()[0].lower(),
            str(team_a["name"]).split()[-1].lower(),
        ],
        team_b["abbreviation"]: [
            str(team_b["abbreviation"]).lower(),
            str(team_b["name"]).lower(),
            str(team_b["name"]).split()[0].lower(),
            str(team_b["name"]).split()[-1].lower(),
        ],
    }
    tied = "series tied" in status.lower()
    if tied:
        return any(
            f"{phrase} leads" in lowered
            or f"{phrase} leading" in lowered
            or f"{phrase} is leading" in lowered
            or f"{phrase} are leading" in lowered
            or f"{phrase} holding a series lead" in lowered
            or f"{phrase} holds a series lead" in lowered
            for phrases in team_phrases.values()
            for phrase in phrases
        )

    leader = str(context.get("series_leader") or "").upper()
    non_leader_phrases = [
        phrase
        for abbreviation, phrases in team_phrases.items()
        if abbreviation.upper() != leader
        for phrase in phrases
    ]
    return any(
        f"{phrase} leads" in lowered
        or f"{phrase} leading" in lowered
        or f"{phrase} is leading" in lowered
        or f"{phrase} are leading" in lowered
        or f"{phrase} holding a series lead" in lowered
        or f"{phrase} holds a series lead" in lowered
        for phrase in non_leader_phrases
    )


def factor_lines_for_side(context: dict, side: str, limit: int = 5) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for row in context["top_factors"]:
        feature = str(row["feature"])
        if feature in seen:
            continue
        if factor_helped_side(row) != side:
            continue
        lines.append(factor_basketball_sentence(row, context))
        seen.add(feature)
        if len(lines) >= limit:
            break
    return lines


def probability_read(context: dict) -> str:
    return (
        f"{context['team_a']['name']} ({context['team_a']['abbreviation']}) is at "
        f"{context['team_a']['win_probability']:.1%}, while "
        f"{context['team_b']['name']} ({context['team_b']['abbreviation']}) is at "
        f"{context['team_b']['win_probability']:.1%}."
    )


def side_display(context: dict, side: str) -> str:
    team = context["team_a"] if side == "Team A" else context["team_b"]
    return f"{team['name']} ({team['abbreviation']})"


def side_probability(context: dict, side: str) -> float:
    return float(context["team_a" if side == "Team A" else "team_b"]["win_probability"])


def mentioned_side(context: dict, question: str) -> str | None:
    question_lower = question.lower()
    for side, key in [("Team A", "team_a"), ("Team B", "team_b")]:
        team = context[key]
        candidates = {
            str(team["name"]).lower(),
            str(team["abbreviation"]).lower(),
            str(team["display"]).lower(),
        }
        if any(candidate and candidate in question_lower for candidate in candidates):
            return side
    return None


def detect_chat_intent(question: str) -> str | None:
    question_lower = question.lower()
    if any(phrase in question_lower for phrase in ["most important indicator", "biggest reason", "single biggest factor", "pick one reason", "strongest factor", "main indicator"]):
        return "strongest_factor"
    if any(term in question_lower for term in ["home court", "home-court", "home team", "home affect", "home impact"]):
        return "home_court"
    if any(phrase in question_lower for phrase in ["how much stronger", "who is stronger", "stronger than", "better than", "why are they better", "compare", "gap between", "edge over"]):
        return "team_strength"
    if any(phrase in question_lower for phrase in ["how reliable", "why low confidence", "why high confidence", "how confident", "confidence", "sure", "certain", "reliable"]):
        return "confidence"
    if mentioned_feature(question_lower):
        return "feature_impact"
    if any(phrase in question_lower for phrase in ["underdog", "needs to", "path to win", "how can", "upset", "competitive"]):
        return "underdog_path"
    if any(word in question_lower for word in ["why", "reason", "factor", "important"]):
        return "why_favored"
    return None


def mentioned_feature(question: str) -> str | None:
    question_lower = question.lower()
    for feature, terms in FEATURE_SYNONYMS.items():
        if any(term in question_lower for term in terms):
            return feature
    return None


def top_contribution_factor(context: dict) -> dict | None:
    factors = context.get("top_factors", [])
    if not factors:
        return None
    return max(factors, key=lambda row: abs(float(row.get("signed_contribution", 0.0) or 0.0)))


def strongest_factor_answer(context: dict) -> str:
    row = top_contribution_factor(context)
    if row is None:
        return "The current prediction context does not include enough factor detail to isolate one strongest signal."

    feature = str(row["feature"])
    side = factor_helped_side(row) or "Team A"
    team = side_display(context, side)
    label = basketball_feature_label(feature)
    meaning = FEATURE_MEANINGS.get(feature, factor_basketball_sentence(row, context))
    return f"{label.capitalize()} appears to be the strongest signal here. {meaning} In this model read, pushes_toward says it helped {team}, which is why it nudges the prediction toward them."


def feature_impact_answer(context: dict, question: str) -> str:
    feature = mentioned_feature(question)
    if feature is None:
        return strongest_factor_answer(context)

    row = next((factor for factor in context.get("top_factors", []) if factor.get("feature") == feature), None)
    meaning = FEATURE_MEANINGS.get(feature, f"{basketball_feature_label(feature).capitalize()} is one of the model inputs for this matchup.")
    if row is None:
        return f"{meaning} It is part of the feature row, but it is not available in the current top-factor context with a model direction."

    side = factor_helped_side(row)
    if side is None:
        return f"{meaning} The current context does not include a pushes_toward direction for this feature, so I should not infer which team it helped from the raw value."
    team = side_display(context, side)
    return f"{meaning} In this model read, pushes_toward says it helped {team}, so it acts as one of the top signals shifting probability in their direction."


def home_scenario_summary(context: dict) -> str:
    scenarios = context.get("home_scenarios", [])
    if len(scenarios) >= 2:
        lines = []
        probabilities = []
        for row in scenarios:
            if row.get("team_A_probability") is None:
                continue
            team_a_probability = float(row["team_A_probability"])
            probabilities.append(team_a_probability)
            lines.append(
                f"- {row['selection']}: {context['team_a']['abbreviation']} {team_a_probability:.1%}, "
                f"{context['team_b']['abbreviation']} {1 - team_a_probability:.1%}"
            )
        if lines:
            swing = abs(max(probabilities) - min(probabilities)) * 100 if len(probabilities) >= 2 else 0.0
            return (
                "Here is the home-court comparison from the same matchup:\n"
                + "\n".join(lines)
                + f"\nThat is about a {swing:.1f}-point swing in {context['team_a']['abbreviation']} win probability."
            )

    home_side = "Team A" if context["feature_values"].get("home_team_A") == 1.0 else "Team B"
    return (
        f"{team_chat_name(context, home_side)} is home in the current prediction. "
        "The production model uses the home-team flag plus a clipped home-advantage signal, so home court can matter without letting home/away record splits overpower the matchup."
    )


def home_court_answer(context: dict) -> str:
    return (
        f"{home_scenario_summary(context)} "
        "That swing comes from model features, not from travel news, crowd noise, or arena-specific information outside the dataset."
    )


def strength_comparison_answer(context: dict, question: str) -> str:
    favorite_side = "Team A" if context["favorite"]["side"] == "team_a" else "Team B"
    underdog_side = other_side(favorite_side)
    requested_side = mentioned_side(context, question)
    lead_side = requested_side or favorite_side
    other = other_side(lead_side)
    lead_team = side_display(context, lead_side)
    other_team = side_display(context, other)
    lead_probability = context["team_a" if lead_side == "Team A" else "team_b"]["win_probability"]
    other_probability = context["team_a" if other == "Team A" else "team_b"]["win_probability"]
    lead_lines = factor_lines_for_side(context, lead_side, limit=3)
    other_lines = factor_lines_for_side(context, other, limit=2)
    if not lead_lines:
        lead_lines = ["The top model factors are fairly balanced rather than clearly concentrated on one team."]
    if not other_lines:
        other_lines = ["The other side has fewer top-ranked model supports in this prediction context."]

    support = " ".join(lead_lines[:2])
    resistance = f" {other_lines[0]}" if other_lines else ""
    return (
        f"{lead_team} is at {lead_probability:.1%} compared with {other_team} at {other_probability:.1%}, "
        f"a {abs(lead_probability - other_probability) * 100:.1f}-point probability gap with {context['confidence'].lower()} confidence. "
        f"The strongest supports are: {support}{resistance}"
    )


def underdog_answer(context: dict) -> str:
    favorite_side = "Team A" if context["favorite"]["side"] == "team_a" else "Team B"
    underdog_side = other_side(favorite_side)
    underdog = side_display(context, underdog_side)
    favorite = side_display(context, favorite_side)
    underdog_lines = factor_lines_for_side(context, underdog_side, limit=3)
    if not underdog_lines:
        underdog_lines = [
            "keep the game close enough that shooting variance, transition chances, and late-game execution can matter",
            "turn the model's weaker areas into neutral categories rather than letting them become clear disadvantages",
        ]
    return "\n".join(
        [
            f"For {underdog} to beat this model read against {favorite}, the path is to shrink the biggest team-strength gaps and win the high-leverage possessions.",
            "The factors giving the underdog some support are:",
            *[f"- {line}" for line in underdog_lines],
            "The app does not know injuries, rotations, or game-plan changes, so this is strictly based on the current model context.",
        ]
    )


def confidence_answer(context: dict) -> str:
    favorite_side = "Team A" if context["favorite"]["side"] == "team_a" else "Team B"
    favorite = side_display(context, favorite_side)
    underdog = side_display(context, other_side(favorite_side))
    margin = float(context["favorite"]["margin_pp"])
    if context["confidence"] == "Low":
        reason = "the probability gap is narrow, so the model sees the matchup as relatively close."
    elif context["confidence"] == "Medium":
        reason = "the probability gap is noticeable but not overwhelming."
    else:
        reason = "the probability gap is large enough that the model sees a clear favorite."
    return f"Reliability is {context['confidence'].lower()}: {favorite} is ahead of {underdog} by {margin:.1f} probability points, and {reason} It still does not account for injuries, lineup changes, or news outside the dataset."


def biggest_advantage_answer(context: dict) -> str:
    favorite_side = "Team A" if context["favorite"]["side"] == "team_a" else "Team B"
    favorite = side_display(context, favorite_side)
    top_factor = next(
        (row for row in context.get("top_factors", []) if factor_helped_side(row) == favorite_side),
        None,
    )
    if top_factor is None:
        return (
            f"{favorite}'s biggest model advantage is not identifiable from the current top-factor context.\n"
            "- The available context has the prediction, margin, and confidence, but no ranked factor that clearly favors the predicted team.\n"
            "- I should not infer an advantage from raw feature signs without model direction."
        )
    label = str(top_factor.get("friendly_label") or chat_feature_label(str(top_factor.get("feature", ""))))
    meaning = str(top_factor.get("short_basketball_meaning") or chat_feature_meaning(str(top_factor.get("feature", ""))))
    return "\n".join(
        [
            f"{favorite}'s biggest model advantage is {label}.",
            f"- Model evidence: {meaning}",
            f"- This is the highest-ranked current factor in the chat context that favors {favorite}.",
            f"- The model still rates the overall edge as {context['confidence'].lower()} confidence.",
        ]
    )


def routed_chat_answer(context: dict, user_question: str) -> str | None:
    question = user_question.lower()
    if any(phrase in question for phrase in ["biggest advantage", "largest advantage", "main advantage", "strongest advantage"]):
        return biggest_advantage_answer(context)
    if any(word in question for word in ["injury", "injuries", "hurt", "available", "lineup", "trade", "news"]):
        return (
            "The app does not include injuries, lineup news, trades, or current roster context. "
            f"Using only the model context, {probability_read(context)}"
        )

    intent = detect_chat_intent(user_question)
    if intent == "strongest_factor":
        return strongest_factor_answer(context)
    if intent == "feature_impact":
        return feature_impact_answer(context, user_question)
    if intent == "team_strength":
        return strength_comparison_answer(context, user_question)
    if intent == "home_court":
        return home_court_answer(context)
    if intent == "confidence":
        return confidence_answer(context)
    if intent == "underdog_path":
        return underdog_answer(context)
    if intent == "why_favored":
        favorite_side = "Team A" if context["favorite"]["side"] == "team_a" else "Team B"
        lines = factor_lines_for_side(context, favorite_side, limit=3)
        return f"{side_display(context, favorite_side)} is favored mainly because " + " ".join(lines[:3])
    return None


def markdown_to_basic_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_lines: list[str] = []
    in_list = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(line[2:])}</li>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        if line.startswith("**") and line.endswith("**"):
            html_lines.append(f"<p><strong>{html.escape(line.strip('*'))}</strong></p>")
        else:
            html_lines.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "".join(html_lines)


def _build_prediction_context_impl(
    *,
    season: str,
    season_type: str,
    prediction_date: date,
    home_team: str,
    team_a: pd.Series,
    team_b: pd.Series,
    team_a_label: str,
    team_b_label: str,
    team_a_display: str,
    team_b_display: str,
    team_a_probability: float,
    features: dict[str, float],
    factors: pd.DataFrame,
    importances: pd.DataFrame,
    model_bundle: dict,
    feature_columns: list[str],
    team_a_series_probability: float | None = None,
    home_debug_rows: list[dict] | None = None,
    prediction_context_mode: str = "Current Hypothetical",
    game_number: int = 1,
    team_a_series_wins: int = 0,
    team_b_series_wins: int = 0,
    raw_live_game_series_label: str = "",
) -> dict:
    team_b_probability = 1 - team_a_probability
    team_b_series_probability = None if team_a_series_probability is None else 1 - team_a_series_probability
    if team_a_probability >= team_b_probability:
        favorite_key = "team_a"
        favorite_name = team_a_display
        underdog_name = team_b_display
        favorite_probability = team_a_probability
    else:
        favorite_key = "team_b"
        favorite_name = team_b_display
        underdog_name = team_a_display
        favorite_probability = team_b_probability

    margin = abs(team_a_probability - team_b_probability)
    top_factors = filter_noninformative_factors(factors).head(10).copy()
    favorite_side = "Team A" if favorite_key == "team_a" else "Team B"
    underdog_side = "Team B" if favorite_key == "team_a" else "Team A"
    favorite_factors = top_factors[top_factors["pushes_toward"].eq(favorite_side)].head(5)
    underdog_factors = top_factors[top_factors["pushes_toward"].eq(underdog_side)].head(5)

    clean_features = {column: None if pd.isna(features.get(column)) else float(features.get(column)) for column in feature_columns}
    chat_team_context = {
        "team_a": {
            "name": str(team_a["TEAM_NAME"]),
            "abbreviation": str(team_a["TEAM_ABBREVIATION"]),
            "display": team_a_display,
        },
        "team_b": {
            "name": str(team_b["TEAM_NAME"]),
            "abbreviation": str(team_b["TEAM_ABBREVIATION"]),
            "display": team_b_display,
        },
    }
    selected_abbr = str(team_a["TEAM_ABBREVIATION"])
    opponent_abbr = str(team_b["TEAM_ABBREVIATION"])
    selected_team_series_wins = int(team_a_series_wins)
    opponent_series_wins = int(team_b_series_wins)
    computed_series_score_diff = selected_team_series_wins - opponent_series_wins
    series_text, series_leader = series_status_text(
        selected_abbr,
        opponent_abbr,
        selected_team_series_wins,
        opponent_series_wins,
    )
    if prediction_context_mode == PREDICTION_MODE_CURRENT:
        series_text = "Series tied 0-0"
        series_leader = "Tied"
    seed_a = features.get("seed_A")
    seed_b = features.get("seed_B")
    seed_difference = features.get("seed_difference")
    higher_seed_a = features.get("higher_seed_A")

    def clean_optional_float(value):
        return None if value is None or pd.isna(value) else float(value)

    return {
        "season": season,
        "season_type": season_type,
        "prediction_date": prediction_date.isoformat(),
        "home_team": home_team,
        "prediction_context_mode": prediction_context_mode,
        "selected_team": selected_abbr,
        "opponent": opponent_abbr,
        "seed_A": clean_optional_float(seed_a),
        "seed_B": clean_optional_float(seed_b),
        "seed_difference": clean_optional_float(seed_difference),
        "higher_seed_A": clean_optional_float(higher_seed_a),
        "selected_team_series_wins": selected_team_series_wins,
        "opponent_series_wins": opponent_series_wins,
        "series_score_diff": computed_series_score_diff,
        "series_leader": series_leader,
        "series_status_text": series_text,
        "raw_live_game_series_label": raw_live_game_series_label,
        "user_series_context": {
            "game_number": int(game_number),
            "team_a_series_wins": selected_team_series_wins,
            "team_b_series_wins": opponent_series_wins,
            "selected_team_series_wins": selected_team_series_wins,
            "opponent_series_wins": opponent_series_wins,
            "series_score_diff": computed_series_score_diff,
            "series_leader": series_leader,
            "series_status_text": series_text,
            "elimination_game": int(selected_team_series_wins == 3 or opponent_series_wins == 3),
        },
        "team_a": {
            "label": team_a_label,
            "display": team_a_display,
            "name": str(team_a["TEAM_NAME"]),
            "abbreviation": str(team_a["TEAM_ABBREVIATION"]),
            "win_probability": float(team_a_probability),
            "series_win_probability": None
            if team_a_series_probability is None
            else float(team_a_series_probability),
        },
        "team_b": {
            "label": team_b_label,
            "display": team_b_display,
            "name": str(team_b["TEAM_NAME"]),
            "abbreviation": str(team_b["TEAM_ABBREVIATION"]),
            "win_probability": float(team_b_probability),
            "series_win_probability": None
            if team_b_series_probability is None
            else float(team_b_series_probability),
        },
        "favorite": {
            "side": favorite_key,
            "display": favorite_name,
            "win_probability": float(favorite_probability),
            "margin_pp": float(margin * 100),
        },
        "underdog": {"display": underdog_name},
        "confidence": confidence_label(margin),
        "series": {
            "team_a_series_win_probability": None
            if team_a_series_probability is None
            else float(team_a_series_probability),
            "team_b_series_win_probability": None
            if team_b_series_probability is None
            else float(team_b_series_probability),
            "favorite": None
            if team_a_series_probability is None
            else (
                {
                    "side": "team_a",
                    "display": team_a_display,
                    "win_probability": float(team_a_series_probability),
                }
                if team_a_series_probability >= 0.5
                else {
                    "side": "team_b",
                    "display": team_b_display,
                    "win_probability": float(1 - team_a_series_probability),
                }
            ),
        },
        "feature_values": clean_features,
        "top_factors": clean_chat_factors(top_factors, chat_team_context),
        "favorite_factors": clean_chat_factors(favorite_factors, chat_team_context),
        "underdog_factors": clean_chat_factors(underdog_factors, chat_team_context),
        "home_scenarios": home_debug_rows or [],
        "top_importances": importances.head(10).to_dict(orient="records"),
        "model_limitations": CHAT_MODEL_LIMITATIONS,
        "model_metrics": model_bundle.get("metrics", {}),
    }


def build_prediction_context(*args, **kwargs) -> dict:
    with perf_timer("chat context generation"):
        return _build_prediction_context_impl(*args, **kwargs)


def deterministic_initial_explanation(context: dict) -> str:
    favorite = context["favorite"]
    underdog = context["underdog"]
    favorite_side = "Team A" if favorite["side"] == "team_a" else "Team B"
    underdog_side = other_side(favorite_side)
    favorite_name = team_chat_name(context, favorite_side)
    underdog_name = team_chat_name(context, underdog_side)
    favorite_lines = factor_lines_for_side(context, favorite_side, limit=5)
    underdog_lines = factor_lines_for_side(context, underdog_side, limit=3)
    if not favorite_lines:
        favorite_lines = ["The model's strongest signals are balanced rather than concentrated in one obvious category."]
    if not underdog_lines:
        underdog_lines = ["The underdog does not have many top-ranked factors in this particular model read, but basketball outcomes can still swing on shooting variance, matchup execution, and late-game possessions."]

    game_read = (
        f"The model favors {favorite_name} to win this game at {favorite['win_probability']:.1%}. "
        f"That is a {favorite['margin_pp']:.1f}-point game-probability edge over {underdog_name}."
    )
    series_context = context.get("series") or {}
    series_favorite = series_context.get("favorite")
    if series_favorite:
        series_side = "Team A" if series_favorite.get("side") == "team_a" else "Team B"
        series_favorite_name = team_chat_name(context, series_side)
        game_read += (
            f" Separately, {series_favorite_name} is favored to win the series at "
            f"{float(series_favorite['win_probability']):.1%}."
        )

    sections = [
        "**Overall read**",
        game_read,
        "",
        "**Why the favorite is favored**",
        *[f"- {line}" for line in favorite_lines[:5]],
        "",
        "**What keeps the underdog competitive**",
        *[f"- {line}" for line in underdog_lines[:3]],
        "",
        "**Confidence/limitations**",
        f"Confidence is {context['confidence'].lower()}. This is a model-based read from team strength, form, home-court, efficiency, seeding, rest, and context features. The app does not include injuries, lineup news, trades, or other real-world updates outside the dataset.",
    ]
    fixed_series_sentence = series_context_sentence(context)
    if fixed_series_sentence:
        sections = [fixed_series_sentence, ""] + sections
    return "\n".join(sections)


def deterministic_initial_explanation_body(context: dict) -> str:
    explanation = deterministic_initial_explanation(context)
    fixed_series_sentence = series_context_sentence(context)
    if fixed_series_sentence and explanation.startswith(fixed_series_sentence):
        return explanation[len(fixed_series_sentence):].lstrip()
    return explanation


def compose_initial_explanation(context: dict, explanation_body: str) -> str:
    fixed_series_sentence = series_context_sentence(context)
    if fixed_series_sentence:
        return f"{fixed_series_sentence}\n\n{explanation_body.strip()}"
    return explanation_body.strip()


def selected_chat_provider(environ: dict[str, str] | None = None) -> str:
    values = environ if environ is not None else os.environ
    if values.get("GEMINI_API_KEY"):
        return CHAT_PROVIDER_GEMINI
    if values.get("OPENAI_API_KEY"):
        return CHAT_PROVIDER_OPENAI
    return CHAT_PROVIDER_FALLBACK


def active_chat_provider_label() -> str:
    return selected_chat_provider()


def gemini_model_name(environ: dict[str, str] | None = None) -> str:
    values = environ if environ is not None else os.environ
    return values.get("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "429",
            "quota",
            "resource_exhausted",
            "resource exhausted",
            "rate limit",
            "ratelimit",
            "too many requests",
        ]
    )


def record_chat_provider_debug(provider: str, error: Exception | str) -> dict[str, str]:
    row = {"provider": provider, "error": str(error)}
    try:
        rows = list(st.session_state.get("chat_provider_debug", []))
        rows.append(row)
        st.session_state["chat_provider_debug"] = rows
    except Exception:
        pass
    return row


def _chat_messages_text(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def call_gemini_for_chat(system_prompt: str, messages: list[dict[str, str]]) -> str:
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        prompt = f"{system_prompt}\n\nConversation:\n{_chat_messages_text(messages)}"
        response = client.models.generate_content(
            model=gemini_model_name(),
            contents=prompt,
        )
        return getattr(response, "text", "") or ""
    except Exception as exc:
        record_chat_provider_debug(CHAT_PROVIDER_GEMINI, exc)
        if is_gemini_quota_error(exc):
            return GEMINI_QUOTA_FALLBACK_NOTICE
        return f"LLM response unavailable, using deterministic fallback. Error: {exc}"


def call_openai_for_chat(system_prompt: str, messages: list[dict[str, str]]) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return ""

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        record_chat_provider_debug(CHAT_PROVIDER_OPENAI, exc)
        return f"LLM response unavailable, using deterministic fallback. Error: {exc}"


def call_llm_for_chat(system_prompt: str, messages: list[dict[str, str]]) -> str | None:
    provider = selected_chat_provider()
    if provider == CHAT_PROVIDER_GEMINI:
        return call_gemini_for_chat(system_prompt, messages)
    if provider == CHAT_PROVIDER_OPENAI:
        return call_openai_for_chat(system_prompt, messages)
    return None


def llm_response_or_fallback(llm_response: str | None, fallback: str) -> str:
    if llm_response is None:
        return fallback
    if llm_response == GEMINI_QUOTA_FALLBACK_NOTICE:
        return f"{GEMINI_QUOTA_FALLBACK_NOTICE}\n\n{fallback}"
    if llm_response.startswith("LLM response unavailable"):
        return f"{llm_response}\n\n{fallback}"
    return llm_response


def validated_llm_response_or_fallback(llm_response: str | None, fallback: str, context: dict) -> str:
    response = llm_response_or_fallback(llm_response, fallback)
    if text_contradicts_series_status(response, context):
        record_chat_provider_debug(
            "Series grounding",
            f"Discarded generated chat because it contradicted series_status_text={context.get('series_status_text')!r}: {response}",
        )
        return fallback
    return response


def context_system_prompt(context: dict) -> str:
    team_a_name = f"{context['team_a']['name']} ({context['team_a']['abbreviation']})"
    team_b_name = f"{context['team_b']['name']} ({context['team_b']['abbreviation']})"
    return (
        "You are a focused NBA model analyst. Use only the supplied JSON prediction context. "
        "Answer the user's exact question first in one direct sentence. "
        "Do not lead with repeated probabilities unless the user asks for the overall prediction, odds, probability, chances, confidence, or who is favored. "
        "Then add 2-4 short bullets only when they help the answer. "
        "Keep model evidence separate from limitations, and include one limitations sentence only when relevant. "
        "Do not invent injuries, player availability, player news, trades, lineups, betting odds, or external facts. "
        "If the context does not contain enough information, say that directly. "
        f"The returned team_a probability is for {team_a_name}; {team_b_name} probability is 1 minus that value. "
        "In user-visible text, use actual team names or abbreviations and never say Team A or Team B. "
        "Explain model factors in plain basketball language using friendly_label and short_basketball_meaning. "
        "Avoid vague filler such as 'stronger regular season performance' unless tied to a specific feature. "
        "Do not write raw feature values like 'value -3.200' or 'negative edge.' "
        "For feature direction, only use each factor's pushes_toward field and signed_contribution; never infer which team a feature helped from the raw feature value. "
        "For biggest-advantage questions, identify the highest-ranked top factor favoring the predicted team and explain it first. "
        "For underdog-win questions, identify top factors favoring the underdog plus uncertainty or limitations. "
        "For injury/player questions, say the model does not include injuries or player availability. "
        "For playoff series context, use series_status_text, selected_team_series_wins, opponent_series_wins, series_leader, and series win probabilities as the source of truth. "
        "Do not infer the series leader from home team, favorite, pushes_toward, or raw feature signs. "
        "series_score_diff is semantic context only: positive means the selected team leads the series, negative means the opponent leads, and zero means tied. "
        "Always distinguish game win probability from series win probability. "
        "Do not give generic suggestions for what to ask next. "
        "For strength-comparison questions, compare probability edge, important model factors, and confidence. "
        "For home-court questions, use home_scenarios when available. "
        "Explain concepts: offensive rating is points scored per 100 possessions; defensive rating is points allowed per 100 possessions and lower is better; net rating is scoring margin per 100 possessions; eFG% adjusts shooting efficiency for threes; TS% includes free throws; pace is possessions per game; Elo is a rolling team-strength rating; seed difference is playoff seeding advantage; home_win_pct_diff is home/away split advantage. "
        "Group the first explanation into: Overall read, Why the favorite is favored, What keeps the underdog competitive, and Confidence/limitations.\n\n"
        f"Prediction context JSON:\n{json.dumps(context, indent=2, default=str)}"
    )


def generate_initial_explanation(context: dict) -> str:
    fallback_body = deterministic_initial_explanation_body(context)
    llm_response = call_llm_for_chat(
        context_system_prompt(context),
        [
            {
                "role": "user",
                "content": (
                    "Write the first assistant message. Include predicted winner and probability, why they are favored, "
                    "top 3-5 model factors in readable basketball language, what helped the underdog, and confidence. "
                    "Use the required four-section structure and avoid raw feature-output wording."
                ),
            }
        ],
    )
    explanation_body = validated_llm_response_or_fallback(llm_response, fallback_body, context)
    return compose_initial_explanation(context, explanation_body)


def deterministic_chat_answer(context: dict, user_question: str) -> str:
    routed = routed_chat_answer(context, user_question)
    if routed:
        return routed

    question = user_question.lower()
    favorite = context["favorite"]
    favorite_side = "Team A" if favorite["side"] == "team_a" else "Team B"
    favorite_name = team_chat_name(context, favorite_side)
    if any(word in question for word in ["injury", "injuries", "hurt", "available", "lineup", "trade", "news"]):
        return (
            f"The app does not include current injuries, lineup news, trades, or roster/news context. "
            f"For this prediction, the available model context says {probability_read(context)} "
            f"{favorite_name} is favored with {context['confidence'].lower()} confidence based on the model features shown in the dashboard."
        )
    if any(phrase in question for phrase in ["how much stronger", "who is stronger", "stronger than", "better than", "compare", "gap between", "edge over"]):
        return strength_comparison_answer(context, user_question)
    if "home" in question:
        return (
            f"{home_scenario_summary(context)}\n"
            "Home-court impact here is based on home/away split advantage, not travel news or arena-specific context outside the dataset."
        )
    if any(phrase in question for phrase in ["underdog", "needs to", "path to win", "how can", "upset", "competitive"]):
        return underdog_answer(context)
    if any(word in question for word in ["factor", "why", "reason", "important"]):
        lines = factor_lines_for_side(context, favorite_side, limit=5)
        if not lines:
            lines = [factor_basketball_sentence(row, context) for row in context["top_factors"][:5]]
        return (
            f"{favorite_name} is favored at {favorite['win_probability']:.1%}. "
            f"The main basketball reasons are:\n" + "\n".join(f"- {line}" for line in lines[:5])
        )
    if "offensive" in question or "offense" in question:
        return (
            "Offensive rating estimates points scored per 100 possessions. "
            f"In this matchup, {probability_read(context)} A higher offensive-rating signal helps the team that has produced more efficiently after normalizing for pace."
        )
    if "defensive" in question or "defense" in question:
        return (
            "Defensive rating estimates points allowed per 100 possessions, and lower is better. "
            f"For this prediction, {favorite_name} is favored overall, but the defensive-rating factor may still support either side depending on which team allowed fewer points per 100 possessions."
        )
    if "net rating" in question:
        return (
            "Net rating is scoring margin per 100 possessions, combining offense and defense into one overall strength signal. "
            f"That matters here because {favorite_name} is favored at {favorite['win_probability']:.1%}, and net rating is one of the model's ways of measuring broad team quality."
        )
    if "efg" in question or "effective field" in question:
        return (
            "eFG% is shooting efficiency adjusted for the extra value of three-pointers. "
            "It helps the model separate ordinary shot-making from more valuable three-point-heavy efficiency in this matchup."
        )
    if "ts" in question or "true shooting" in question:
        return (
            "True shooting percentage is overall scoring efficiency, including twos, threes, and free throws. "
            f"It is part of the efficiency context behind the current {context['team_a']['abbreviation']} vs {context['team_b']['abbreviation']} prediction."
        )
    if "pace" in question:
        return (
            "Pace is possessions per game. It is not automatically good or bad; it tells the model about tempo and whether this matchup is closer to one team's normal style."
        )
    if "elo" in question:
        return (
            "Elo is a rolling team-strength rating that updates after games. In this app it is used as a pre-game signal, so it supports the matchup read without using results from after the prediction date."
        )
    if "seed" in question:
        return (
            "Seed difference is playoff seeding context. Lower NBA playoff seeds are better, so a 1 seed has a seeding advantage over an 8 seed. "
            f"In this matchup it is one of the context features behind the {favorite_name} prediction."
        )
    if any(word in question for word in ["confidence", "sure", "certain"]):
        return (
            f"Confidence is {context['confidence'].lower()}. "
            f"The model margin is {favorite['margin_pp']:.1f} percentage points, with {favorite_name} at {favorite['win_probability']:.1%}. "
            "This is not a guarantee; it only reflects the features available to the app."
        )
    top_lines = factor_lines_for_side(context, favorite_side, limit=3)
    return "\n".join(
        [
            f"The current model context points to {favorite_name}, with the answer driven by the matchup factors below.",
            f"- Probability context: {probability_read(context)}",
            f"- Model edge: {favorite_name} by {favorite['margin_pp']:.1f} percentage points with {context['confidence'].lower()} confidence.",
            "- Top model evidence:",
            *[f"- {line}" for line in top_lines[:3]],
        ]
    )


def answer_chat_question(context: dict, chat_history: list[dict[str, str]], user_question: str) -> str:
    routed = routed_chat_answer(context, user_question)
    if routed:
        return routed

    recent_history = chat_history[-8:]
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in recent_history
        if message.get("role") in {"user", "assistant"}
    ]
    messages.append({"role": "user", "content": user_question})
    llm_response = call_llm_for_chat(context_system_prompt(context), messages)
    fallback = deterministic_chat_answer(context, user_question)
    return validated_llm_response_or_fallback(llm_response, fallback, context)


def save_chat_transcript(context: dict, messages: list[dict[str, str]]) -> None:
    if not messages:
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "season": context["season"],
        "prediction_date": context["prediction_date"],
        "team_a": context["team_a"]["display"],
        "team_b": context["team_b"]["display"],
        "home_team": context["home_team"],
        "team_a_probability": context["team_a"]["win_probability"],
        "team_b_probability": context["team_b"]["win_probability"],
        "transcript": json.dumps(messages),
    }
    output = pd.DataFrame([row])
    if PREDICTION_CHATS_PATH.exists():
        existing = pd.read_csv(PREDICTION_CHATS_PATH)
        output = pd.concat([existing, output], ignore_index=True)
    output.to_csv(PREDICTION_CHATS_PATH, index=False)


def render_chat_panel(context: dict) -> None:
    message_count = len(st.session_state.get("prediction_chat_messages", []))
    bubble_html = [
        f"""
        <div class="chat-panel">
            <div class="chat-panel-title">Prediction Chat</div>
            <div class="chat-panel-note">Ask follow-ups about this matchup using only the current model context.</div>
            <div id="prediction-chat-history" class="chat-history" data-chat-message-count="{message_count}">
                <div class="chat-stream">
        """
    ]
    for message in st.session_state.get("prediction_chat_messages", []):
        role = "user" if message["role"] == "user" else "assistant"
        content = markdown_to_basic_html(str(message["content"]))
        bubble_html.append(
            f'<div class="chat-row {role}"><div class="chat-bubble {role}">{content}</div></div>'
        )
    bubble_html.append('<div id="prediction-chat-bottom-anchor"></div></div></div></div>')
    st.markdown("".join(bubble_html), unsafe_allow_html=True)
    chat_scroll_script = """
        <script>
            (() => {
                const expectedMessageCount = __MESSAGE_COUNT__;
                const latestVisibleChatHistory = () => {
                    const doc = window.parent.document;
                    const histories = Array.from(
                        doc.querySelectorAll('#prediction-chat-history, .chat-history[data-chat-message-count]')
                    );
                    const visibleHistories = histories.filter((history) => {
                        const rect = history.getBoundingClientRect();
                        return rect.width > 0 && rect.height > 0;
                    });
                    return visibleHistories[visibleHistories.length - 1] || histories[histories.length - 1] || null;
                };
                const scrollChatHistory = () => {
                    const history = latestVisibleChatHistory();
                    if (!history) {
                        return;
                    }
                    if (history.dataset.chatMessageCount !== String(expectedMessageCount)) {
                        return;
                    }
                    const maxScrollTop = history.scrollHeight - history.clientHeight;
                    if (maxScrollTop > 0) {
                        history.scrollTop = maxScrollTop;
                        history.scrollTo({ top: maxScrollTop, behavior: 'auto' });
                    }
                };
                window.requestAnimationFrame(scrollChatHistory);
                [40, 120, 260, 520, 900, 1400, 2000].forEach((delay) => {
                    window.setTimeout(scrollChatHistory, delay);
                });
                const history = latestVisibleChatHistory();
                if (history) {
                    const observer = new MutationObserver(scrollChatHistory);
                    observer.observe(history, { childList: true, subtree: true });
                    window.setTimeout(() => observer.disconnect(), 2200);
                }
            })();
        </script>
        """.replace("__MESSAGE_COUNT__", str(message_count))
    components.html(
        chat_scroll_script,
        height=0,
    )

    st.markdown('<div class="chat-input-card">', unsafe_allow_html=True)
    with st.form("prediction_chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([1, 0.30], gap="small", vertical_alignment="center")
        with input_col:
            user_question = st.text_input(
                "Prediction chat question",
                placeholder="Ask about this prediction",
                label_visibility="collapsed",
            )
        with send_col:
            submitted = st.form_submit_button("Send")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and user_question.strip():
        st.session_state["prediction_chat_messages"].append({"role": "user", "content": user_question})
        answer = answer_chat_question(context, st.session_state["prediction_chat_messages"], user_question)
        st.session_state["prediction_chat_messages"].append({"role": "assistant", "content": answer})
        save_chat_transcript(context, st.session_state["prediction_chat_messages"])
        st.rerun()


def render_prediction_summary(team_a_display: str, team_b_display: str, team_a_probability: float) -> None:
    team_b_probability = 1 - team_a_probability
    if team_a_probability >= team_b_probability:
        favorite = team_a_display
        margin = team_a_probability - team_b_probability
    else:
        favorite = team_b_display
        margin = team_b_probability - team_a_probability

    confidence = confidence_label(margin)
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-main">{favorite} favored by {margin * 100:.1f} percentage points</div>
            <span class="confidence-pill">Confidence: {confidence}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_prediction_explanation(
    season: str,
    prediction_date: date,
    team_a_label: str,
    team_b_label: str,
    home_team: str,
    team_a_probability: float,
    features: dict[str, float],
    factors: pd.DataFrame,
    importances: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "season": season,
        "prediction_date": prediction_date.isoformat(),
        "team_a": team_a_label,
        "team_b": team_b_label,
        "home_team": home_team,
        "team_a_probability": team_a_probability,
        "team_b_probability": 1 - team_a_probability,
        "feature_values": json.dumps({key: None if pd.isna(value) else float(value) for key, value in features.items()}),
        "top_factors": filter_noninformative_factors(factors).head(10).to_json(orient="records"),
        "top_importances": importances.head(10).to_json(orient="records"),
    }
    output = pd.DataFrame([row])
    if PREDICTION_EXPLANATIONS_PATH.exists():
        existing = pd.read_csv(PREDICTION_EXPLANATIONS_PATH)
        output = pd.concat([existing, output], ignore_index=True)
    output.to_csv(PREDICTION_EXPLANATIONS_PATH, index=False)


def main() -> None:
    st.set_page_config(page_title="NBA Playoff Predictor", layout="wide")
    inject_dashboard_css()
    render_header()

    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.exists():
        st.error("Train the model first: python -m src.predictor train --start-season 2015-16 --end-season 2023-24")
        st.stop()

    with st.sidebar:
        st.markdown("## Matchup Setup")
        if st.session_state.get("season") not in SEASON_OPTIONS:
            st.session_state["season"] = DEFAULT_SEASON
        if st.session_state.get("season_type") not in ["Regular Season", "Playoffs"]:
            st.session_state["season_type"] = DEFAULT_SEASON_TYPE
        st.session_state.setdefault("prediction_date", date.today())
        season = st.selectbox("Season", SEASON_OPTIONS, key="season")
        season_type = st.selectbox("Stats source", ["Regular Season", "Playoffs"], key="season_type")
        prediction_date = st.date_input("Prediction date", key="prediction_date")
        st.markdown('<div class="sidebar-compact-separator"></div>', unsafe_allow_html=True)

    model_bundle = cached_model(str(model_path), model_artifact_version(model_path))
    saved_feature_columns = all_saved_feature_columns(model_bundle) or model_bundle.get("feature_columns", DIFF_COLUMNS)
    unknown_model_features = [column for column in saved_feature_columns if column not in FEATURE_COLUMNS]
    if unknown_model_features:
        st.error("Retrain the model before using the current feature schema.")
        st.code("python -m src.predictor train --start-season 2015-16 --end-season 2024-25")
        st.stop()

    try:
        team_stats = cached_team_stats(season, season_type)
    except Exception as exc:
        st.error(f"Could not load NBA stats: {exc}")
        st.stop()

    team_stats = team_stats.sort_values("TEAM_NAME").reset_index(drop=True)
    team_options = {
        f"{row.TEAM_NAME} ({row.TEAM_ABBREVIATION})": int(row.TEAM_ID)
        for row in team_stats.itertuples(index=False)
    }
    labels = list(team_options)

    with st.sidebar:
        default_a_label = "Boston Celtics (BOS)" if "Boston Celtics (BOS)" in labels else labels[0]
        if st.session_state.get("selected_team_label") not in labels:
            st.session_state["selected_team_label"] = default_a_label
        team_a_label = st.selectbox("Selected team", labels, key="selected_team_label")
        default_b_index = labels.index("New York Knicks (NYK)") if "New York Knicks (NYK)" in labels else min(1, len(labels) - 1)
        if st.session_state.get("opponent_team_label") not in labels or st.session_state.get("opponent_team_label") == team_a_label:
            st.session_state["opponent_team_label"] = labels[default_b_index]
        team_b_label = st.selectbox("Opponent", labels, key="opponent_team_label")
        if st.session_state.get("home_team_label") not in [team_a_label, team_b_label]:
            st.session_state["home_team_label"] = team_a_label
        home_team = st.radio("Home Team", [team_a_label, team_b_label], key="home_team_label")
        if st.session_state.get("prediction_context_mode") not in [PREDICTION_MODE_CURRENT, PREDICTION_MODE_PLAYOFF]:
            st.session_state["prediction_context_mode"] = PREDICTION_MODE_CURRENT
        prediction_context_mode = st.radio("Prediction Mode", [PREDICTION_MODE_CURRENT, PREDICTION_MODE_PLAYOFF], key="prediction_context_mode")
        game_number = 1
        team_a_series_wins = 0
        team_b_series_wins = 0
        if prediction_context_mode == PREDICTION_MODE_PLAYOFF:
            st.markdown(
                '<div class="sidebar-context-note">User-provided hypothetical playoff series context.</div>',
                unsafe_allow_html=True,
            )
            game_label_col, game_select_col = st.columns([1.45, 0.7], gap="small")
            with game_label_col:
                st.markdown('<div class="sidebar-inline-label">Game Number:</div>', unsafe_allow_html=True)
            with game_select_col:
                if st.session_state.get("game_number") not in list(range(1, 8)):
                    st.session_state["game_number"] = 1
                game_number = st.selectbox("Game number", list(range(1, 8)), label_visibility="collapsed", key="game_number")
            team_a_control_abbr = _abbr_from_label(team_a_label)
            team_b_control_abbr = _abbr_from_label(team_b_label)
            expected_series_wins = int(game_number) - 1
            team_a_win_options = valid_team_a_series_win_options(int(game_number))
            if st.session_state.get("team_a_series_wins") not in team_a_win_options:
                st.session_state["team_a_series_wins"] = team_a_win_options[0]
            team_a_series_wins = st.selectbox(
                "Series Score",
                team_a_win_options,
                key="team_a_series_wins",
                format_func=lambda wins: f"{team_a_control_abbr} {wins} - {expected_series_wins - int(wins)} {team_b_control_abbr}",
            )
            team_b_series_wins = expected_series_wins - int(team_a_series_wins)
        predict_clicked = st.button("Predict", type="primary")

    if team_a_label == team_b_label:
        st.warning("Pick two different teams.")
        st.stop()

    stats_by_id = team_stats.set_index("TEAM_ID")
    team_a = stats_by_id.loc[team_options[team_a_label]]
    team_b = stats_by_id.loc[team_options[team_b_label]]
    team_a_colors, team_b_colors = inject_team_css(
        str(team_a["TEAM_ABBREVIATION"]),
        str(team_b["TEAM_ABBREVIATION"]),
    )
    active_model_entry = get_model_entry_for_mode(model_bundle, prediction_context_mode)
    feature_columns = active_model_entry.get("feature_columns", model_bundle.get("feature_columns", DIFF_COLUMNS))
    home_team_arg = "team1" if home_team == team_a_label else "team2"
    selection_key = {
        "season": season,
        "season_type": season_type,
        "team_a": team_a_label,
        "team_b": team_b_label,
        "home_team": home_team,
        "prediction_date": prediction_date.isoformat(),
        "prediction_context_mode": prediction_context_mode,
        "game_number": game_number,
        "team_a_series_wins": team_a_series_wins,
        "team_b_series_wins": team_b_series_wins,
    }
    if st.session_state.get("active_selection_key") != selection_key:
        st.session_state["active_selection_key"] = selection_key
        st.session_state.pop("prediction_payload", None)
        st.session_state.pop("prediction_context", None)
        st.session_state.pop("prediction_chat_messages", None)

    render_live_games_topbar_safe(model_available=model_path.exists(), labels=labels)
    render_matchup_strip(team_a, team_b, home_team, team_a_label, team_b_label, team_a_colors, team_b_colors)

    if predict_clicked:
        st.session_state.pop("prediction_payload", None)
        st.session_state.pop("prediction_context", None)
        st.session_state["prediction_chat_messages"] = []
        st.session_state["chat_provider_debug"] = []
        if prediction_context_mode == PREDICTION_MODE_PLAYOFF:
            try:
                validate_series_score(int(game_number), int(team_a_series_wins), int(team_b_series_wins))
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

    if not predict_clicked and "prediction_payload" not in st.session_state:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Ready To Predict</div>
                <div class="panel-note">Choose a matchup in the sidebar, then run the model to see probabilities and explainability.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    if predict_clicked:
        try:
            alternate_home_team_arg = "team2" if home_team_arg == "team1" else "team1"
            prediction_result = compute_matchup_prediction(
                team_a=str(team_a["TEAM_ABBREVIATION"]),
                team_b=str(team_b["TEAM_ABBREVIATION"]),
                season=season,
                prediction_date=prediction_date.isoformat(),
                home_team=home_team_arg,
                feature_season_type=season_type,
                prediction_context_mode=prediction_context_mode,
                game_number=int(game_number),
                team_a_series_wins=int(team_a_series_wins),
                team_b_series_wins=int(team_b_series_wins),
            )
            team_a_probability = prediction_result["team_a_probability"]
            team_a_series_probability = prediction_result["team_a_series_probability"]
            features = prediction_result["features"]
            game_features = prediction_result["game_features"]
        except Exception as exc:
            st.error(f"Could not run prediction: {exc}")
            st.stop()

        home_debug_rows = [
            {
                "selection": f"{team_a['TEAM_ABBREVIATION']} home"
                if home_team_arg == "team1"
                else f"{team_b['TEAM_ABBREVIATION']} home",
                "team_A_probability": team_a_probability,
                "p_direct": prediction_result.get("p_direct"),
                "p_reverse_complement": prediction_result.get("p_reverse_complement"),
                "p_symmetric_final": prediction_result.get("p_symmetric_final"),
                "home_team_A": features.get("home_team_A"),
                "home_win_pct_diff": features.get("home_win_pct_diff"),
                "away_win_pct_diff": features.get("away_win_pct_diff"),
                "home_advantage_diff": features.get("home_advantage_diff"),
                "team_A_home_advantage": features.get("team_A_home_advantage"),
                "team_B_home_advantage": features.get("team_B_home_advantage"),
            }
        ]
        try:
            alternate_prediction_result = compute_matchup_prediction(
                team_a=str(team_a["TEAM_ABBREVIATION"]),
                team_b=str(team_b["TEAM_ABBREVIATION"]),
                season=season,
                prediction_date=prediction_date.isoformat(),
                home_team=alternate_home_team_arg,
                feature_season_type=season_type,
                prediction_context_mode=prediction_context_mode,
                game_number=int(game_number),
                team_a_series_wins=int(team_a_series_wins),
                team_b_series_wins=int(team_b_series_wins),
            )
            alternate_features = alternate_prediction_result["features"]
            alternate_probability = alternate_prediction_result["team_a_probability"]
            home_debug_rows.append(
                {
                    "selection": f"{team_a['TEAM_ABBREVIATION']} home"
                    if alternate_home_team_arg == "team1"
                    else f"{team_b['TEAM_ABBREVIATION']} home",
                    "team_A_probability": alternate_probability,
                    "p_direct": alternate_prediction_result.get("p_direct"),
                    "p_reverse_complement": alternate_prediction_result.get("p_reverse_complement"),
                    "p_symmetric_final": alternate_prediction_result.get("p_symmetric_final"),
                    "home_team_A": alternate_features.get("home_team_A"),
                    "home_win_pct_diff": alternate_features.get("home_win_pct_diff"),
                    "away_win_pct_diff": alternate_features.get("away_win_pct_diff"),
                    "home_advantage_diff": alternate_features.get("home_advantage_diff"),
                    "team_A_home_advantage": alternate_features.get("team_A_home_advantage"),
                    "team_B_home_advantage": alternate_features.get("team_B_home_advantage"),
                }
            )
        except Exception as exc:
            home_debug_rows.append(
                {
                    "selection": "Alternate home unavailable",
                    "team_A_probability": None,
                    "home_team_A": None,
                    "home_win_pct_diff": None,
                    "away_win_pct_diff": None,
                    "home_advantage_diff": None,
                    "team_A_home_advantage": None,
                    "team_B_home_advantage": None,
                    "error": str(exc),
                }
            )

        feature_frame = pd.DataFrame([game_features], columns=feature_columns)
        print("Final feature row before prediction:")
        print(
            json.dumps(
                {column: None if pd.isna(game_features.get(column)) else float(game_features.get(column)) for column in feature_columns},
                indent=2,
                sort_keys=True,
            )
        )
        importances = load_feature_importances(active_model_entry, feature_columns, prediction_context_mode)
        factors = local_factor_table(
            game_features,
            importances,
            feature_columns,
            pipeline=active_model_entry["pipeline"],
            full_probability=team_a_probability,
        )
        explanation_factors, explanation_importances = filter_explanation_features_for_mode(
            factors,
            importances,
            prediction_context_mode,
        )

        team_a_display = f"{team_a['TEAM_ABBREVIATION']}{' (Home)' if home_team == team_a_label else ''}"
        team_b_display = f"{team_b['TEAM_ABBREVIATION']}{' (Home)' if home_team == team_b_label else ''}"
        save_prediction_explanation(
            season=season,
            prediction_date=prediction_date,
            team_a_label=team_a_display,
            team_b_label=team_b_display,
            home_team=home_team,
            team_a_probability=team_a_probability,
            features={column: game_features.get(column) for column in feature_columns},
            factors=explanation_factors,
            importances=explanation_importances,
        )

        context = build_prediction_context(
            season=season,
            season_type=season_type,
            prediction_date=prediction_date,
            home_team=home_team,
            team_a=team_a,
            team_b=team_b,
            team_a_label=team_a_label,
            team_b_label=team_b_label,
            team_a_display=team_a_display,
            team_b_display=team_b_display,
            team_a_probability=team_a_probability,
            team_a_series_probability=team_a_series_probability,
            features=game_features,
            factors=explanation_factors,
            importances=explanation_importances,
            model_bundle=active_model_entry,
            feature_columns=feature_columns,
            home_debug_rows=home_debug_rows,
            prediction_context_mode=prediction_context_mode,
            game_number=int(game_number),
            team_a_series_wins=int(team_a_series_wins),
            team_b_series_wins=int(team_b_series_wins),
            raw_live_game_series_label=str(st.session_state.get("raw_live_game_series_label", "")),
        )
        initial_message = generate_initial_explanation(context)
        st.session_state["prediction_chat_messages"] = [{"role": "assistant", "content": initial_message}]
        save_chat_transcript(context, st.session_state["prediction_chat_messages"])

        st.session_state["prediction_context"] = context
        st.session_state["prediction_payload"] = {
            "team_a_probability": team_a_probability,
            "team_a_series_probability": team_a_series_probability,
            "features": features,
            "game_features": game_features,
            "feature_frame": feature_frame,
            "factors": factors,
            "importances": importances,
            "explanation_factors": explanation_factors,
            "explanation_importances": explanation_importances,
            "team_a_display": team_a_display,
            "team_b_display": team_b_display,
            "home_debug_rows": home_debug_rows,
        }

    payload = st.session_state["prediction_payload"]
    context = st.session_state["prediction_context"]
    team_a_probability = payload["team_a_probability"]
    team_a_series_probability = payload.get("team_a_series_probability")
    features = payload["features"]
    game_features = payload.get("game_features", features)
    feature_frame = payload["feature_frame"]
    factors = payload["factors"]
    importances = payload["importances"]
    explanation_factors = payload.get("explanation_factors", factors)
    explanation_importances = payload.get("explanation_importances", importances)
    team_a_display = payload["team_a_display"]
    team_b_display = payload["team_b_display"]
    home_debug_rows = payload["home_debug_rows"]
    team_a_abbreviation = str(team_a["TEAM_ABBREVIATION"])
    team_b_abbreviation = str(team_b["TEAM_ABBREVIATION"])

    result_col, chat_col = st.columns([1.65, 1], gap="large")
    with result_col:
        render_probability_summary(team_a_display, team_b_display, team_a_probability, home_team)
        if prediction_context_mode == PREDICTION_MODE_PLAYOFF and team_a_series_probability is not None:
            render_series_probability_summary(team_a_display, team_b_display, team_a_series_probability)
        render_prediction_summary(team_a_display, team_b_display, team_a_probability)

        st.markdown('<div class="panel-title">Why this prediction?</div>', unsafe_allow_html=True)
        factor_view = display_factor_table(explanation_factors, team_a_abbreviation, team_b_abbreviation)
        st.dataframe(factor_view, width="stretch", hide_index=True)

        with st.expander("Advanced Details", expanded=False):
            chart_col, factor_col = st.columns(2)
            with chart_col:
                importance_fig = feature_importance_chart(explanation_importances)
                st.pyplot(importance_fig, width="stretch")
                plt.close(importance_fig)
            with factor_col:
                factor_fig = factor_chart(
                    explanation_factors,
                    team_a_display,
                    team_b_display,
                    team_a_colors["primary"],
                    team_b_colors["primary"],
                )
                st.pyplot(factor_fig, width="stretch")
                plt.close(factor_fig)

            st.markdown('<div class="panel-title">Feature Values Used</div>', unsafe_allow_html=True)
            st.dataframe(
                display_feature_frame(feature_frame, features, team_a_abbreviation, team_b_abbreviation),
                width="stretch",
                hide_index=True,
            )

            with st.expander("Developer Debug", expanded=False):
                detail_a, detail_b, detail_c = st.columns(3)
                detail_a.metric("Selected features", len(feature_columns))
                detail_b.metric("Missing values", f"{features.get('MISSING_FEATURE_COUNT', 0):.0f}")
                detail_c.metric("Model", str(active_model_entry.get("metrics", {}).get("model", "Saved model")))
                st.caption(f"Active chat provider: {active_chat_provider_label()}")
                chat_debug_rows = st.session_state.get("chat_provider_debug") or []
                if chat_debug_rows:
                    st.markdown('<div class="panel-title">Chat Provider Debug</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(chat_debug_rows), width="stretch", hide_index=True)

                st.markdown('<div class="panel-title">Series Context Debug</div>', unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "selected_team": context.get("selected_team"),
                                "opponent": context.get("opponent"),
                                "home_team": _abbr_from_label(str(context.get("home_team", ""))),
                                "seed_A": context.get("seed_A"),
                                "seed_B": context.get("seed_B"),
                                "seed_difference": context.get("seed_difference"),
                                "higher_seed_A": context.get("higher_seed_A"),
                                "selected_team_series_wins": context.get("selected_team_series_wins"),
                                "opponent_series_wins": context.get("opponent_series_wins"),
                                "series_score_diff": context.get("series_score_diff"),
                                "series_status_text": context.get("series_status_text"),
                                "selected_team_series_win_probability": (
                                    context.get("series", {}).get("team_a_series_win_probability")
                                ),
                                "raw_live_game_series_label": context.get("raw_live_game_series_label", ""),
                            }
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.markdown('<div class="panel-title">Home-Court Debug</div>', unsafe_allow_html=True)
                st.dataframe(
                    display_home_debug_table(home_debug_rows, team_a_abbreviation, team_b_abbreviation),
                    width="stretch",
                    hide_index=True,
                )
                live_debug_rows = st.session_state.get("live_game_prediction_debug") or []
                if live_debug_rows:
                    st.markdown('<div class="panel-title">Live Games Prediction Debug</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(live_debug_rows), width="stretch", hide_index=True)

        st.markdown(
            f'<div class="saved-note">Saved prediction explanation to {PREDICTION_EXPLANATIONS_PATH}</div>',
            unsafe_allow_html=True,
        )

    with chat_col:
        render_chat_panel(context)


if __name__ == "__main__":
    main()
