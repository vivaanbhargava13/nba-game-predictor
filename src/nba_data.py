from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, leaguedashplayerstats, leaguedashteamstats, leaguestandings, teamgamelog
from nba_api.stats.static import teams


API_MAX_RETRIES = 3
API_BASE_DELAY_SECONDS = 0.75


TEAM_STRENGTH_COLUMNS = [
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "W_PCT",
    "PLUS_MINUS",
    "PACE",
]


TEAM_ID_COLUMNS = ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]
TIER_1_FEATURES = [f"{column}_DIFF" for column in TEAM_STRENGTH_COLUMNS]
TIER_2_FEATURES = [
    "last_5_win_pct_diff",
    "last_10_win_pct_diff",
    "last_5_net_rating_diff",
    "last_10_net_rating_diff",
    "last_5_point_diff_diff",
    "last_10_point_diff_diff",
    "weighted_recent_win_pct_diff",
    "weighted_recent_net_rating_diff",
    "weighted_recent_ts_pct_diff",
    "weighted_recent_def_rating_diff",
]
TIER_3_FEATURES = [
    "top_1_ppg_diff",
    "top_1_mpg_diff",
    "top_1_ts_pct_diff",
    "top_3_ppg_diff",
    "top_3_mpg_diff",
    "top_3_ts_pct_diff",
    "top_5_ppg_diff",
    "top_5_mpg_diff",
]
TIER_4_FEATURES = [
    "home_team_A",
    "home_win_pct_diff",
    "away_win_pct_diff",
    "team_A_home_advantage",
    "team_B_home_advantage",
    "home_advantage_diff",
    "clipped_home_win_pct_diff",
    "clipped_away_win_pct_diff",
]
TIER_5_FEATURES = ["rest_days_diff", "team_A_rest_days", "team_B_rest_days"]
TIER_6_FEATURES = ["elo_diff"]
TIER_7_FEATURES = [
    "efg_pct_diff",
    "ts_pct_diff",
    "turnover_pct_diff",
    "reb_pct_diff",
    "assist_turnover_ratio_diff",
    "ft_rate_diff",
]
TIER_8_FEATURES = [
    "season_h2h_win_pct_diff",
    "season_h2h_margin_diff",
    "h2h_win_pct_diff",
    "h2h_margin_diff",
    "h2h_net_rating_diff",
    "h2h_efg_pct_diff",
    "h2h_ts_pct_diff",
    "h2h_turnover_pct_diff",
    "h2h_reb_pct_diff",
]
TIER_9_FEATURES = ["seed_difference", "higher_seed_A", "game_number", "elimination_game", "series_score_diff"]
TIER_10_FEATURES = [
    "three_point_offense_vs_defense_diff",
    "paint_scoring_vs_paint_defense_diff",
    "offensive_rebound_vs_defensive_rebound_diff",
    "turnover_creation_vs_turnover_rate_diff",
    "ft_rate_vs_foul_rate_diff",
]
MODEL_FEATURE_COLUMNS = (
    TIER_1_FEATURES
    + TIER_2_FEATURES
    + TIER_3_FEATURES
    + TIER_4_FEATURES
    + TIER_5_FEATURES
    + TIER_6_FEATURES
    + TIER_7_FEATURES
    + TIER_8_FEATURES
    + TIER_9_FEATURES
    + TIER_10_FEATURES
)
FEATURE_SCHEMA_VERSION = "2026-05-24-feature-v5-home-context-models"

# Human-readable notes for the expanded model. The code below follows these
# definitions so every training and prediction path uses the same feature math.
FEATURE_DESCRIPTIONS = {
    # Tier 1: season-level team strength from LeagueDashTeamStats.
    "OFF_RATING_DIFF": "Team A offensive rating minus Team B offensive rating.",
    "DEF_RATING_DIFF": "Team B defensive rating minus Team A defensive rating; positive means Team A allowed fewer points per 100 possessions.",
    "NET_RATING_DIFF": "Team A net rating minus Team B net rating.",
    "W_PCT_DIFF": "Team A win percentage minus Team B win percentage.",
    "PLUS_MINUS_DIFF": "Team A point differential minus Team B point differential.",
    "PACE_DIFF": "Team A pace minus Team B pace.",
    # Tier 2: recent form from TeamGameLog games strictly before prediction date.
    "last_5_win_pct_diff": "Team A last 5 win percentage minus Team B last 5 win percentage.",
    "last_10_win_pct_diff": "Team A last 10 win percentage minus Team B last 10 win percentage.",
    "last_5_net_rating_diff": "Team A last 5 estimated net rating minus Team B last 5 estimated net rating.",
    "last_10_net_rating_diff": "Team A last 10 estimated net rating minus Team B last 10 estimated net rating.",
    "last_5_point_diff_diff": "Team A last 5 average point differential minus Team B last 5 average point differential.",
    "last_10_point_diff_diff": "Team A last 10 average point differential minus Team B last 10 average point differential.",
    "weighted_recent_win_pct_diff": "Team A weighted recent win percentage minus Team B weighted recent win percentage.",
    "weighted_recent_net_rating_diff": "Team A weighted recent net rating minus Team B weighted recent net rating.",
    "weighted_recent_ts_pct_diff": "Team A weighted recent true shooting percentage minus Team B weighted recent true shooting percentage.",
    "weighted_recent_def_rating_diff": "Team B weighted recent defensive rating minus Team A weighted recent defensive rating; positive favors Team A.",
    # Tier 3: star power from regular-season player averages before prediction date.
    "top_1_ppg_diff": "Team A top scorer PPG minus Team B top scorer PPG.",
    "top_1_mpg_diff": "Team A top scorer MPG minus Team B top scorer MPG.",
    "top_1_ts_pct_diff": "Team A top scorer TS% minus Team B top scorer TS%.",
    "top_3_ppg_diff": "Team A top 3 combined player PPG minus Team B top 3 combined player PPG.",
    "top_3_mpg_diff": "Team A top 3 combined player MPG minus Team B top 3 combined player MPG.",
    "top_3_ts_pct_diff": "Team A top 3 combined TS% minus Team B top 3 combined TS%.",
    "top_5_ppg_diff": "Team A top 5 combined player PPG minus Team B top 5 combined player PPG.",
    "top_5_mpg_diff": "Team A top 5 combined player MPG minus Team B top 5 combined player MPG.",
    # Tier 4: home-court context from matchup location and prior home/away records.
    "home_team_A": "1 when Team A is the home team, otherwise 0.",
    "home_win_pct_diff": "Legacy Team A current-location win percentage minus Team B current-location win percentage.",
    "away_win_pct_diff": "Legacy Team A opposite-location win percentage minus Team B opposite-location win percentage.",
    "team_A_home_advantage": "Team A home win percentage minus Team A overall win percentage before prediction date.",
    "team_B_home_advantage": "Team B home win percentage minus Team B overall win percentage before prediction date.",
    "home_advantage_diff": "Selected home team's own home edge, oriented toward Team A probability and clipped to a modest range.",
    "clipped_home_win_pct_diff": "Legacy current-location split feature clipped to a modest range for home-feature ablation.",
    "clipped_away_win_pct_diff": "Legacy opposite-location split feature clipped to a modest range for home-feature ablation.",
    # Tier 5: rest/fatigue from each team's previous game before prediction date.
    "rest_days_diff": "Team A rest days minus Team B rest days.",
    "team_A_rest_days": "Days since Team A's previous game.",
    "team_B_rest_days": "Days since Team B's previous game.",
    "elo_diff": "Team A pre-game Elo minus Team B pre-game Elo.",
    "efg_pct_diff": "Team A prior effective field goal percentage minus Team B prior effective field goal percentage.",
    "ts_pct_diff": "Team A prior true shooting percentage minus Team B prior true shooting percentage.",
    "turnover_pct_diff": "Team A prior turnover percentage minus Team B prior turnover percentage.",
    "reb_pct_diff": "Team A prior rebound share minus Team B prior rebound share.",
    "assist_turnover_ratio_diff": "Team A prior assist/turnover ratio minus Team B prior assist/turnover ratio.",
    "ft_rate_diff": "Team A prior free throw rate minus Team B prior free throw rate.",
    "season_h2h_win_pct_diff": "Team A prior head-to-head win percentage minus Team B prior head-to-head win percentage.",
    "season_h2h_margin_diff": "Team A prior head-to-head average margin minus Team B prior head-to-head average margin.",
    "h2h_win_pct_diff": "Team A prior head-to-head win percentage minus Team B prior head-to-head win percentage.",
    "h2h_margin_diff": "Team A prior head-to-head scoring margin minus Team B prior head-to-head scoring margin.",
    "h2h_net_rating_diff": "Team A prior head-to-head net rating minus Team B prior head-to-head net rating.",
    "h2h_efg_pct_diff": "Team A prior head-to-head eFG% minus Team B prior head-to-head eFG%.",
    "h2h_ts_pct_diff": "Team A prior head-to-head TS% minus Team B prior head-to-head TS%.",
    "h2h_turnover_pct_diff": "Team B prior head-to-head turnover percentage minus Team A prior head-to-head turnover percentage; positive favors Team A.",
    "h2h_reb_pct_diff": "Team A prior head-to-head rebound share minus Team B prior head-to-head rebound share.",
    "seed_difference": "Team B playoff seed minus Team A playoff seed; positive means Team A has the better seed.",
    "higher_seed_A": "1 when Team A has the better playoff seed, otherwise 0.",
    "game_number": "Series game number before the current game is played.",
    "elimination_game": "1 when either team can be eliminated before this game, otherwise 0.",
    "series_score_diff": "Team A series wins minus Team B series wins before this game.",
    "three_point_offense_vs_defense_diff": "Team A three-point attack versus Team B three-point defense minus the reverse matchup.",
    "paint_scoring_vs_paint_defense_diff": "Team A two-point scoring proxy versus Team B two-point defense proxy minus the reverse matchup.",
    "offensive_rebound_vs_defensive_rebound_diff": "Team A offensive rebounding matchup minus Team B offensive rebounding matchup.",
    "turnover_creation_vs_turnover_rate_diff": "Team A turnover creation matchup minus Team B turnover creation matchup.",
    "ft_rate_vs_foul_rate_diff": "Team A free throw rate matchup minus Team B free throw rate matchup.",
}

# Canonical feature list used by model training and prediction.
FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS


def season_range(start_season: str, end_season: str) -> list[str]:
    """Return NBA season labels from start to end, inclusive."""
    start_year = int(start_season[:4])
    end_year = int(end_season[:4])
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def _cache_path(cache_dir: Path, name: str, season: str, season_type: str) -> Path:
    safe_type = season_type.lower().replace(" ", "_")
    return cache_dir / f"{name}_{season}_{safe_type}.csv"


def _safe_date_label(value: pd.Timestamp | str) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def _nba_date(value: pd.Timestamp | str) -> str:
    return pd.to_datetime(value).strftime("%m/%d/%Y")


def _read_or_fetch(path: Path, fetcher, *, allow_failure: bool = False) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            df = fetcher()
            break
        except Exception as exc:
            last_error = exc
            if attempt == API_MAX_RETRIES:
                if allow_failure:
                    print(f"NBA API request failed for {path.name}; using empty fallback. Error: {exc}")
                    return pd.DataFrame()
                raise

            delay = API_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(f"NBA API request failed for {path.name}; retry {attempt}/{API_MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay)
    else:
        if allow_failure:
            print(f"NBA API request failed for {path.name}; using empty fallback. Error: {last_error}")
            return pd.DataFrame()
        raise RuntimeError(f"NBA API request failed for {path}")

    df.to_csv(path, index=False)
    # A small pause keeps repeated runs friendlier to stats.nba.com.
    time.sleep(API_BASE_DELAY_SECONDS)
    return df


def _add_team_metadata(team_stats: pd.DataFrame) -> pd.DataFrame:
    team_directory = load_team_directory().rename(
        columns={
            "id": "TEAM_ID",
            "full_name": "TEAM_NAME_DIRECTORY",
            "abbreviation": "TEAM_ABBREVIATION",
        }
    )
    metadata = team_directory[["TEAM_ID", "TEAM_NAME_DIRECTORY", "TEAM_ABBREVIATION"]]
    enriched = team_stats.merge(metadata, on="TEAM_ID", how="left")
    if "TEAM_NAME" not in enriched.columns:
        enriched["TEAM_NAME"] = enriched["TEAM_NAME_DIRECTORY"]
    enriched["TEAM_NAME"] = enriched["TEAM_NAME"].fillna(enriched["TEAM_NAME_DIRECTORY"])
    return enriched.drop(columns=["TEAM_NAME_DIRECTORY"])


def load_team_stats(
    season: str,
    cache_dir: str | Path = "data/raw",
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Load team-level base and advanced stats for one season."""
    cache_dir = Path(cache_dir)
    base_path = _cache_path(cache_dir, "team_base", season, season_type)
    advanced_path = _cache_path(cache_dir, "team_advanced", season, season_type)

    def fetch_base() -> pd.DataFrame:
        response = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Base",
        )
        return response.get_data_frames()[0]

    def fetch_advanced() -> pd.DataFrame:
        response = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
        )
        return response.get_data_frames()[0]

    base = _read_or_fetch(base_path, fetch_base)
    advanced = _read_or_fetch(advanced_path, fetch_advanced)

    merged = advanced.merge(
        base[["TEAM_ID", "W_PCT", "PLUS_MINUS"]],
        on="TEAM_ID",
        suffixes=("", "_BASE"),
    )
    merged = _add_team_metadata(merged)
    return merged[TEAM_ID_COLUMNS + TEAM_STRENGTH_COLUMNS].copy()


def load_team_seeds(
    season: str,
    cache_dir: str | Path = "data/raw",
) -> dict[int, int]:
    """Load playoff seed by team id, falling back to W_PCT rank when standings are unavailable."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"team_seeds_{season}.csv"

    def fetch_standings() -> pd.DataFrame:
        response = leaguestandings.LeagueStandings(season=season, season_type="Regular Season")
        return response.get_data_frames()[0]

    standings = _read_or_fetch(path, fetch_standings, allow_failure=True)
    if standings.empty:
        stats = load_team_stats(season, cache_dir, season_type="Regular Season")
        ranked = stats.sort_values("W_PCT", ascending=False).reset_index(drop=True)
        return {int(row.TEAM_ID): int(index + 1) for index, row in ranked.iterrows()}

    team_col = "TeamID" if "TeamID" in standings.columns else "TEAM_ID"
    seed_col = next((col for col in ["PlayoffRank", "ConferenceRank", "CONF_RANK", "WINS_RANK"] if col in standings.columns), None)
    if seed_col is None:
        win_col = "WinPCT" if "WinPCT" in standings.columns else "W_PCT"
        standings = standings.sort_values(win_col, ascending=False).reset_index(drop=True)
        return {int(row[team_col]): int(index + 1) for index, row in standings.iterrows()}

    return {int(row[team_col]): int(row[seed_col]) for _, row in standings.iterrows()}


def _normalize_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    games = games.rename(columns={"Team_ID": "TEAM_ID", "Game_ID": "GAME_ID"}).copy()
    games["TEAM_ID"] = games["TEAM_ID"].astype(int)
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["IS_HOME"] = games["MATCHUP"].str.contains("vs.", regex=False, na=False).astype(int)
    games["WON"] = games["WL"].eq("W").astype(int)
    return games


def load_team_game_log(
    team_id: int,
    season: str,
    cache_dir: str | Path = "data/raw",
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Load one team's TeamGameLog rows for a season and season type."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"team_game_log_{season}_{season_type.lower().replace(' ', '_')}_{team_id}.csv"

    def fetch_games() -> pd.DataFrame:
        response = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
        )
        return response.get_data_frames()[0]

    games = _read_or_fetch(path, fetch_games)
    if games.empty:
        return games

    games = _normalize_team_game_log(games)
    games["SEASON_TYPE"] = season_type
    return games


def load_all_team_game_logs(
    season: str,
    cache_dir: str | Path = "data/raw",
    season_types: Iterable[str] = ("Regular Season", "Playoffs"),
) -> pd.DataFrame:
    """Load TeamGameLog rows for every NBA team so opponent metrics can be derived."""
    team_ids = load_team_directory()["id"].astype(int).tolist()
    frames: list[pd.DataFrame] = []

    for season_type in season_types:
        print(f"    Loading {season_type} TeamGameLog rows")
        for team_id in team_ids:
            games = load_team_game_log(team_id, season, cache_dir, season_type)
            if not games.empty:
                frames.append(games)

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(frames, ignore_index=True)
    logs = logs.drop_duplicates(subset=["TEAM_ID", "GAME_ID", "SEASON_TYPE"])
    return add_team_game_metrics(logs)


def add_team_game_metrics(games: pd.DataFrame) -> pd.DataFrame:
    """Add point differential and estimated net rating from paired TeamGameLog rows."""
    if games.empty:
        return games

    games = games.copy()
    games["POSS"] = games["FGA"] + 0.44 * games["FTA"] - games["OREB"] + games["TOV"]
    opponent_columns = [
        "GAME_ID",
        "TEAM_ID",
        "PTS",
        "POSS",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "REB",
        "TOV",
    ]
    available_opponent_columns = [column for column in opponent_columns if column in games.columns]
    opponent = games[available_opponent_columns].rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "PTS": "OPP_PTS",
            "POSS": "OPP_POSS",
            "FGM": "OPP_FGM",
            "FGA": "OPP_FGA",
            "FG3M": "OPP_FG3M",
            "FG3A": "OPP_FG3A",
            "FTM": "OPP_FTM",
            "FTA": "OPP_FTA",
            "OREB": "OPP_OREB",
            "DREB": "OPP_DREB",
            "REB": "OPP_REB",
            "TOV": "OPP_TOV",
        }
    )
    paired = games.merge(opponent, on="GAME_ID", how="left")
    paired = paired[paired["TEAM_ID"] != paired["OPP_TEAM_ID"]].copy()
    paired["POINT_DIFF"] = paired["PTS"] - paired["OPP_PTS"]
    paired["OFF_RATING_GAME"] = np.where(paired["POSS"] > 0, paired["PTS"] / paired["POSS"] * 100, np.nan)
    paired["DEF_RATING_GAME"] = np.where(
        paired["OPP_POSS"] > 0,
        paired["OPP_PTS"] / paired["OPP_POSS"] * 100,
        np.nan,
    )
    paired["NET_RATING_GAME"] = paired["OFF_RATING_GAME"] - paired["DEF_RATING_GAME"]
    return paired.drop_duplicates(subset=["TEAM_ID", "GAME_ID", "SEASON_TYPE"])


def load_playoff_games(
    season: str,
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load playoff game logs for one season."""
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, "playoff_games", season, "Playoffs")

    def fetch_games() -> pd.DataFrame:
        response = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Playoffs",
            player_or_team_abbreviation="T",
        )
        return response.get_data_frames()[0]

    games = _read_or_fetch(path, fetch_games)
    if games.empty:
        return games

    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    return games


def load_player_stats_before_date(
    season: str,
    prediction_date: pd.Timestamp | str,
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load regular-season player averages for all teams before the prediction date."""
    cache_dir = Path(cache_dir)
    date_label = _safe_date_label(prediction_date)
    path = cache_dir / f"player_stats_{season}_all_teams_through_{date_label}.csv"

    def fetch_players() -> pd.DataFrame:
        response = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            date_to_nullable=_nba_date(prediction_date),
        )
        return response.get_data_frames()[0]

    return _read_or_fetch(path, fetch_players, allow_failure=True)


def _player_stats_for_team(player_stats: pd.DataFrame, team_id: int) -> pd.DataFrame:
    if player_stats.empty or "TEAM_ID" not in player_stats.columns:
        return pd.DataFrame()
    return player_stats[player_stats["TEAM_ID"].astype(int).eq(int(team_id))].copy()


def _team_strength_features(stats_a: pd.Series, stats_b: pd.Series) -> dict[str, float]:
    # Tier 1 features: season-level team-strength differences.
    features = {
        f"{column}_DIFF": float(stats_a[column]) - float(stats_b[column])
        for column in TEAM_STRENGTH_COLUMNS
    }
    # Lower defensive rating is better, so invert this one to keep positive values
    # aligned with better Team A title odds.
    features["DEF_RATING_DIFF"] = float(stats_b["DEF_RATING"]) - float(stats_a["DEF_RATING"])
    return features


def _games_before_date(game_logs: pd.DataFrame, team_id: int, prediction_date: pd.Timestamp) -> pd.DataFrame:
    if game_logs.empty:
        return pd.DataFrame()

    games = game_logs[
        (game_logs["TEAM_ID"].astype(int) == int(team_id))
        & (game_logs["GAME_DATE"] < prediction_date)
    ].copy()
    return games.sort_values("GAME_DATE")


def _win_pct(games: pd.DataFrame) -> float:
    if games.empty:
        return np.nan
    return float(games["WON"].mean())


def _recent_form_value(games: pd.DataFrame, window: int, column: str) -> float:
    if games.empty:
        return np.nan

    recent = games.tail(window)
    if column == "WON":
        return float(recent[column].mean())
    return float(recent[column].mean())


def _recent_form_features(
    team_a_games: pd.DataFrame,
    team_b_games: pd.DataFrame,
) -> dict[str, float]:
    # Tier 2 features: rolling form from only games before prediction_date.
    return {
        "last_5_win_pct_diff": _recent_form_value(team_a_games, 5, "WON")
        - _recent_form_value(team_b_games, 5, "WON"),
        "last_10_win_pct_diff": _recent_form_value(team_a_games, 10, "WON")
        - _recent_form_value(team_b_games, 10, "WON"),
        "last_5_net_rating_diff": _recent_form_value(team_a_games, 5, "NET_RATING_GAME")
        - _recent_form_value(team_b_games, 5, "NET_RATING_GAME"),
        "last_10_net_rating_diff": _recent_form_value(team_a_games, 10, "NET_RATING_GAME")
        - _recent_form_value(team_b_games, 10, "NET_RATING_GAME"),
        "last_5_point_diff_diff": _recent_form_value(team_a_games, 5, "POINT_DIFF")
        - _recent_form_value(team_b_games, 5, "POINT_DIFF"),
        "last_10_point_diff_diff": _recent_form_value(team_a_games, 10, "POINT_DIFF")
        - _recent_form_value(team_b_games, 10, "POINT_DIFF"),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return float(numerator / denominator)


def _efficiency_values(games: pd.DataFrame) -> dict[str, float]:
    if games.empty:
        return {
            "efg_pct": np.nan,
            "ts_pct": np.nan,
            "turnover_pct": np.nan,
            "reb_pct": np.nan,
            "assist_turnover_ratio": np.nan,
            "ft_rate": np.nan,
        }

    fgm = games["FGM"].sum()
    fga = games["FGA"].sum()
    fg3m = games["FG3M"].sum()
    fta = games["FTA"].sum()
    pts = games["PTS"].sum()
    tov = games["TOV"].sum()
    ast = games["AST"].sum()
    reb = games["REB"].sum()
    opp_reb = games["OPP_REB"].sum() if "OPP_REB" in games.columns else np.nan

    return {
        "efg_pct": _safe_divide(fgm + 0.5 * fg3m, fga),
        "ts_pct": _safe_divide(pts, 2 * (fga + 0.44 * fta)),
        "turnover_pct": _safe_divide(tov, fga + 0.44 * fta + tov),
        "reb_pct": _safe_divide(reb, reb + opp_reb),
        "assist_turnover_ratio": _safe_divide(ast, tov),
        "ft_rate": _safe_divide(fta, fga),
    }


def _weighted_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    weights = np.linspace(1.0, 2.0, len(values))
    return float(np.average(values.to_numpy(dtype=float), weights=weights))


def _game_ts_values(games: pd.DataFrame) -> pd.Series:
    denominator = 2 * (games["FGA"] + 0.44 * games["FTA"])
    return pd.Series(np.where(denominator > 0, games["PTS"] / denominator, np.nan), index=games.index)


def _weighted_recent_features(team_a_games: pd.DataFrame, team_b_games: pd.DataFrame, window: int = 10) -> dict[str, float]:
    # More recent games get larger linear weights, and all rows are before prediction_date.
    team_a_recent = team_a_games.tail(window).copy()
    team_b_recent = team_b_games.tail(window).copy()
    team_a_ts = _game_ts_values(team_a_recent)
    team_b_ts = _game_ts_values(team_b_recent)
    return {
        "weighted_recent_win_pct_diff": _weighted_mean(team_a_recent["WON"]) - _weighted_mean(team_b_recent["WON"]),
        "weighted_recent_net_rating_diff": _weighted_mean(team_a_recent["NET_RATING_GAME"])
        - _weighted_mean(team_b_recent["NET_RATING_GAME"]),
        "weighted_recent_ts_pct_diff": _weighted_mean(team_a_ts) - _weighted_mean(team_b_ts),
        "weighted_recent_def_rating_diff": _weighted_mean(team_b_recent["DEF_RATING_GAME"])
        - _weighted_mean(team_a_recent["DEF_RATING_GAME"]),
    }


def _efficiency_features(team_a_games: pd.DataFrame, team_b_games: pd.DataFrame) -> dict[str, float]:
    # Efficiency features use only games before prediction_date.
    team_a = _efficiency_values(team_a_games)
    team_b = _efficiency_values(team_b_games)
    return {
        "efg_pct_diff": team_a["efg_pct"] - team_b["efg_pct"],
        "ts_pct_diff": team_a["ts_pct"] - team_b["ts_pct"],
        "turnover_pct_diff": team_b["turnover_pct"] - team_a["turnover_pct"],
        "reb_pct_diff": team_a["reb_pct"] - team_b["reb_pct"],
        "assist_turnover_ratio_diff": team_a["assist_turnover_ratio"] - team_b["assist_turnover_ratio"],
        "ft_rate_diff": team_a["ft_rate"] - team_b["ft_rate"],
    }


def _top_player_totals(player_stats: pd.DataFrame, count: int) -> tuple[float, float, float]:
    if player_stats.empty or "PTS" not in player_stats.columns or "MIN" not in player_stats.columns:
        return 0.0, 0.0, 0.0

    top_players = player_stats.sort_values("PTS", ascending=False).head(count)
    if len(top_players) < count:
        return 0.0, 0.0, 0.0

    points = float(top_players["PTS"].sum())
    minutes = float(top_players["MIN"].sum())
    if {"FGA", "FTA"}.issubset(top_players.columns):
        ts_pct = _safe_divide(points, 2 * (float(top_players["FGA"].sum()) + 0.44 * float(top_players["FTA"].sum())))
    else:
        ts_pct = 0.0
    return points, minutes, 0.0 if pd.isna(ts_pct) else ts_pct


def _star_power_features(player_stats_a: pd.DataFrame, player_stats_b: pd.DataFrame) -> dict[str, float]:
    # Tier 3 features: regular-season star-power estimates before prediction_date.
    team_a_top_1_ppg, team_a_top_1_mpg, team_a_top_1_ts = _top_player_totals(player_stats_a, 1)
    team_b_top_1_ppg, team_b_top_1_mpg, team_b_top_1_ts = _top_player_totals(player_stats_b, 1)
    team_a_top_3_ppg, team_a_top_3_mpg, team_a_top_3_ts = _top_player_totals(player_stats_a, 3)
    team_b_top_3_ppg, team_b_top_3_mpg, team_b_top_3_ts = _top_player_totals(player_stats_b, 3)
    team_a_top_5_ppg, team_a_top_5_mpg, _team_a_top_5_ts = _top_player_totals(player_stats_a, 5)
    team_b_top_5_ppg, team_b_top_5_mpg, _team_b_top_5_ts = _top_player_totals(player_stats_b, 5)

    return {
        "top_1_ppg_diff": team_a_top_1_ppg - team_b_top_1_ppg,
        "top_1_mpg_diff": team_a_top_1_mpg - team_b_top_1_mpg,
        "top_1_ts_pct_diff": team_a_top_1_ts - team_b_top_1_ts,
        "top_3_ppg_diff": team_a_top_3_ppg - team_b_top_3_ppg,
        "top_3_mpg_diff": team_a_top_3_mpg - team_b_top_3_mpg,
        "top_3_ts_pct_diff": team_a_top_3_ts - team_b_top_3_ts,
        "top_5_ppg_diff": team_a_top_5_ppg - team_b_top_5_ppg,
        "top_5_mpg_diff": team_a_top_5_mpg - team_b_top_5_mpg,
    }


def _home_court_features(
    team_a_games: pd.DataFrame,
    team_b_games: pd.DataFrame,
    team_a_id: int,
    home_team_id: int,
) -> dict[str, float]:
    # Tier 4 features: explicit home flag plus modest home-edge signals.
    # The legacy split features are kept for research ablation, but production
    # uses home_team_A and home_advantage_diff to avoid double-counting strength.
    team_a_home_games = team_a_games[team_a_games["IS_HOME"].eq(1)]
    team_b_home_games = team_b_games[team_b_games["IS_HOME"].eq(1)]
    team_a_away_games = team_a_games[team_a_games["IS_HOME"].eq(0)]
    team_b_away_games = team_b_games[team_b_games["IS_HOME"].eq(0)]
    team_a_is_home = int(team_a_id) == int(home_team_id)
    clip_range = (-0.25, 0.25)

    if team_a_is_home:
        matchup_split_diff = _win_pct(team_a_home_games) - _win_pct(team_b_away_games)
        opposite_split_diff = _win_pct(team_a_away_games) - _win_pct(team_b_home_games)
    else:
        matchup_split_diff = _win_pct(team_a_away_games) - _win_pct(team_b_home_games)
        opposite_split_diff = _win_pct(team_a_home_games) - _win_pct(team_b_away_games)

    team_a_home_advantage = _win_pct(team_a_home_games) - _win_pct(team_a_games)
    team_b_home_advantage = _win_pct(team_b_home_games) - _win_pct(team_b_games)
    selected_home_advantage = team_a_home_advantage if team_a_is_home else -team_b_home_advantage

    return {
        "home_team_A": float(int(team_a_is_home)),
        "home_win_pct_diff": matchup_split_diff,
        "away_win_pct_diff": opposite_split_diff,
        "team_A_home_advantage": team_a_home_advantage,
        "team_B_home_advantage": team_b_home_advantage,
        "home_advantage_diff": float(np.clip(selected_home_advantage, *clip_range)),
        "clipped_home_win_pct_diff": float(np.clip(matchup_split_diff, *clip_range)),
        "clipped_away_win_pct_diff": float(np.clip(opposite_split_diff, *clip_range)),
    }


def _rest_days(games: pd.DataFrame, prediction_date: pd.Timestamp) -> float:
    if games.empty:
        return np.nan

    previous_date = games["GAME_DATE"].max()
    return float((prediction_date.normalize() - previous_date.normalize()).days)


def _rest_features(
    team_a_games: pd.DataFrame,
    team_b_games: pd.DataFrame,
    prediction_date: pd.Timestamp,
) -> dict[str, float]:
    # Tier 5 features: rest/fatigue based on each team's previous game date.
    team_a_rest = _rest_days(team_a_games, prediction_date)
    team_b_rest = _rest_days(team_b_games, prediction_date)
    return {
        "rest_days_diff": team_a_rest - team_b_rest,
        "team_A_rest_days": team_a_rest,
        "team_B_rest_days": team_b_rest,
    }


def compute_elo_snapshots(game_logs: pd.DataFrame, k_factor: float = 20.0) -> dict[tuple[str, int], float]:
    """Return pre-game Elo for each (GAME_ID, TEAM_ID), resetting all teams to 1500 per season load."""
    if game_logs.empty:
        return {}

    snapshots: dict[tuple[str, int], float] = {}
    ratings = {int(team_id): 1500.0 for team_id in game_logs["TEAM_ID"].dropna().unique()}
    game_rows = game_logs.sort_values(["GAME_DATE", "GAME_ID"]).groupby("GAME_ID", sort=False)

    for game_id, group in game_rows:
        if len(group) < 2:
            continue

        teams_in_game = group.sort_values("TEAM_ID").head(2)
        row_a = teams_in_game.iloc[0]
        row_b = teams_in_game.iloc[1]
        team_a_id = int(row_a["TEAM_ID"])
        team_b_id = int(row_b["TEAM_ID"])
        rating_a = ratings.get(team_a_id, 1500.0)
        rating_b = ratings.get(team_b_id, 1500.0)
        snapshots[(str(game_id), team_a_id)] = rating_a
        snapshots[(str(game_id), team_b_id)] = rating_b

        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        actual_a = float(row_a["WON"])
        actual_b = 1.0 - actual_a
        expected_b = 1.0 - expected_a
        ratings[team_a_id] = rating_a + k_factor * (actual_a - expected_a)
        ratings[team_b_id] = rating_b + k_factor * (actual_b - expected_b)

    return snapshots


def compute_current_elos(game_logs: pd.DataFrame, prediction_date: pd.Timestamp, k_factor: float = 20.0) -> dict[int, float]:
    """Return Elo after all games before prediction_date."""
    ratings = {int(team_id): 1500.0 for team_id in game_logs["TEAM_ID"].dropna().unique()}
    prior_logs = game_logs[game_logs["GAME_DATE"] < prediction_date]

    for _, group in prior_logs.sort_values(["GAME_DATE", "GAME_ID"]).groupby("GAME_ID", sort=False):
        if len(group) < 2:
            continue

        teams_in_game = group.sort_values("TEAM_ID").head(2)
        row_a = teams_in_game.iloc[0]
        row_b = teams_in_game.iloc[1]
        team_a_id = int(row_a["TEAM_ID"])
        team_b_id = int(row_b["TEAM_ID"])
        rating_a = ratings.get(team_a_id, 1500.0)
        rating_b = ratings.get(team_b_id, 1500.0)
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        actual_a = float(row_a["WON"])
        ratings[team_a_id] = rating_a + k_factor * (actual_a - expected_a)
        ratings[team_b_id] = rating_b + k_factor * ((1.0 - actual_a) - (1.0 - expected_a))

    return ratings


def _elo_features(
    team_a_id: int,
    team_b_id: int,
    prediction_date: pd.Timestamp,
    game_logs: pd.DataFrame,
    elo_snapshots: dict[tuple[str, int], float] | None = None,
    game_id: str | None = None,
) -> dict[str, float]:
    if elo_snapshots and game_id is not None:
        elo_a = elo_snapshots.get((str(game_id), int(team_a_id)), 1500.0)
        elo_b = elo_snapshots.get((str(game_id), int(team_b_id)), 1500.0)
        return {"elo_diff": elo_a - elo_b}

    current_elos = compute_current_elos(game_logs, prediction_date)
    elo_a = current_elos.get(int(team_a_id), 1500.0)
    elo_b = current_elos.get(int(team_b_id), 1500.0)
    return {"elo_diff": elo_a - elo_b}


def _head_to_head_features(
    team_a_games: pd.DataFrame,
    team_b_id: int,
) -> dict[str, float]:
    # Head-to-head uses only Team A rows from games before prediction_date.
    h2h = team_a_games[team_a_games["OPP_TEAM_ID"].astype(int).eq(int(team_b_id))].copy()
    if h2h.empty:
        return {
            "season_h2h_win_pct_diff": 0.0,
            "season_h2h_margin_diff": 0.0,
            "h2h_win_pct_diff": 0.0,
            "h2h_margin_diff": 0.0,
            "h2h_net_rating_diff": 0.0,
            "h2h_efg_pct_diff": 0.0,
            "h2h_ts_pct_diff": 0.0,
            "h2h_turnover_pct_diff": 0.0,
            "h2h_reb_pct_diff": 0.0,
        }

    team_a_win_pct = float(h2h["WON"].mean())
    team_a_margin = float(h2h["POINT_DIFF"].mean())
    efficiency = _efficiency_values(h2h)
    opp_fgm = h2h["OPP_FGM"].sum() if "OPP_FGM" in h2h.columns else np.nan
    opp_fga = h2h["OPP_FGA"].sum() if "OPP_FGA" in h2h.columns else np.nan
    opp_fg3m = h2h["OPP_FG3M"].sum() if "OPP_FG3M" in h2h.columns else np.nan
    opp_fta = h2h["OPP_FTA"].sum() if "OPP_FTA" in h2h.columns else np.nan
    opp_pts = h2h["OPP_PTS"].sum() if "OPP_PTS" in h2h.columns else np.nan
    opp_tov = h2h["OPP_TOV"].sum() if "OPP_TOV" in h2h.columns else np.nan
    net_rating = h2h["NET_RATING_GAME"].mean() if "NET_RATING_GAME" in h2h.columns else np.nan
    return {
        "season_h2h_win_pct_diff": team_a_win_pct - (1.0 - team_a_win_pct),
        "season_h2h_margin_diff": team_a_margin - (-team_a_margin),
        "h2h_win_pct_diff": team_a_win_pct - (1.0 - team_a_win_pct),
        "h2h_margin_diff": team_a_margin - (-team_a_margin),
        "h2h_net_rating_diff": float(2 * net_rating) if not pd.isna(net_rating) else np.nan,
        "h2h_efg_pct_diff": efficiency["efg_pct"] - _safe_divide(opp_fgm + 0.5 * opp_fg3m, opp_fga),
        "h2h_ts_pct_diff": efficiency["ts_pct"] - _safe_divide(opp_pts, 2 * (opp_fga + 0.44 * opp_fta)),
        "h2h_turnover_pct_diff": _safe_divide(opp_tov, opp_fga + 0.44 * opp_fta + opp_tov)
        - efficiency["turnover_pct"],
        "h2h_reb_pct_diff": efficiency["reb_pct"] - (1.0 - efficiency["reb_pct"]),
    }


def _style_values(games: pd.DataFrame) -> dict[str, float]:
    if games.empty:
        return {
            "three_point_rate": np.nan,
            "opp_three_point_rate": np.nan,
            "paint_proxy_rate": np.nan,
            "opp_paint_proxy_rate": np.nan,
            "oreb_rate": np.nan,
            "dreb_rate": np.nan,
            "tov_rate": np.nan,
            "opp_tov_rate": np.nan,
            "ft_rate": np.nan,
            "opp_ft_rate": np.nan,
        }

    two_point_points = 2 * (games["FGM"].sum() - games["FG3M"].sum())
    opp_two_point_points = 2 * (games["OPP_FGM"].sum() - games["OPP_FG3M"].sum())
    return {
        "three_point_rate": _safe_divide(games["FG3A"].sum(), games["FGA"].sum()),
        "opp_three_point_rate": _safe_divide(games["OPP_FG3A"].sum(), games["OPP_FGA"].sum()),
        "paint_proxy_rate": _safe_divide(two_point_points, games["FGA"].sum()),
        "opp_paint_proxy_rate": _safe_divide(opp_two_point_points, games["OPP_FGA"].sum()),
        "oreb_rate": _safe_divide(games["OREB"].sum(), games["OREB"].sum() + games["OPP_DREB"].sum()),
        "dreb_rate": _safe_divide(games["DREB"].sum(), games["DREB"].sum() + games["OPP_OREB"].sum()),
        "tov_rate": _safe_divide(games["TOV"].sum(), games["FGA"].sum() + 0.44 * games["FTA"].sum() + games["TOV"].sum()),
        "opp_tov_rate": _safe_divide(games["OPP_TOV"].sum(), games["OPP_FGA"].sum() + 0.44 * games["OPP_FTA"].sum() + games["OPP_TOV"].sum()),
        "ft_rate": _safe_divide(games["FTA"].sum(), games["FGA"].sum()),
        "opp_ft_rate": _safe_divide(games["OPP_FTA"].sum(), games["OPP_FGA"].sum()),
    }


def _style_matchup_features(team_a_games: pd.DataFrame, team_b_games: pd.DataFrame) -> dict[str, float]:
    team_a = _style_values(team_a_games)
    team_b = _style_values(team_b_games)
    return {
        "three_point_offense_vs_defense_diff": (team_a["three_point_rate"] - team_b["opp_three_point_rate"])
        - (team_b["three_point_rate"] - team_a["opp_three_point_rate"]),
        "paint_scoring_vs_paint_defense_diff": (team_a["paint_proxy_rate"] - team_b["opp_paint_proxy_rate"])
        - (team_b["paint_proxy_rate"] - team_a["opp_paint_proxy_rate"]),
        "offensive_rebound_vs_defensive_rebound_diff": (team_a["oreb_rate"] - team_b["dreb_rate"])
        - (team_b["oreb_rate"] - team_a["dreb_rate"]),
        "turnover_creation_vs_turnover_rate_diff": (team_a["opp_tov_rate"] - team_b["tov_rate"])
        - (team_b["opp_tov_rate"] - team_a["tov_rate"]),
        "ft_rate_vs_foul_rate_diff": (team_a["ft_rate"] - team_b["opp_ft_rate"])
        - (team_b["ft_rate"] - team_a["opp_ft_rate"]),
    }


def _playoff_context_features(
    team_a_id: int,
    team_b_id: int,
    prediction_date: pd.Timestamp,
    playoff_games: pd.DataFrame,
    seeds: dict[int, int] | None,
) -> dict[str, float]:
    seed_a = seeds.get(int(team_a_id), np.nan) if seeds else np.nan
    seed_b = seeds.get(int(team_b_id), np.nan) if seeds else np.nan
    prior_series = pd.DataFrame()
    if not playoff_games.empty:
        prior_series = playoff_games[
            (playoff_games["GAME_DATE"] < prediction_date)
            & (playoff_games["TEAM_ID"].astype(int).isin([int(team_a_id), int(team_b_id)]))
        ].copy()
        game_team_sets = prior_series.groupby("GAME_ID")["TEAM_ID"].apply(lambda ids: set(ids.astype(int)))
        h2h_game_ids = [
            game_id
            for game_id, team_ids in game_team_sets.items()
            if {int(team_a_id), int(team_b_id)}.issubset(team_ids)
        ]
        prior_series = prior_series[prior_series["GAME_ID"].isin(h2h_game_ids)]

    team_a_wins = 0
    team_b_wins = 0
    if not prior_series.empty:
        team_a_wins = int(((prior_series["TEAM_ID"].astype(int) == int(team_a_id)) & prior_series["WL"].eq("W")).sum())
        team_b_wins = int(((prior_series["TEAM_ID"].astype(int) == int(team_b_id)) & prior_series["WL"].eq("W")).sum())

    return {
        "seed_difference": float(seed_b - seed_a) if not pd.isna(seed_a) and not pd.isna(seed_b) else np.nan,
        "higher_seed_A": float(int(seed_a < seed_b)) if not pd.isna(seed_a) and not pd.isna(seed_b) else np.nan,
        "game_number": float(team_a_wins + team_b_wins + 1),
        "elimination_game": float(int(team_a_wins == 3 or team_b_wins == 3)),
        "series_score_diff": float(team_a_wins - team_b_wins),
    }


def _neutral_playoff_context_features(seeds: dict[int, int] | None, team_a_id: int, team_b_id: int) -> dict[str, float]:
    seed_a = seeds.get(int(team_a_id), np.nan) if seeds else np.nan
    seed_b = seeds.get(int(team_b_id), np.nan) if seeds else np.nan
    return {
        "seed_difference": float(seed_b - seed_a) if not pd.isna(seed_a) and not pd.isna(seed_b) else np.nan,
        "higher_seed_A": float(int(seed_a < seed_b)) if not pd.isna(seed_a) and not pd.isna(seed_b) else np.nan,
        "game_number": 1.0,
        "elimination_game": 0.0,
        "series_score_diff": 0.0,
    }


def user_playoff_series_context_features(
    game_number: int,
    team_a_series_wins: int,
    team_b_series_wins: int,
) -> dict[str, float]:
    """Return user-supplied hypothetical series context for an explicit playoff mode."""
    return {
        "game_number": float(game_number),
        "elimination_game": float(int(team_a_series_wins == 3 or team_b_series_wins == 3)),
        "series_score_diff": float(team_a_series_wins - team_b_series_wins),
    }


def build_matchup_feature_row(
    season: str,
    team_a_id: int,
    team_b_id: int,
    prediction_date: pd.Timestamp | str,
    home_team_id: int,
    cache_dir: str | Path = "data/raw",
    feature_season_type: str = "Regular Season",
    game_logs: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
    playoff_games: pd.DataFrame | None = None,
    seeds: dict[int, int] | None = None,
    elo_snapshots: dict[tuple[str, int], float] | None = None,
    game_id: str | None = None,
    include_playoff_context: bool = False,
    user_series_context: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build one matchup row with Tier 1 through Tier 5 features."""
    prediction_date = pd.to_datetime(prediction_date)
    team_stats = load_team_stats(season, cache_dir, season_type=feature_season_type)
    stats_by_team = team_stats.set_index("TEAM_ID")

    missing_team_stats = [team_id for team_id in [team_a_id, team_b_id] if team_id not in stats_by_team.index]
    if missing_team_stats:
        raise ValueError(f"Missing team-strength stats for team ids: {missing_team_stats}")

    if game_logs is None:
        game_logs = load_all_team_game_logs(season, cache_dir, season_types=(feature_season_type,))

    if game_logs.empty:
        raise ValueError(f"Missing TeamGameLog data for {season}.")

    if player_stats is None:
        player_stats = load_player_stats_before_date(season, prediction_date, cache_dir)
    if playoff_games is None:
        playoff_games = load_playoff_games(season, cache_dir)
    if seeds is None:
        seeds = load_team_seeds(season, cache_dir)

    player_stats_a = _player_stats_for_team(player_stats, team_a_id)
    player_stats_b = _player_stats_for_team(player_stats, team_b_id)

    team_a_games = _games_before_date(game_logs, team_a_id, prediction_date)
    team_b_games = _games_before_date(game_logs, team_b_id, prediction_date)
    if team_a_games.empty or team_b_games.empty:
        missing = []
        if team_a_games.empty:
            missing.append(str(team_a_id))
        if team_b_games.empty:
            missing.append(str(team_b_id))
        raise ValueError(f"Missing prior game logs before {prediction_date.date()} for team ids: {', '.join(missing)}")

    features = {
        **_team_strength_features(stats_by_team.loc[team_a_id], stats_by_team.loc[team_b_id]),
        **_recent_form_features(team_a_games, team_b_games),
        **_weighted_recent_features(team_a_games, team_b_games),
        **_star_power_features(player_stats_a, player_stats_b),
        **_home_court_features(team_a_games, team_b_games, team_a_id, home_team_id),
        **_rest_features(team_a_games, team_b_games, prediction_date),
        **_elo_features(team_a_id, team_b_id, prediction_date, game_logs, elo_snapshots, game_id),
        **_efficiency_features(team_a_games, team_b_games),
        **_head_to_head_features(team_a_games, team_b_id),
        **(
            _playoff_context_features(team_a_id, team_b_id, prediction_date, playoff_games, seeds)
            if include_playoff_context
            else _neutral_playoff_context_features(seeds, team_a_id, team_b_id)
        ),
        **_style_matchup_features(team_a_games, team_b_games),
    }
    if user_series_context is not None:
        features.update(user_series_context)

    missing_features = [column for column in MODEL_FEATURE_COLUMNS if column not in features or pd.isna(features[column])]
    if missing_features:
        features["MISSING_FEATURE_COUNT"] = float(len(missing_features))
    else:
        features["MISSING_FEATURE_COUNT"] = 0.0

    return {column: features.get(column, np.nan) for column in MODEL_FEATURE_COLUMNS + ["MISSING_FEATURE_COUNT"]}


def _safe_season_type_label(season_type: str) -> str:
    return season_type.lower().replace(" ", "_")


def _processed_training_path(processed_dir: str | Path, feature_season_type: str = "Regular Season") -> Path:
    return Path(processed_dir) / f"training_matchups_{_safe_season_type_label(feature_season_type)}.csv"


def _season_training_path(processed_dir: str | Path, season: str, feature_season_type: str = "Regular Season") -> Path:
    return Path(processed_dir) / f"training_matchups_{season}_{_safe_season_type_label(feature_season_type)}.csv"


def _read_processed_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def _has_current_feature_schema(df: pd.DataFrame) -> bool:
    required_columns = set(FEATURE_COLUMNS + ["FEATURE_SCHEMA_VERSION"])
    if not required_columns.issubset(df.columns):
        return False
    return df["FEATURE_SCHEMA_VERSION"].astype(str).eq(FEATURE_SCHEMA_VERSION).all()


def _build_training_frame_for_season(
    season: str,
    cache_dir: str | Path = "data/raw",
    feature_season_type: str = "Regular Season",
) -> pd.DataFrame:
    rows: list[dict] = []

    print(f"Processing season {season}")
    games = load_playoff_games(season, cache_dir)
    if games.empty:
        print(f"  No playoff games found for {season}; skipping")
        return pd.DataFrame()

    print(f"  Loading team game logs for {season}")
    game_logs = load_all_team_game_logs(season, cache_dir, season_types=(feature_season_type,))
    player_stats = load_player_stats_before_date(season, games["GAME_DATE"].max(), cache_dir)
    seeds = load_team_seeds(season, cache_dir)
    elo_snapshots = compute_elo_snapshots(game_logs)
    playoff_groups = list(games.groupby("GAME_ID"))

    for index, (game_id, group) in enumerate(playoff_groups, start=1):
        if len(group) != 2:
            continue

        sorted_group = group.sort_values("TEAM_ID").reset_index(drop=True)
        team_a = sorted_group.iloc[0]
        team_b = sorted_group.iloc[1]
        home_team_id = int(team_a["TEAM_ID"]) if "vs." in str(team_a["MATCHUP"]) else int(team_b["TEAM_ID"])
        print(
            f"  [{index}/{len(playoff_groups)}] {season} game {game_id}: "
            f"{team_a['TEAM_ABBREVIATION']} vs {team_b['TEAM_ABBREVIATION']} on {team_a['GAME_DATE'].date()}"
        )

        try:
            feature_row = build_matchup_feature_row(
                season=season,
                team_a_id=int(team_a["TEAM_ID"]),
                team_b_id=int(team_b["TEAM_ID"]),
                prediction_date=team_a["GAME_DATE"],
                home_team_id=home_team_id,
                cache_dir=cache_dir,
                feature_season_type=feature_season_type,
                game_logs=game_logs,
                player_stats=player_stats,
                playoff_games=games,
                seeds=seeds,
                elo_snapshots=elo_snapshots,
                game_id=str(game_id),
                include_playoff_context=True,
            )
        except ValueError as exc:
            print(f"    Skipping game {game_id}: {exc}")
            continue

        row = {
            "FEATURE_SCHEMA_VERSION": FEATURE_SCHEMA_VERSION,
            "SEASON": season,
            "GAME_ID": game_id,
            "GAME_DATE": team_a["GAME_DATE"],
            "TEAM_A_ID": int(team_a["TEAM_ID"]),
            "TEAM_A": team_a["TEAM_ABBREVIATION"],
            "TEAM_B_ID": int(team_b["TEAM_ID"]),
            "TEAM_B": team_b["TEAM_ABBREVIATION"],
            "TEAM_A_WON": int(team_a["WL"] == "W"),
        }
        row.update(feature_row)
        rows.append(row)

    return pd.DataFrame(rows)


def load_team_directory() -> pd.DataFrame:
    """Return active NBA team ids, names, cities, and abbreviations."""
    return pd.DataFrame(teams.get_teams())


def resolve_team_id(team_query: str) -> int:
    """Resolve a team abbreviation, nickname, city, or full name to TEAM_ID."""
    query = team_query.strip().lower()
    team_df = load_team_directory()

    candidates = team_df[
        team_df["abbreviation"].str.lower().eq(query)
        | team_df["nickname"].str.lower().eq(query)
        | team_df["city"].str.lower().eq(query)
        | team_df["full_name"].str.lower().eq(query)
    ]

    if candidates.empty:
        valid = ", ".join(sorted(team_df["abbreviation"].tolist()))
        raise ValueError(f"Unknown team '{team_query}'. Try an abbreviation like one of: {valid}")

    return int(candidates.iloc[0]["id"])


def load_training_frame(
    seasons: Iterable[str],
    cache_dir: str | Path = "data/raw",
    feature_season_type: str = "Regular Season",
    processed_dir: str | Path = "data/processed",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load or build a matchup-level training set from team stats and playoff results."""
    seasons = list(seasons)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    combined_path = _processed_training_path(processed_dir, feature_season_type)

    if combined_path.exists() and not force_refresh:
        cached = _read_processed_training(combined_path)
        if not _has_current_feature_schema(cached):
            print(f"Processed cache {combined_path} has an old feature schema; rebuilding")
            cached = pd.DataFrame()
        else:
            cached_seasons = set(cached["SEASON"].astype(str).unique()) if "SEASON" in cached.columns else set()
            if set(seasons).issubset(cached_seasons):
                print(f"Loading processed training data from {combined_path}")
                return cached[cached["SEASON"].astype(str).isin(seasons)].copy()
            print(f"Processed cache {combined_path} exists but does not cover all requested seasons; checking season caches")

    frames: list[pd.DataFrame] = []

    for season in seasons:
        season_path = _season_training_path(processed_dir, season, feature_season_type)
        if season_path.exists() and not force_refresh:
            season_cached = _read_processed_training(season_path)
            if _has_current_feature_schema(season_cached):
                print(f"Loading cached engineered rows for {season} from {season_path}")
                frames.append(season_cached)
                continue
            print(f"Season cache {season_path} has an old feature schema; rebuilding")

        season_frame = _build_training_frame_for_season(
            season,
            cache_dir=cache_dir,
            feature_season_type=feature_season_type,
        )
        if not season_frame.empty:
            season_frame.to_csv(season_path, index=False)
            print(f"Saved engineered rows for {season} to {season_path}")
            frames.append(season_frame)

    if not frames:
        return pd.DataFrame()

    training_frame = pd.concat(frames, ignore_index=True)
    training_frame.to_csv(combined_path, index=False)
    print(f"Saved processed training data to {combined_path}")
    return training_frame
