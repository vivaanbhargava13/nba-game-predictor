from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .advanced_boxscore_v3_cache_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_GAME_LOG_DIR,
    PLAYER_FIELDS,
    REQUEST_DELAY_SECONDS,
    TEAM_FIELDS,
    UNAVAILABLE_PLAYER_FIELDS,
    cache_game_payload,
    cache_path_for_game,
    discover_playoff_games,
    fetch_v3_payload,
    load_cached_payload,
    parse_v3_payload,
)

BACKFILL_SEASONS = tuple(
    f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2025)
)
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_DOC_PATH = Path("docs/advanced_boxscore_v3_playoff_backfill.md")
SYSTEMIC_FAILURE_THRESHOLD = 5
SUCCESS_CLASSES = {"success", "cache_hit"}


def classify_result(result: dict) -> str:
    if result["status"] == "fetched":
        return "success"
    if result["status"] in {"already_cached", "stale_cache"}:
        return "cache_hit"
    error_type = str(result.get("error_type", "")).lower()
    error = str(result.get("error", "")).lower()
    if error_type == "empty_response":
        return "empty_response"
    if error_type in {
        "malformed_response",
        "missing_team_rows",
        "missing_player_rows",
    }:
        return "malformed_response"
    if any(
        token in error_type
        for token in (
            "connection",
            "timeout",
            "proxy",
            "ssl",
            "dns",
        )
    ) or any(
        token in error
        for token in (
            "connection",
            "timed out",
            "timeout",
            "name resolution",
        )
    ):
        return "transport_error"
    return "api_error"


def _duplicates(frame: pd.DataFrame, columns: list[str]) -> int:
    if frame.empty or not set(columns).issubset(frame.columns):
        return 0
    return int(frame.duplicated(columns, keep=False).sum())


def _ledger_row(game: pd.Series, result: dict) -> dict:
    quality_class = classify_result(result)
    payload = result.get("payload")
    team_rows, player_rows = (
        parse_v3_payload(payload)
        if payload is not None
        else (pd.DataFrame(), pd.DataFrame())
    )
    return {
        "season": game["SEASON"],
        "game_id": game["GAME_ID"],
        "game_date": game["GAME_DATE"],
        "quality_class": quality_class,
        "cached_or_fetched": (
            "cached"
            if quality_class == "cache_hit"
            else "fetched"
            if quality_class == "success"
            else "none"
        ),
        "attempt_count": int(result.get("attempt_count", 0)),
        "retries_used": max(int(result.get("attempt_count", 0)) - 1, 0),
        "error_type": result.get("error_type", ""),
        "error": result.get("error", ""),
        "team_rows_count": len(team_rows),
        "player_rows_count": len(player_rows),
        "duplicate_team_rows": _duplicates(
            team_rows, ["gameId", "teamId"]
        ),
        "duplicate_player_rows": _duplicates(
            player_rows, ["gameId", "personId"]
        ),
        "request_seconds": round(
            float(result.get("request_seconds", 0.0)), 4
        ),
        "cache_path": result["cache_path"],
    }


def _circuit_breaker_row(
    game: pd.Series,
    cache_dir: str | Path,
    reason: str,
) -> dict:
    return {
        "season": game["SEASON"],
        "game_id": game["GAME_ID"],
        "game_date": game["GAME_DATE"],
        "quality_class": "circuit_breaker_skipped",
        "cached_or_fetched": "none",
        "attempt_count": 0,
        "retries_used": 0,
        "error_type": "circuit_breaker",
        "error": reason,
        "team_rows_count": 0,
        "player_rows_count": 0,
        "duplicate_team_rows": 0,
        "duplicate_player_rows": 0,
        "request_seconds": 0.0,
        "cache_path": str(
            cache_path_for_game(
                cache_dir, game["SEASON"], game["GAME_ID"]
            )
        ),
    }


def materialize_rows(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_frames = []
    player_frames = []
    successful = ledger[
        ledger["quality_class"].isin(SUCCESS_CLASSES)
    ]
    for row in successful.itertuples(index=False):
        payload = load_cached_payload(Path(row.cache_path))
        if payload is None:
            continue
        team_rows, player_rows = parse_v3_payload(payload)
        for frame in (team_rows, player_rows):
            frame.insert(0, "season", row.season)
            frame.insert(1, "gameDate", row.game_date)
        team_frames.append(team_rows)
        player_frames.append(player_rows)

    teams = (
        pd.concat(team_frames, ignore_index=True)
        if team_frames
        else pd.DataFrame()
    )
    players = (
        pd.concat(player_frames, ignore_index=True)
        if player_frames
        else pd.DataFrame()
    )
    return teams, players


def build_missingness_report(
    ledger: pd.DataFrame,
    team_rows: pd.DataFrame,
    player_rows: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    seasons = sorted(ledger["season"].dropna().astype(str).unique())
    for season in seasons:
        season_ledger = ledger[ledger["season"].eq(season)]
        season_teams = team_rows[
            team_rows["season"].eq(season)
        ] if not team_rows.empty else pd.DataFrame()
        season_players = player_rows[
            player_rows["season"].eq(season)
        ] if not player_rows.empty else pd.DataFrame()

        for split, frame, fields in (
            ("team", season_teams, TEAM_FIELDS),
            ("player", season_players, PLAYER_FIELDS),
        ):
            for field in fields:
                expected = len(frame)
                present = (
                    int(frame[field].notna().sum())
                    if field in frame
                    else 0
                )
                missing = max(expected - present, 0)
                records.append(
                    _missingness_record(
                        season,
                        split,
                        "field_missingness",
                        field,
                        expected,
                        present,
                        missing,
                        "available",
                    )
                )

        for field in UNAVAILABLE_PLAYER_FIELDS:
            records.append(
                _missingness_record(
                    season,
                    "player",
                    "field_availability",
                    field,
                    len(season_players),
                    0,
                    0,
                    "unavailable_in_v3",
                )
            )

        expected_team_rows = (
            int(
                season_ledger["quality_class"]
                .isin(SUCCESS_CLASSES)
                .sum()
            )
            * 2
        )
        records.extend(
            [
                _row_check(
                    season,
                    "team",
                    "team_rows_exactly_two_per_successful_game",
                    expected_team_rows,
                    len(season_teams),
                ),
                _row_check(
                    season,
                    "player",
                    "successful_games_with_player_rows",
                    int(
                        season_ledger["quality_class"]
                        .isin(SUCCESS_CLASSES)
                        .sum()
                    ),
                    (
                        int(season_players["gameId"].nunique())
                        if "gameId" in season_players
                        else 0
                    ),
                ),
                _count_check(
                    season,
                    "team",
                    "duplicate_team_rows",
                    _duplicates(season_teams, ["gameId", "teamId"]),
                ),
                _count_check(
                    season,
                    "player",
                    "duplicate_player_rows",
                    _duplicates(
                        season_players, ["gameId", "personId"]
                    ),
                ),
            ]
        )
    return pd.DataFrame(records)


def _missingness_record(
    season: str,
    split: str,
    metric_type: str,
    field: str,
    expected: int,
    observed: int,
    missing: int,
    availability: str,
) -> dict:
    return {
        "season": season,
        "split": split,
        "metric_type": metric_type,
        "field_or_check": field,
        "availability": availability,
        "expected_count": expected,
        "observed_count": observed,
        "missing_count": missing,
        "missingness_pct": (
            round(100 * missing / expected, 2)
            if expected and availability == "available"
            else 0.0
        ),
    }


def _row_check(
    season: str,
    split: str,
    check: str,
    expected: int,
    observed: int,
) -> dict:
    return _missingness_record(
        season,
        split,
        "row_completeness",
        check,
        expected,
        observed,
        max(expected - observed, 0),
        "available",
    )


def _count_check(
    season: str,
    split: str,
    check: str,
    count: int,
) -> dict:
    return {
        "season": season,
        "split": split,
        "metric_type": "duplicate_check",
        "field_or_check": check,
        "availability": "available",
        "expected_count": 0,
        "observed_count": count,
        "missing_count": count,
        "missingness_pct": 0.0,
    }


def cache_size_bytes(cache_dir: str | Path) -> int:
    return sum(
        path.stat().st_size
        for path in Path(cache_dir).glob("*/*.json")
        if path.is_file()
    )


def estimate_regular_season_games(
    game_log_dir: str | Path,
) -> int:
    game_ids = set()
    root = Path(game_log_dir)
    for season in BACKFILL_SEASONS:
        for path in root.glob(
            f"team_game_log_{season}_regular_season_*.csv"
        ):
            frame = pd.read_csv(path).rename(
                columns={"Game_ID": "GAME_ID"}
            )
            if "GAME_ID" in frame:
                game_ids.update(
                    (season, str(game_id))
                    for game_id in frame["GAME_ID"].dropna()
                )
    return len(game_ids)


def modeling_readiness(
    ledger: pd.DataFrame,
    missingness: pd.DataFrame,
) -> tuple[str, str]:
    if ledger.empty:
        return "not ready", "No playoff games were discovered."
    successful = ledger["quality_class"].isin(SUCCESS_CLASSES)
    core = missingness[
        missingness["metric_type"].eq("field_missingness")
        & missingness["availability"].eq("available")
    ]
    duplicate_count = int(
        ledger["duplicate_team_rows"].sum()
        + ledger["duplicate_player_rows"].sum()
    )
    if (
        successful.all()
        and (core.empty or core["missingness_pct"].max() == 0)
        and duplicate_count == 0
    ):
        return (
            "ready for offline feature experiments",
            "All discovered playoff games are cached with complete core V3 fields and unique team/player rows.",
        )
    return (
        "not ready",
        "Resolve failed games, missing core fields, or duplicate rows before modeling experiments.",
    )


def write_report(
    ledger: pd.DataFrame,
    missingness: pd.DataFrame,
    team_rows: pd.DataFrame,
    player_rows: pd.DataFrame,
    *,
    game_log_dir: str | Path,
    cache_dir: str | Path,
    output_path: str | Path = DEFAULT_DOC_PATH,
) -> None:
    by_season = (
        ledger.groupby("season")
        .agg(
            games_discovered=("game_id", "size"),
            games_cached=(
                "quality_class",
                lambda values: int((values == "cache_hit").sum()),
            ),
            games_fetched=(
                "quality_class",
                lambda values: int((values == "success").sum()),
            ),
            team_rows=("team_rows_count", "sum"),
            player_rows=("player_rows_count", "sum"),
            retries=("retries_used", "sum"),
        )
        .reset_index()
    )
    successful_requests = ledger[
        ledger["quality_class"].eq("success")
        & ledger["request_seconds"].gt(0)
    ]
    average_seconds = (
        float(successful_requests["request_seconds"].mean())
        if not successful_requests.empty
        else 0.0
    )
    regular_games = estimate_regular_season_games(game_log_dir)
    readiness, readiness_reason = modeling_readiness(
        ledger, missingness
    )
    failures = ledger[
        ~ledger["quality_class"].isin(SUCCESS_CLASSES)
    ]["quality_class"].value_counts()
    core_missing = missingness[
        missingness["metric_type"].eq("field_missingness")
        & missingness["missing_count"].gt(0)
    ]

    lines = [
        "# BoxScoreAdvancedV3 Playoff Backfill",
        "",
        "Standalone historical data ingestion only. No model, prediction, or Streamlit behavior was changed.",
        "",
        "## Coverage By Season",
        "",
        "| Season | Games | Cache hits | Newly fetched | Team rows | Player rows | Retries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_season.itertuples(index=False):
        lines.append(
            f"| {row.season} | {row.games_discovered} | "
            f"{row.games_cached} | {row.games_fetched} | "
            f"{row.team_rows} | {row.player_rows} | {row.retries} |"
        )
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Games discovered: {len(ledger)}",
            f"- Games already cached: {int(ledger['quality_class'].eq('cache_hit').sum())}",
            f"- Games newly fetched: {int(ledger['quality_class'].eq('success').sum())}",
            f"- Total successful games: {int(ledger['quality_class'].isin(SUCCESS_CLASSES).sum())}",
            f"- Retries used: {int(ledger['retries_used'].sum())}",
            f"- Team rows: {len(team_rows)}",
            f"- Player rows: {len(player_rows)}",
            f"- Duplicate team rows: {_duplicates(team_rows, ['gameId', 'teamId'])}",
            f"- Duplicate player rows: {_duplicates(player_rows, ['gameId', 'personId'])}",
            f"- Average successful request time: {average_seconds:.2f} seconds",
            f"- Raw JSON cache size: {cache_size_bytes(cache_dir) / 1024 / 1024:.1f} MB",
            "",
            "## Failures By Type",
            "",
        ]
    )
    if failures.empty:
        lines.append("- None.")
    else:
        for failure_type, count in failures.items():
            lines.append(f"- {failure_type}: {count}")
    lines.extend(["", "## Missingness", ""])
    if core_missing.empty:
        lines.append(
            "- No missing values across available core team or player V3 fields."
        )
    else:
        for row in core_missing.itertuples(index=False):
            lines.append(
                f"- {row.season} {row.split} `{row.field_or_check}`: "
                f"{row.missingness_pct:.2f}% missing"
            )
    lines.extend(
        [
            "- `plusMinus` is unavailable in BoxScoreAdvancedV3 and is explicitly marked `unavailable_in_v3`; it is not counted as a failed or missing field.",
            "",
            "## Regular-Season Cost Estimate",
            "",
            f"- Historical regular-season games discovered: {regular_games}",
            f"- Estimated serial runtime: {(regular_games * (average_seconds + REQUEST_DELAY_SECONDS)) / 3600:.1f} hours",
            "- Estimate assumes the measured successful request time plus the configured courtesy delay and excludes retry overhead.",
            "",
            "## Modeling Readiness",
            "",
            f"- Status: **{readiness}**",
            f"- Reason: {readiness_reason}",
            "- Recommended next experiment: `team_advanced_recent_deltas_experiment.py` using strictly pregame rolling team-level V3 statistics.",
            "- Player rotation strength should follow later because it requires minute/identity aggregation; V3 does not provide plus-minus.",
        ]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_backfill(
    *,
    game_log_dir: str | Path = DEFAULT_GAME_LOG_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    fetcher: Callable[[str, int], dict] = fetch_v3_payload,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    systemic_failure_threshold: int = SYSTEMIC_FAILURE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = discover_playoff_games(game_log_dir, BACKFILL_SEASONS)
    rows = []
    consecutive_failures = 0
    circuit_reason = ""
    for index, game in games.iterrows():
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(games):
            print(
                f"[{index + 1}/{len(games)}] "
                f"{game['SEASON']} {game['GAME_ID']}"
            )
        path = cache_path_for_game(
            cache_dir, game["SEASON"], game["GAME_ID"]
        )
        if (
            consecutive_failures >= systemic_failure_threshold
            and load_cached_payload(path) is None
        ):
            rows.append(
                _circuit_breaker_row(
                    game, cache_dir, circuit_reason
                )
            )
            continue

        result = cache_game_payload(
            season=game["SEASON"],
            game_id=game["GAME_ID"],
            cache_dir=cache_dir,
            fetcher=fetcher,
            sleeper=sleeper,
            clock=clock,
        )
        row = _ledger_row(game, result)
        rows.append(row)
        if row["quality_class"] in SUCCESS_CLASSES:
            consecutive_failures = 0
            circuit_reason = ""
        else:
            consecutive_failures += 1
            circuit_reason = (
                f"Stopped after {consecutive_failures} consecutive "
                f"uncached failures. Last class: "
                f"{row['quality_class']}; {row['error']}"
            )

    ledger = pd.DataFrame(rows)
    team_rows, player_rows = materialize_rows(ledger)
    missingness = build_missingness_report(
        ledger, team_rows, player_rows
    )
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(
        processed_dir / "advanced_boxscore_v3_playoff_backfill.csv",
        index=False,
    )
    missingness.to_csv(
        processed_dir / "advanced_boxscore_v3_playoff_missingness.csv",
        index=False,
    )
    team_rows.to_csv(
        processed_dir / "advanced_boxscore_v3_team_rows.csv",
        index=False,
    )
    player_rows.to_csv(
        processed_dir / "advanced_boxscore_v3_player_rows.csv",
        index=False,
    )
    write_report(
        ledger,
        missingness,
        team_rows,
        player_rows,
        game_log_dir=game_log_dir,
        cache_dir=cache_dir,
        output_path=doc_path,
    )
    return ledger, missingness, team_rows, player_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical playoff BoxScoreAdvancedV3 data"
    )
    parser.add_argument("--game-log-dir", default=DEFAULT_GAME_LOG_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--doc-path", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()
    ledger, missingness, _, _ = run_backfill(
        game_log_dir=args.game_log_dir,
        cache_dir=args.cache_dir,
        processed_dir=args.processed_dir,
        doc_path=args.doc_path,
    )
    print("\nQuality classes:")
    print(ledger["quality_class"].value_counts().to_string())
    print("\nModeling readiness:")
    print(modeling_readiness(ledger, missingness)[0])


if __name__ == "__main__":
    main()
