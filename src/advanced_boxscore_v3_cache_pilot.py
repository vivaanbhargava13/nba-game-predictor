from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv3

PILOT_SEASONS = ("2023-24", "2024-25")
DEFAULT_GAME_LOG_DIR = Path("data/raw")
DEFAULT_CACHE_DIR = Path("data/raw/boxscore_advanced_v3")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_DOC_PATH = Path("docs/advanced_boxscore_v3_cache_pilot.md")

MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_DELAY_SECONDS = 0.75
SYSTEMIC_FAILURE_THRESHOLD = 5

TEAM_FIELDS = [
    "offensiveRating",
    "defensiveRating",
    "netRating",
    "pace",
    "possessions",
    "trueShootingPercentage",
    "effectiveFieldGoalPercentage",
    "assistRatio",
    "offensiveReboundPercentage",
    "defensiveReboundPercentage",
    "reboundPercentage",
    "turnoverRatio",
    "PIE",
]
PLAYER_FIELDS = [
    "minutes",
    "usagePercentage",
    "offensiveRating",
    "defensiveRating",
    "netRating",
    "trueShootingPercentage",
    "PIE",
]
UNAVAILABLE_PLAYER_FIELDS = ["plusMinus"]


def canonical_game_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10)


def discover_playoff_games(
    game_log_dir: str | Path = DEFAULT_GAME_LOG_DIR,
    seasons: tuple[str, ...] = PILOT_SEASONS,
) -> pd.DataFrame:
    game_log_dir = Path(game_log_dir)
    frames = []
    for season in seasons:
        for path in sorted(
            game_log_dir.glob(f"team_game_log_{season}_playoffs_*.csv")
        ):
            frame = pd.read_csv(path).rename(
                columns={"Game_ID": "GAME_ID", "Team_ID": "TEAM_ID"}
            )
            if "GAME_ID" not in frame:
                continue
            columns = [
                column
                for column in ["GAME_ID", "GAME_DATE", "TEAM_ID"]
                if column in frame
            ]
            selected = frame[columns].copy()
            selected["SEASON"] = season
            frames.append(selected)
    if not frames:
        return pd.DataFrame(
            columns=["SEASON", "GAME_ID", "GAME_DATE", "TEAM_ROWS"]
        )

    games = pd.concat(frames, ignore_index=True)
    games["GAME_ID"] = games["GAME_ID"].map(canonical_game_id)
    games["GAME_DATE"] = pd.to_datetime(
        games.get("GAME_DATE"), format="%b %d, %Y", errors="coerce"
    )
    return (
        games.groupby(["SEASON", "GAME_ID"], as_index=False)
        .agg(
            GAME_DATE=("GAME_DATE", "min"),
            TEAM_ROWS=("TEAM_ID", "nunique"),
        )
        .sort_values(["SEASON", "GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )


def cache_path_for_game(
    cache_dir: str | Path,
    season: str,
    game_id: str,
) -> Path:
    return Path(cache_dir) / season / f"{game_id}.json"


def parse_v3_payload(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    boxscore = payload.get("boxScoreAdvanced")
    if not isinstance(boxscore, dict):
        return pd.DataFrame(), pd.DataFrame()

    team_rows = []
    player_rows = []
    game_id = boxscore.get("gameId")
    for side in ("awayTeam", "homeTeam"):
        team = boxscore.get(side)
        if not isinstance(team, dict):
            continue
        team_metadata = {
            "gameId": game_id,
            "teamId": team.get("teamId"),
            "teamCity": team.get("teamCity"),
            "teamName": team.get("teamName"),
            "teamTricode": team.get("teamTricode"),
            "homeAway": "home" if side == "homeTeam" else "away",
        }
        statistics = team.get("statistics")
        if isinstance(statistics, dict):
            team_rows.append({**team_metadata, **statistics})

        players = team.get("players")
        if not isinstance(players, list):
            continue
        for player in players:
            if not isinstance(player, dict):
                continue
            player_metadata = {
                **team_metadata,
                "personId": player.get("personId"),
                "firstName": player.get("firstName"),
                "familyName": player.get("familyName"),
                "position": player.get("position"),
                "comment": player.get("comment"),
            }
            player_statistics = player.get("statistics")
            if isinstance(player_statistics, dict):
                player_rows.append(
                    {**player_metadata, **player_statistics}
                )
            else:
                player_rows.append(player_metadata)
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)


def validate_payload(payload: object) -> tuple[bool, str, str]:
    if not isinstance(payload, dict) or not payload:
        return False, "empty_response", "response is empty"
    if not isinstance(payload.get("boxScoreAdvanced"), dict):
        return (
            False,
            "malformed_response",
            "boxScoreAdvanced object is missing",
        )
    team_rows, player_rows = parse_v3_payload(payload)
    if len(team_rows) != 2:
        return (
            False,
            "missing_team_rows",
            f"expected 2 team rows, received {len(team_rows)}",
        )
    if player_rows.empty:
        return False, "missing_player_rows", "player rows are empty"
    return True, "", ""


def load_cached_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    valid, _, _ = validate_payload(payload)
    return payload if valid else None


def _write_payload_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def fetch_v3_payload(
    game_id: str,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> dict:
    response = boxscoreadvancedv3.BoxScoreAdvancedV3(
        game_id=game_id,
        timeout=timeout_seconds,
    )
    return response.get_dict()


def cache_game_payload(
    *,
    season: str,
    game_id: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    fetcher: Callable[[str, int], dict] = fetch_v3_payload,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    max_retries: int = MAX_RETRIES,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> dict:
    path = cache_path_for_game(cache_dir, season, game_id)
    cached_payload = load_cached_payload(path)
    if cached_payload is not None:
        return {
            "status": "already_cached",
            "cached_or_fetched": "cached",
            "payload": cached_payload,
            "attempt_count": 0,
            "request_seconds": 0.0,
            "error_type": "",
            "error": "",
            "cache_path": str(path),
        }

    stale_payload = None
    if path.exists():
        try:
            stale_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stale_payload = None

    started = clock()
    attempts = 0
    error_type = ""
    error = ""
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            payload = fetcher(game_id, timeout_seconds)
            valid, error_type, error = validate_payload(payload)
            if not valid:
                raise ValueError(error)
            _write_payload_atomic(path, payload)
            elapsed = clock() - started
            sleeper(request_delay_seconds)
            return {
                "status": "fetched",
                "cached_or_fetched": "fetched",
                "payload": payload,
                "attempt_count": attempts,
                "request_seconds": elapsed,
                "error_type": "",
                "error": "",
                "cache_path": str(path),
            }
        except Exception as exc:
            if not error_type:
                error_type = type(exc).__name__
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                sleeper(request_delay_seconds * (2 ** (attempt - 1)))

    elapsed = clock() - started
    if stale_payload is not None:
        valid, _, _ = validate_payload(stale_payload)
        if valid:
            return {
                "status": "stale_cache",
                "cached_or_fetched": "cached",
                "payload": stale_payload,
                "attempt_count": attempts,
                "request_seconds": elapsed,
                "error_type": error_type,
                "error": error,
                "cache_path": str(path),
            }
    return {
        "status": "failed",
        "cached_or_fetched": "none",
        "payload": None,
        "attempt_count": attempts,
        "request_seconds": elapsed,
        "error_type": error_type,
        "error": error,
        "cache_path": str(path),
    }


def _duplicate_count(
    frame: pd.DataFrame,
    columns: list[str],
) -> int:
    if frame.empty or not set(columns).issubset(frame.columns):
        return 0
    return int(frame.duplicated(columns, keep=False).sum())


def _ledger_row(game: pd.Series, result: dict) -> dict:
    payload = result["payload"]
    team_rows, player_rows = (
        parse_v3_payload(payload)
        if payload is not None
        else (pd.DataFrame(), pd.DataFrame())
    )
    return {
        "season": game["SEASON"],
        "game_id": game["GAME_ID"],
        "game_date": game["GAME_DATE"],
        "status": result["status"],
        "cached_or_fetched": result["cached_or_fetched"],
        "attempt_count": result["attempt_count"],
        "error_type": result["error_type"],
        "error": result["error"],
        "team_rows_count": len(team_rows),
        "player_rows_count": len(player_rows),
        "missing_team_rows": max(2 - len(team_rows), 0),
        "missing_player_rows": int(player_rows.empty),
        "duplicate_team_rows": _duplicate_count(
            team_rows, ["gameId", "teamId"]
        ),
        "duplicate_player_rows": _duplicate_count(
            player_rows, ["gameId", "teamId", "personId"]
        ),
        "request_seconds": round(float(result["request_seconds"]), 4),
        "cache_path": result["cache_path"],
    }


def _skipped_row(
    game: pd.Series,
    cache_dir: str | Path,
    reason: str,
) -> dict:
    return {
        "season": game["SEASON"],
        "game_id": game["GAME_ID"],
        "game_date": game["GAME_DATE"],
        "status": "skipped_systemic_failure",
        "cached_or_fetched": "none",
        "attempt_count": 0,
        "error_type": "circuit_breaker",
        "error": reason,
        "team_rows_count": 0,
        "player_rows_count": 0,
        "missing_team_rows": 2,
        "missing_player_rows": 1,
        "duplicate_team_rows": 0,
        "duplicate_player_rows": 0,
        "request_seconds": 0.0,
        "cache_path": str(
            cache_path_for_game(
                cache_dir, game["SEASON"], game["GAME_ID"]
            )
        ),
    }


def build_missingness_report(ledger: pd.DataFrame) -> pd.DataFrame:
    records = []
    successful_statuses = {
        "fetched",
        "already_cached",
        "stale_cache",
    }
    seasons = (
        sorted(ledger["season"].dropna().astype(str).unique())
        if "season" in ledger
        else []
    )
    for season in seasons:
        season_ledger = ledger[ledger["season"].eq(season)]
        successful = season_ledger[
            season_ledger["status"].isin(successful_statuses)
        ]
        parsed = []
        for path in successful["cache_path"]:
            payload = load_cached_payload(Path(path))
            if payload is not None:
                parsed.append(parse_v3_payload(payload))

        for split, fields, position in [
            ("team", TEAM_FIELDS, 0),
            ("player", PLAYER_FIELDS + UNAVAILABLE_PLAYER_FIELDS, 1),
        ]:
            frames = [pair[position] for pair in parsed if not pair[position].empty]
            combined = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame()
            )
            for field in fields:
                expected = len(combined)
                present = (
                    int(combined[field].notna().sum())
                    if field in combined
                    else 0
                )
                missing = expected - present if expected else 0
                records.append(
                    {
                        "season": season,
                        "split": split,
                        "metric_type": "field_missingness",
                        "field_or_check": field,
                        "expected_count": expected,
                        "observed_count": present,
                        "missing_count": missing,
                        "missingness_pct": (
                            round(100 * missing / expected, 2)
                            if expected
                            else 100.0
                        ),
                    }
                )

        expected_team_rows = len(season_ledger) * 2
        observed_team_rows = int(season_ledger["team_rows_count"].sum())
        records.extend(
            [
                _quality_record(
                    season,
                    "team",
                    "missing_team_rows",
                    expected_team_rows,
                    observed_team_rows,
                ),
                _quality_record(
                    season,
                    "player",
                    "games_missing_player_rows",
                    len(season_ledger),
                    int(
                        season_ledger["player_rows_count"].gt(0).sum()
                    ),
                ),
                _count_record(
                    season,
                    "response",
                    "malformed_responses",
                    int(
                        season_ledger["error_type"]
                        .eq("malformed_response")
                        .sum()
                    ),
                ),
                _count_record(
                    season,
                    "response",
                    "empty_responses",
                    int(
                        season_ledger["error_type"]
                        .eq("empty_response")
                        .sum()
                    ),
                ),
                _count_record(
                    season,
                    "team",
                    "duplicate_team_rows",
                    int(season_ledger["duplicate_team_rows"].sum()),
                ),
                _count_record(
                    season,
                    "player",
                    "duplicate_player_rows",
                    int(season_ledger["duplicate_player_rows"].sum()),
                ),
            ]
        )
    return pd.DataFrame(records)


def _quality_record(
    season: str,
    split: str,
    check: str,
    expected: int,
    observed: int,
) -> dict:
    missing = max(expected - observed, 0)
    return {
        "season": season,
        "split": split,
        "metric_type": "row_completeness",
        "field_or_check": check,
        "expected_count": expected,
        "observed_count": observed,
        "missing_count": missing,
        "missingness_pct": (
            round(100 * missing / expected, 2) if expected else 100.0
        ),
    }


def _count_record(
    season: str,
    split: str,
    check: str,
    count: int,
) -> dict:
    return {
        "season": season,
        "split": split,
        "metric_type": "response_quality",
        "field_or_check": check,
        "expected_count": 0,
        "observed_count": count,
        "missing_count": count,
        "missingness_pct": 0.0,
    }


def estimate_game_counts(
    game_log_dir: str | Path,
) -> tuple[int, int]:
    seasons = tuple(
        f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2025)
    )
    playoff_games = len(discover_playoff_games(game_log_dir, seasons))
    all_game_ids = set()
    game_log_dir = Path(game_log_dir)
    for season in seasons:
        for season_type in ("regular_season", "playoffs"):
            for path in game_log_dir.glob(
                f"team_game_log_{season}_{season_type}_*.csv"
            ):
                frame = pd.read_csv(path).rename(
                    columns={"Game_ID": "GAME_ID"}
                )
                if "GAME_ID" in frame:
                    all_game_ids.update(
                        (season, canonical_game_id(game_id))
                        for game_id in frame["GAME_ID"].dropna()
                    )
    return playoff_games, len(all_game_ids)


def recommend_backfill(
    ledger: pd.DataFrame,
    missingness: pd.DataFrame,
) -> tuple[str, str]:
    if ledger.empty:
        return "no backfill", "No pilot games were discovered."
    successful = ledger["status"].isin(
        ["fetched", "already_cached", "stale_cache"]
    )
    success_rate = float(successful.mean())
    core_fields = missingness[
        missingness["metric_type"].eq("field_missingness")
        & ~missingness["field_or_check"].isin(UNAVAILABLE_PLAYER_FIELDS)
    ]
    worst_core_missingness = (
        float(core_fields["missingness_pct"].max())
        if not core_fields.empty
        else 100.0
    )
    duplicate_rows = int(
        ledger["duplicate_team_rows"].sum()
        + ledger["duplicate_player_rows"].sum()
    )
    if (
        success_rate >= 0.99
        and worst_core_missingness <= 1.0
        and duplicate_rows == 0
    ):
        return (
            "full playoff backfill",
            "V3 is highly reliable for playoff games and core advanced fields are stable; validate a regular-season sample before expanding further.",
        )
    if success_rate >= 0.90 and worst_core_missingness <= 5.0:
        return (
            "small backfill only",
            "V3 is usable but needs a larger stratified reliability check before a complete historical ingestion.",
        )
    return (
        "no backfill",
        "Pilot success or core-field completeness is too low for historical ingestion.",
    )


def _sample_rows(
    ledger: pd.DataFrame,
) -> tuple[dict, dict]:
    successful = ledger[
        ledger["status"].isin(
            ["fetched", "already_cached", "stale_cache"]
        )
    ]
    if successful.empty:
        return {}, {}
    payload = load_cached_payload(Path(successful.iloc[0]["cache_path"]))
    if payload is None:
        return {}, {}
    team_rows, player_rows = parse_v3_payload(payload)
    team_columns = ["teamTricode", *TEAM_FIELDS]
    player_columns = [
        "teamTricode",
        "personId",
        "firstName",
        "familyName",
        *PLAYER_FIELDS,
    ]
    team_sample = (
        team_rows.iloc[0][
            [column for column in team_columns if column in team_rows]
        ].to_dict()
        if not team_rows.empty
        else {}
    )
    player_sample = (
        player_rows.iloc[0][
            [column for column in player_columns if column in player_rows]
        ].to_dict()
        if not player_rows.empty
        else {}
    )
    return team_sample, player_sample


def write_report(
    ledger: pd.DataFrame,
    missingness: pd.DataFrame,
    *,
    game_log_dir: str | Path,
    output_path: str | Path = DEFAULT_DOC_PATH,
) -> None:
    successful = ledger[
        ledger["status"].isin(
            ["fetched", "already_cached", "stale_cache"]
        )
    ]
    successful_requests = ledger[
        ledger["status"].eq("fetched")
        & ledger["request_seconds"].gt(0)
    ]
    average_seconds = (
        float(successful_requests["request_seconds"].mean())
        if not successful_requests.empty
        else 0.0
    )
    playoff_count, all_game_count = estimate_game_counts(game_log_dir)
    recommendation, rationale = recommend_backfill(
        ledger, missingness
    )
    team_sample, player_sample = _sample_rows(ledger)
    failed = int(ledger["status"].eq("failed").sum())
    attempted = int(ledger["attempt_count"].gt(0).sum())
    failure_rate = failed / attempted if attempted else 0.0

    lines = [
        "# BoxScoreAdvancedV3 Cache Pilot",
        "",
        "This is a cache and schema reliability pilot only. It does not train or deploy a model.",
        "",
        "## Results",
        "",
        f"- Games discovered: {len(ledger)}",
        f"- Games fetched: {int(ledger['status'].eq('fetched').sum())}",
        f"- Games already cached: {int(ledger['status'].eq('already_cached').sum())}",
        f"- Failures: {failed}",
        f"- Circuit-breaker skips: {int(ledger['status'].eq('skipped_systemic_failure').sum())}",
        f"- Failure rate among attempted games: {failure_rate:.2%}",
        f"- Average successful request time: {average_seconds:.2f} seconds",
        f"- Team rows parsed: {int(successful['team_rows_count'].sum())}",
        f"- Player rows parsed: {int(successful['player_rows_count'].sum())}",
        "",
        "## Schema Notes",
        "",
        f"- Stable team fields audited: {', '.join(f'`{field}`' for field in TEAM_FIELDS)}",
        f"- Stable player fields audited: {', '.join(f'`{field}`' for field in PLAYER_FIELDS)}",
        "- `plusMinus` is not present in BoxScoreAdvancedV3 and would require a separate traditional box-score join.",
        "",
        "## Sample Parsed Team Row",
        "",
        "```json",
        json.dumps(team_sample, indent=2, default=str),
        "```",
        "",
        "## Sample Parsed Player Row",
        "",
        "```json",
        json.dumps(player_sample, indent=2, default=str),
        "```",
        "",
        "## Backfill Estimate",
        "",
        f"- Cached historical playoff games, 2015-16 through 2024-25: {playoff_count}",
        f"- Cached regular-season plus playoff games: {all_game_count}",
        f"- Estimated playoff runtime: {(playoff_count * (average_seconds + REQUEST_DELAY_SECONDS)) / 60:.1f} minutes",
        f"- Estimated regular-season plus playoff runtime: {(all_game_count * (average_seconds + REQUEST_DELAY_SECONDS)) / 60 / 60:.1f} hours",
        "",
        "## Recommendation",
        "",
        f"- Recommendation: **{recommendation}**",
        f"- Rationale: {rationale}",
        "- Next experiment after a successful backfill: team advanced recent deltas first; player rotation strength second because it requires identity/minutes aggregation and a separate plus-minus source.",
        "",
        "No production model or application behavior was changed.",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(
    *,
    game_log_dir: str | Path = DEFAULT_GAME_LOG_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    fetcher: Callable[[str, int], dict] = fetch_v3_payload,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    systemic_failure_threshold: int = SYSTEMIC_FAILURE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = discover_playoff_games(game_log_dir)
    rows = []
    consecutive_failures = 0
    circuit_reason = ""
    for index, game in games.iterrows():
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(games):
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
            rows.append(_skipped_row(game, cache_dir, circuit_reason))
            continue

        result = cache_game_payload(
            season=game["SEASON"],
            game_id=game["GAME_ID"],
            cache_dir=cache_dir,
            fetcher=fetcher,
            sleeper=sleeper,
            clock=clock,
        )
        rows.append(_ledger_row(game, result))
        if result["status"] == "failed":
            consecutive_failures += 1
            circuit_reason = (
                f"Stopped after {consecutive_failures} consecutive "
                f"uncached V3 failures. Last error: {result['error']}"
            )
        else:
            consecutive_failures = 0
            circuit_reason = ""

    ledger = pd.DataFrame(rows)
    missingness = build_missingness_report(ledger)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(
        processed_dir / "advanced_boxscore_v3_cache_pilot.csv",
        index=False,
    )
    missingness.to_csv(
        processed_dir / "advanced_boxscore_v3_missingness.csv",
        index=False,
    )
    write_report(
        ledger,
        missingness,
        game_log_dir=game_log_dir,
        output_path=doc_path,
    )
    return ledger, missingness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache and audit BoxScoreAdvancedV3 playoff data"
    )
    parser.add_argument("--game-log-dir", default=DEFAULT_GAME_LOG_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--doc-path", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()
    ledger, missingness = run_pilot(
        game_log_dir=args.game_log_dir,
        cache_dir=args.cache_dir,
        processed_dir=args.processed_dir,
        doc_path=args.doc_path,
    )
    print("\nFetch summary:")
    print(ledger["status"].value_counts(dropna=False).to_string())
    print("\nRecommendation:")
    print(recommend_backfill(ledger, missingness)[0])


if __name__ == "__main__":
    main()
