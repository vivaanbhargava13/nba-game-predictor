from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv2

PILOT_SEASONS = ("2023-24", "2024-25")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CACHE_DIR = Path("data/cache/advanced_boxscore")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_DOC_PATH = Path("docs/advanced_boxscore_cache_pilot.md")

MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_DELAY_SECONDS = 0.75
SYSTEMIC_FAILURE_THRESHOLD = 3

TEAM_FIELDS = [
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TOV",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "PACE",
    "POSS",
    "PIE",
]
PLAYER_FIELDS = [
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TOV",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PACE",
    "POSS",
    "PIE",
]


def canonical_game_id(value: object) -> str:
    """Return the zero-padded NBA game ID used by box-score endpoints."""
    text = str(value).strip().replace(".0", "")
    return text.zfill(10)


def discover_playoff_games(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    seasons: tuple[str, ...] = PILOT_SEASONS,
) -> pd.DataFrame:
    """Discover unique playoff games from existing cached team game logs."""
    raw_dir = Path(raw_dir)
    rows: list[pd.DataFrame] = []
    for season in seasons:
        for path in sorted(
            raw_dir.glob(f"team_game_log_{season}_playoffs_*.csv")
        ):
            frame = pd.read_csv(path)
            frame = frame.rename(
                columns={"Game_ID": "GAME_ID", "Team_ID": "TEAM_ID"}
            )
            if "GAME_ID" not in frame:
                continue
            keep = [
                column
                for column in [
                    "GAME_ID",
                    "GAME_DATE",
                    "TEAM_ID",
                    "MATCHUP",
                ]
                if column in frame
            ]
            selected = frame[keep].copy()
            selected["SEASON"] = season
            rows.append(selected)

    if not rows:
        return pd.DataFrame(
            columns=["SEASON", "GAME_ID", "GAME_DATE", "TEAM_ROWS"]
        )

    games = pd.concat(rows, ignore_index=True)
    games["GAME_ID"] = games["GAME_ID"].map(canonical_game_id)
    if "GAME_DATE" in games:
        games["GAME_DATE"] = pd.to_datetime(
            games["GAME_DATE"], format="%b %d, %Y", errors="coerce"
        )
    grouped = (
        games.groupby(["SEASON", "GAME_ID"], as_index=False)
        .agg(
            GAME_DATE=("GAME_DATE", "min"),
            TEAM_ROWS=("TEAM_ID", "nunique"),
        )
        .sort_values(["SEASON", "GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )
    return grouped


def cache_path_for_game(
    cache_dir: str | Path,
    season: str,
    game_id: str,
) -> Path:
    return Path(cache_dir) / season / "playoffs" / f"{game_id}.json"


def _result_sets(payload: dict) -> dict[str, pd.DataFrame]:
    result_sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    tables: dict[str, pd.DataFrame] = {}
    for result in result_sets:
        name = result.get("name")
        headers = result.get("headers", [])
        rows = result.get("rowSet", [])
        if name:
            tables[name] = pd.DataFrame(rows, columns=headers)
    return tables


def parse_advanced_boxscore(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = _result_sets(payload)
    return (
        tables.get("TeamStats", pd.DataFrame()),
        tables.get("PlayerStats", pd.DataFrame()),
    )


def validate_payload(payload: dict) -> tuple[bool, str]:
    team_stats, player_stats = parse_advanced_boxscore(payload)
    if team_stats.empty:
        return False, "TeamStats is empty"
    if len(team_stats) != 2:
        return False, f"expected 2 TeamStats rows, received {len(team_stats)}"
    if player_stats.empty:
        return False, "PlayerStats is empty"
    return True, ""


def load_cached_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    valid, _ = validate_payload(payload)
    return payload if valid else None


def _write_payload_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def fetch_advanced_boxscore(
    game_id: str,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> dict:
    try:
        response = boxscoreadvancedv2.BoxScoreAdvancedV2(
            game_id=game_id,
            timeout=timeout_seconds,
        )
        return response.get_dict()
    except KeyError as exc:
        raise RuntimeError(
            "BoxScoreAdvancedV2 returned no result sets"
        ) from exc


def cache_game_payload(
    *,
    season: str,
    game_id: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    fetcher: Callable[[str, int], dict] = fetch_advanced_boxscore,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    max_retries: int = MAX_RETRIES,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> dict:
    """Load a valid cache or fetch and atomically cache one game response."""
    path = cache_path_for_game(cache_dir, season, game_id)
    cached_payload = load_cached_payload(path)
    if cached_payload is not None:
        return {
            "status": "already_cached",
            "payload": cached_payload,
            "request_seconds": 0.0,
            "attempts": 0,
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
    last_error = ""
    attempts = 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            payload = fetcher(game_id, timeout_seconds)
            valid, reason = validate_payload(payload)
            if not valid:
                raise ValueError(reason)
            _write_payload_atomic(path, payload)
            elapsed = clock() - started
            sleeper(request_delay_seconds)
            return {
                "status": "fetched",
                "payload": payload,
                "request_seconds": elapsed,
                "attempts": attempts,
                "error": "",
                "cache_path": str(path),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                sleeper(request_delay_seconds * (2 ** (attempt - 1)))

    elapsed = clock() - started
    if stale_payload is not None:
        valid, _ = validate_payload(stale_payload)
        if valid:
            return {
                "status": "stale_cache",
                "payload": stale_payload,
                "request_seconds": elapsed,
                "attempts": attempts,
                "error": last_error,
                "cache_path": str(path),
            }
    return {
        "status": "failed",
        "payload": None,
        "request_seconds": elapsed,
        "attempts": attempts,
        "error": last_error,
        "cache_path": str(path),
    }


def _field_counts(frame: pd.DataFrame, fields: list[str]) -> dict[str, int]:
    return {
        field: (
            int(frame[field].notna().sum())
            if field in frame
            else 0
        )
        for field in fields
    }


def _game_result_row(game: pd.Series, result: dict) -> dict:
    payload = result["payload"]
    team_stats, player_stats = (
        parse_advanced_boxscore(payload)
        if payload is not None
        else (pd.DataFrame(), pd.DataFrame())
    )
    team_counts = _field_counts(team_stats, TEAM_FIELDS)
    player_counts = _field_counts(player_stats, PLAYER_FIELDS)
    return {
        "season": game["SEASON"],
        "game_id": game["GAME_ID"],
        "game_date": game["GAME_DATE"],
        "status": result["status"],
        "attempts": result["attempts"],
        "request_seconds": round(float(result["request_seconds"]), 4),
        "team_rows": len(team_stats),
        "player_rows": len(player_stats),
        "team_fields_present": sum(count > 0 for count in team_counts.values()),
        "team_fields_expected": len(TEAM_FIELDS),
        "player_fields_present": sum(
            count > 0 for count in player_counts.values()
        ),
        "player_fields_expected": len(PLAYER_FIELDS),
        "cache_path": result["cache_path"],
        "error": result["error"],
    }


def _skipped_result(game: pd.Series, reason: str) -> dict:
    return {
        "season": game["SEASON"],
        "game_id": game["GAME_ID"],
        "game_date": game["GAME_DATE"],
        "status": "skipped_systemic_failure",
        "attempts": 0,
        "request_seconds": 0.0,
        "team_rows": 0,
        "player_rows": 0,
        "team_fields_present": 0,
        "team_fields_expected": len(TEAM_FIELDS),
        "player_fields_present": 0,
        "player_fields_expected": len(PLAYER_FIELDS),
        "cache_path": str(
            cache_path_for_game(
                DEFAULT_CACHE_DIR, game["SEASON"], game["GAME_ID"]
            )
        ),
        "error": reason,
    }


def build_missingness_report(
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    successful = ledger[
        ledger["status"].isin(["fetched", "already_cached", "stale_cache"])
    ]
    for season in PILOT_SEASONS:
        season_rows = successful[successful["season"].eq(season)]
        for split, fields in [
            ("team", TEAM_FIELDS),
            ("player", PLAYER_FIELDS),
        ]:
            total_rows = int(season_rows[f"{split}_rows"].sum())
            present_counts = {field: 0 for field in fields}
            for cache_path in season_rows["cache_path"]:
                payload = load_cached_payload(Path(cache_path))
                if payload is None:
                    continue
                team_stats, player_stats = parse_advanced_boxscore(payload)
                frame = team_stats if split == "team" else player_stats
                counts = _field_counts(frame, fields)
                for field, count in counts.items():
                    present_counts[field] += count
            for field in fields:
                present = present_counts[field]
                missing = max(total_rows - present, 0)
                rows.append(
                    {
                        "season": season,
                        "split": split,
                        "field": field,
                        "rows_expected": total_rows,
                        "rows_present": present,
                        "rows_missing": missing,
                        "missingness_pct": (
                            round(100 * missing / total_rows, 2)
                            if total_rows
                            else 100.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def estimate_full_backfill(
    raw_dir: str | Path,
    average_request_seconds: float,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> tuple[int, float]:
    all_games = discover_playoff_games(
        raw_dir,
        tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2025)),
    )
    game_count = len(all_games)
    seconds_per_game = max(average_request_seconds, 0) + request_delay_seconds
    return game_count, game_count * seconds_per_game


def _worth_full_ingestion(
    ledger: pd.DataFrame,
    missingness: pd.DataFrame,
) -> tuple[bool, str]:
    if ledger.empty:
        return False, "No games were discovered."
    successful = ledger["status"].isin(
        ["fetched", "already_cached", "stale_cache"]
    ).sum()
    coverage = successful / len(ledger)
    team_missingness = missingness[
        missingness["split"].eq("team")
    ]["missingness_pct"]
    worst_team_missingness = (
        float(team_missingness.max()) if not team_missingness.empty else 100.0
    )
    worth_ingestion = coverage >= 0.95 and worst_team_missingness <= 5.0
    if worth_ingestion:
        return (
            True,
            "Pilot coverage is at least 95% and requested team fields are at least 95% complete.",
        )
    return (
        False,
        "Pilot coverage or requested team-field completeness is below 95%; investigate failures before a full backfill.",
    )


def write_pilot_report(
    ledger: pd.DataFrame,
    missingness: pd.DataFrame,
    *,
    raw_dir: str | Path,
    output_path: str | Path = DEFAULT_DOC_PATH,
) -> None:
    attempted = int(ledger["attempts"].gt(0).sum())
    fetched = int(ledger["status"].eq("fetched").sum())
    cached = int(ledger["status"].eq("already_cached").sum())
    stale = int(ledger["status"].eq("stale_cache").sum())
    failures = int(ledger["status"].eq("failed").sum())
    skipped = int(ledger["status"].eq("skipped_systemic_failure").sum())
    fresh_requests = ledger[ledger["attempts"].gt(0)]
    average_request_seconds = (
        float(fresh_requests["request_seconds"].mean())
        if not fresh_requests.empty
        else 0.0
    )
    full_games, full_seconds = estimate_full_backfill(
        raw_dir, average_request_seconds
    )
    worth_ingestion, rationale = _worth_full_ingestion(ledger, missingness)
    missing_fields = missingness[
        missingness["missingness_pct"].gt(0)
    ][["season", "split", "field", "missingness_pct"]]

    lines = [
        "# Advanced Box Score Cache Pilot",
        "",
        "This pilot evaluates historical `BoxScoreAdvancedV2` availability only. It does not train or deploy a model.",
        "",
        "## Scope",
        "",
        "- Seasons: 2023-24 playoffs and 2024-25 playoffs",
        "- Source game IDs: existing cached team playoff game logs",
        "- Raw cache: one atomic endpoint JSON response per game",
        "- Endpoint splits: TeamStats and PlayerStats",
        "",
        "## Results",
        "",
        f"- Games discovered: {len(ledger)}",
        f"- Games attempted via API: {attempted}",
        f"- Games fetched: {fetched}",
        f"- Games already cached: {cached}",
        f"- Games served from valid stale cache: {stale}",
        f"- Failures: {failures}",
        f"- Games skipped after systemic V2 failure threshold: {skipped}",
        f"- Average request time: {average_request_seconds:.2f} seconds",
        "",
        "## Missing Fields",
        "",
    ]
    if missing_fields.empty:
        lines.append("- No missing requested fields in successful responses.")
    else:
        for row in missing_fields.itertuples(index=False):
            lines.append(
                f"- {row.season} {row.split} `{row.field}`: {row.missingness_pct:.2f}% missing"
            )
    lines.extend(
        [
            "",
            "## Full Backfill Estimate",
            "",
            f"- Historical playoff games found for 2015-16 through 2024-25: {full_games}",
            f"- Estimated successful endpoint calls required: {full_games}",
            f"- Estimated serial runtime: {full_seconds / 60:.1f} minutes",
            "- Direct API fee: none; practical cost is runtime, retries, cache storage, and stats.nba.com rate-limit exposure.",
            "- Estimate includes measured request time plus the configured courtesy delay; retries can increase it.",
            "",
            "## Source Decision",
            "",
            f"- Worth full ingestion: {'yes' if worth_ingestion else 'not yet'}",
            f"- Rationale: {rationale}",
            "- Compatibility diagnostic: the installed nba_api 1.10.2 V2 endpoint returned HTTP 200 with an empty JSON object for known playoff games, while a one-game V3 diagnostic returned populated team and player tables.",
            "",
            "This is a data-source ingestion decision, not a modeling or deployment recommendation.",
        ]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    fetcher: Callable[[str, int], dict] = fetch_advanced_boxscore,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    systemic_failure_threshold: int = SYSTEMIC_FAILURE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = discover_playoff_games(raw_dir)
    rows = []
    consecutive_uncached_failures = 0
    circuit_reason = ""
    for index, game in games.iterrows():
        print(
            f"[{index + 1}/{len(games)}] {game['SEASON']} "
            f"{game['GAME_ID']}"
        )
        path = cache_path_for_game(
            cache_dir, game["SEASON"], game["GAME_ID"]
        )
        if (
            consecutive_uncached_failures >= systemic_failure_threshold
            and load_cached_payload(path) is None
        ):
            skipped = _skipped_result(game, circuit_reason)
            skipped["cache_path"] = str(path)
            rows.append(skipped)
            continue
        result = cache_game_payload(
            season=game["SEASON"],
            game_id=game["GAME_ID"],
            cache_dir=cache_dir,
            fetcher=fetcher,
            sleeper=sleeper,
            clock=clock,
        )
        rows.append(_game_result_row(game, result))
        if result["status"] == "failed":
            consecutive_uncached_failures += 1
            circuit_reason = (
                "Skipped after "
                f"{consecutive_uncached_failures} consecutive uncached "
                f"BoxScoreAdvancedV2 failures. Last error: {result['error']}"
            )
        else:
            consecutive_uncached_failures = 0
            circuit_reason = ""

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        ledger = pd.DataFrame(
            columns=[
                "season",
                "game_id",
                "game_date",
                "status",
                "attempts",
                "request_seconds",
                "team_rows",
                "player_rows",
                "team_fields_present",
                "team_fields_expected",
                "player_fields_present",
                "player_fields_expected",
                "cache_path",
                "error",
            ]
        )
    missingness = build_missingness_report(ledger)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(
        processed_dir / "advanced_boxscore_cache_pilot.csv", index=False
    )
    missingness.to_csv(
        processed_dir / "advanced_boxscore_cache_pilot_missingness.csv",
        index=False,
    )
    write_pilot_report(
        ledger,
        missingness,
        raw_dir=raw_dir,
        output_path=doc_path,
    )
    return ledger, missingness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache and audit BoxScoreAdvancedV2 playoff responses"
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--doc-path", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()
    ledger, missingness = run_pilot(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        processed_dir=args.processed_dir,
        doc_path=args.doc_path,
    )
    print("\nFetch summary:")
    print(ledger["status"].value_counts(dropna=False).to_string())
    print("\nWorst requested-field missingness:")
    print(
        missingness.sort_values("missingness_pct", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
