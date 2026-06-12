from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2


LIVE_GAMES_CACHE_DIR = Path("data/cache/live_games")
LIVE_GAMES_CACHE_PATH = LIVE_GAMES_CACHE_DIR / "live_games.json"
DEFAULT_TTL_SECONDS = 600
EMPTY_FAILURE_TTL_SECONDS = 60
API_RETRIES = 2
API_TIMEOUT_SECONDS = 4
ESPN_TIMEOUT_SECONDS = 4
API_DELAY_SECONDS = 0.25
SCOREBOARD_SCAN_DAYS = 10
EASTERN_TZ = ZoneInfo("America/New_York")
TEAM_ABBR_ALIASES = {
    "NY": "NYK",
    "SA": "SAS",
    "GS": "GSW",
    "NO": "NOP",
    "PHO": "PHX",
}
TEAM_NAME_ALIASES = {
    "new york knicks": "NYK",
    "san antonio spurs": "SAS",
    "golden state warriors": "GSW",
    "new orleans pelicans": "NOP",
    "phoenix suns": "PHX",
}


def nba_season_from_date(value: date | datetime | str) -> str:
    game_date = pd.to_datetime(value).date()
    start_year = game_date.year if game_date.month >= 10 else game_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def canonical_team_abbr(source_abbr: Any, team_name: Any = "") -> str:
    abbreviation = str(source_abbr or "").strip().upper()
    if abbreviation in TEAM_ABBR_ALIASES:
        return TEAM_ABBR_ALIASES[abbreviation]

    normalized_name = " ".join(str(team_name or "").strip().lower().split())
    if normalized_name in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[normalized_name]

    return abbreviation


def _team_display_name(team: dict[str, Any]) -> str:
    for key in ("displayName", "shortDisplayName", "name"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    location = str(team.get("location") or "").strip()
    name = str(team.get("name") or "").strip()
    return f"{location} {name}".strip()


def _canonicalize_series_status(value: str) -> str:
    status = str(value or "")
    if not status:
        return ""
    for source, canonical in TEAM_ABBR_ALIASES.items():
        status = re.sub(rf"\b{re.escape(source)}\b", canonical, status, flags=re.IGNORECASE)
    return status


def _cache_is_fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) <= ttl_seconds


def _effective_cache_ttl(cached: dict[str, Any], ttl_seconds: int) -> int:
    has_games = bool(cached.get("latest")) or bool(cached.get("upcoming"))
    if not has_games:
        return min(ttl_seconds, EMPTY_FAILURE_TTL_SECONDS)
    return ttl_seconds


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _scoreboard_frames_for_date(game_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_error: Exception | None = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            response = scoreboardv2.ScoreboardV2(
                game_date=game_date.strftime("%m/%d/%Y"),
                timeout=API_TIMEOUT_SECONDS,
            )
            frames = response.get_data_frames()
            return frames[0], frames[1]
        except Exception as exc:
            last_error = exc
            if attempt < API_RETRIES:
                time.sleep(API_DELAY_SECONDS * attempt)
    raise RuntimeError(f"NBA scoreboard request failed for {game_date}: {last_error}")


def _format_date_label(value: Any) -> str:
    timestamp = _to_eastern_timestamp(value)
    if pd.isna(timestamp):
        return ""
    return f"{timestamp.strftime('%b')} {timestamp.day}, {timestamp.year}"


def _format_time_label(value: Any) -> str:
    timestamp = _to_eastern_timestamp(value)
    if pd.isna(timestamp):
        return ""
    hour = timestamp.hour % 12 or 12
    minute = f"{timestamp.minute:02d}"
    am_pm = "AM" if timestamp.hour < 12 else "PM"
    return f"{timestamp.strftime('%b')} {timestamp.day}, {hour}:{minute} {am_pm} ET"


def _to_eastern_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(EASTERN_TZ)
    return timestamp.tz_convert(EASTERN_TZ)


def _espn_scoreboard_json_for_date(game_date: date) -> dict[str, Any]:
    query = urlencode({"dates": game_date.strftime("%Y%m%d")})
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?{query}"
    with urlopen(url, timeout=ESPN_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _iter_text_candidates(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        if value.strip():
            texts.append(value.strip())
    elif isinstance(value, dict):
        for key in ("summary", "displayName", "description", "shortDetail", "detail", "headline", "text"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        for nested_key in ("series", "notes", "status"):
            texts.extend(_iter_text_candidates(value.get(nested_key)))
    elif isinstance(value, list):
        for item in value:
            texts.extend(_iter_text_candidates(item))
    return texts


def _nested_marker_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized_key in keys and item not in (None, ""):
                return item
            nested = _nested_marker_value(item, keys)
            if nested not in (None, ""):
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _nested_marker_value(item, keys)
            if nested not in (None, ""):
                return nested
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _game_number_from_context(event: dict[str, Any], competition: dict[str, Any], candidates: list[str]) -> int | None:
    value = _nested_marker_value(event, {"gamenumber"}) or _nested_marker_value(competition, {"gamenumber"})
    game_number = _int_or_none(value)
    if game_number is not None:
        return game_number

    for text in candidates:
        match = re.search(r"\bgame\s+([1-7])\b", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _has_playoff_marker(event: dict[str, Any], competition: dict[str, Any], candidates: list[str], game_number: int | None) -> bool:
    if game_number is not None:
        return True
    if _nested_marker_value(event, {"round", "serieslabel"}) or _nested_marker_value(competition, {"round", "serieslabel"}):
        return True
    marker_text = " ".join(candidates).lower()
    return any(
        marker in marker_text
        for marker in ("playoff", "finals", "semifinals", "conference finals", "first round", "second round")
    )


def _series_status_from_wins(away_abbr: str, home_abbr: str, away_wins: int, home_wins: int) -> str:
    if away_wins == home_wins:
        return f"Series tied {away_wins}-{home_wins}"

    leader_abbr = away_abbr if away_wins > home_wins else home_abbr
    leader_wins = max(away_wins, home_wins)
    trailer_wins = min(away_wins, home_wins)
    verb = "wins" if leader_wins >= 4 else "leads"
    return f"{leader_abbr} {verb} {leader_wins}-{trailer_wins}"


def _series_context(event: dict[str, Any], competition: dict[str, Any], away_abbr: str, home_abbr: str) -> dict[str, Any]:
    candidates = _iter_text_candidates(event) + _iter_text_candidates(competition)
    series_status = ""
    if_necessary = False
    leader_abbr = ""
    leader_wins: int | None = None
    trailer_wins: int | None = None
    tied_wins: int | None = None

    for text in candidates:
        normalized = " ".join(text.split())
        lowered = normalized.lower()
        if "if necessary" in lowered or "if-necessary" in lowered:
            if_necessary = True
        if not series_status and ("wins" in lowered or "won" in lowered or "leads" in lowered or "series tied" in lowered):
            series_status = _canonicalize_series_status(normalized)

        winner_match = re.search(
            r"\b([A-Z]{2,4})\s+(?:wins|won)(?:\s+(?:the\s+)?series)?\s+(\d+)\s*[-–]\s*(\d+)",
            normalized,
            re.IGNORECASE,
        )
        if winner_match and leader_wins is None:
            leader_abbr = canonical_team_abbr(winner_match.group(1))
            leader_wins = int(winner_match.group(2))
            trailer_wins = int(winner_match.group(3))
            series_status = f"{leader_abbr} wins {leader_wins}-{trailer_wins}"

        leader_match = re.search(
            r"\b([A-Z]{2,4})\s+leads(?:\s+(?:the\s+)?series)?\s+(\d+)\s*[-–]\s*(\d+)",
            normalized,
            re.IGNORECASE,
        )
        if leader_match and leader_wins is None:
            leader_abbr = canonical_team_abbr(leader_match.group(1))
            leader_wins = int(leader_match.group(2))
            trailer_wins = int(leader_match.group(3))
            if not series_status:
                series_status = f"{leader_abbr} leads {leader_wins}-{trailer_wins}"
            continue

        tied_match = re.search(r"series\s+tied\s+(\d+)\s*[-–]\s*(\d+)", normalized, re.IGNORECASE)
        if tied_match and tied_wins is None:
            tied_wins = int(tied_match.group(1))
            if not series_status:
                series_status = _canonicalize_series_status(f"Series tied {tied_match.group(1)}-{tied_match.group(2)}")

    away_wins: int | None = None
    home_wins: int | None = None
    if tied_wins is not None:
        away_wins = tied_wins
        home_wins = tied_wins
    elif leader_abbr and leader_wins is not None and trailer_wins is not None:
        if leader_abbr == away_abbr:
            away_wins = leader_wins
            home_wins = trailer_wins
        elif leader_abbr == home_abbr:
            away_wins = trailer_wins
            home_wins = leader_wins

    parsed_game_number = _game_number_from_context(event, competition, candidates)

    if away_wins is not None and home_wins is not None:
        completed_series = max(away_wins, home_wins) >= 4
        game_number = away_wins + home_wins if completed_series else away_wins + home_wins + 1
    else:
        game_number = parsed_game_number

    if (
        away_wins is None
        and home_wins is None
        and game_number == 1
        and _has_playoff_marker(event, competition, candidates, game_number)
    ):
        away_wins = 0
        home_wins = 0

    if not series_status and away_wins is not None and home_wins is not None:
        series_status = _series_status_from_wins(away_abbr, home_abbr, away_wins, home_wins)

    return {
        "series_status": series_status,
        "if_necessary": if_necessary,
        "away_series_wins": away_wins,
        "home_series_wins": home_wins,
        "game_number": game_number,
        "scheduled_game_number": parsed_game_number,
    }


def parse_espn_scoreboard_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for event in payload.get("events", []) or []:
        competitions = event.get("competitions", []) or []
        if not competitions:
            continue
        competition = competitions[0]
        competitors = competition.get("competitors", []) or []
        home = next((row for row in competitors if str(row.get("homeAway", "")).lower() == "home"), None)
        away = next((row for row in competitors if str(row.get("homeAway", "")).lower() == "away"), None)
        if not home or not away:
            continue

        status = event.get("status", {}).get("type", {}) or {}
        game_datetime = event.get("date") or competition.get("date")
        home_team = home.get("team", {}) or {}
        away_team = away.get("team", {}) or {}
        home_source_abbr = str(home_team.get("abbreviation") or "")
        away_source_abbr = str(away_team.get("abbreviation") or "")
        home_abbr = canonical_team_abbr(home_source_abbr, _team_display_name(home_team))
        away_abbr = canonical_team_abbr(away_source_abbr, _team_display_name(away_team))
        context = _series_context(event, competition, away_abbr, home_abbr)
        games.append(
            {
                "game_id": str(event.get("id") or competition.get("id") or ""),
                "status_id": 3 if bool(status.get("completed")) else (2 if str(status.get("state", "")).lower() == "in" else 1),
                "status_text": str(status.get("shortDetail") or status.get("description") or status.get("name") or ""),
                "game_datetime": game_datetime,
                "date_label": _format_date_label(game_datetime),
                "time_label": _format_time_label(game_datetime),
                "home_team_id": int(home_team.get("id") or 0),
                "away_team_id": int(away_team.get("id") or 0),
                "home_source_abbr": home_source_abbr,
                "away_source_abbr": away_source_abbr,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_score": _score_value(home.get("score")),
                "away_score": _score_value(away.get("score")),
                **context,
            }
        )
    return games


def _games_for_date(game_date: date) -> list[dict[str, Any]]:
    try:
        return parse_espn_scoreboard_games(_espn_scoreboard_json_for_date(game_date))
    except Exception as espn_error:
        try:
            header, lines = _scoreboard_frames_for_date(game_date)
            return parse_scoreboard_games(header, lines)
        except Exception as nba_error:
            raise RuntimeError(f"ESPN failed ({espn_error}); nba_api failed ({nba_error})")


def _score_value(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_scoreboard_games(game_header: pd.DataFrame, line_score: pd.DataFrame) -> list[dict[str, Any]]:
    if game_header.empty:
        return []

    line_by_game = {
        str(game_id): rows.copy()
        for game_id, rows in line_score.groupby("GAME_ID")
    } if not line_score.empty and "GAME_ID" in line_score.columns else {}

    games: list[dict[str, Any]] = []
    for row in game_header.to_dict(orient="records"):
        game_id = str(row.get("GAME_ID", ""))
        lines = line_by_game.get(game_id, pd.DataFrame())
        if lines.empty:
            continue

        home_team_id = int(row.get("HOME_TEAM_ID"))
        away_team_id = int(row.get("VISITOR_TEAM_ID"))
        home_line = lines[lines["TEAM_ID"].astype(int).eq(home_team_id)]
        away_line = lines[lines["TEAM_ID"].astype(int).eq(away_team_id)]
        if home_line.empty or away_line.empty:
            continue

        home = home_line.iloc[0]
        away = away_line.iloc[0]
        game_datetime = pd.to_datetime(row.get("GAME_DATE_EST"), errors="coerce")
        games.append(
            {
                "game_id": game_id,
                "status_id": int(row.get("GAME_STATUS_ID", 0) or 0),
                "status_text": str(row.get("GAME_STATUS_TEXT", "")),
                "game_datetime": None if pd.isna(game_datetime) else game_datetime.isoformat(),
                "date_label": _format_date_label(game_datetime),
                "time_label": _format_time_label(game_datetime) or str(row.get("GAME_STATUS_TEXT", "")),
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_abbr": str(home.get("TEAM_ABBREVIATION", "")),
                "away_abbr": str(away.get("TEAM_ABBREVIATION", "")),
                "home_score": _score_value(home.get("PTS")),
                "away_score": _score_value(away.get("PTS")),
            }
        )
    return games


def _is_completed_game(game: dict[str, Any]) -> bool:
    status_text = str(game.get("status_text", "")).strip().lower()
    return int(game.get("status_id", 0) or 0) == 3 or status_text in {"final", "completed", "complete"}


def _is_unnecessary_game(game: dict[str, Any]) -> bool:
    text = " ".join(
        str(game.get(key) or "")
        for key in ("status_text", "series_status", "time_label", "date_label")
    ).lower()
    return "unnecessary" in text


def _is_upcoming_game(game: dict[str, Any]) -> bool:
    return int(game.get("status_id", 0) or 0) in {1, 2} and not _is_unnecessary_game(game)


def _log_debug(debug: bool, message: str) -> None:
    if debug:
        print(message)


def _scan_scoreboards(today: date, debug: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest: list[dict[str, Any]] = []
    upcoming: list[dict[str, Any]] = []

    for offset in range(0, SCOREBOARD_SCAN_DAYS):
        day = today - timedelta(days=offset)
        try:
            finals = [game for game in _games_for_date(day) if _is_completed_game(game)]
            latest.extend(finals)
            _log_debug(debug, f"latest scan {day}: {len(finals)} final games found")
        except Exception as exc:
            _log_debug(debug, f"latest scan {day}: NBA API failed ({exc}); continuing")
            continue
        if len(latest) >= 3:
            break
        time.sleep(API_DELAY_SECONDS)

    for offset in range(0, SCOREBOARD_SCAN_DAYS):
        day = today + timedelta(days=offset)
        try:
            scheduled = [game for game in _games_for_date(day) if _is_upcoming_game(game)]
            upcoming.extend(scheduled)
            _log_debug(debug, f"upcoming scan {day}: {len(scheduled)} scheduled games found")
        except Exception as exc:
            _log_debug(debug, f"upcoming scan {day}: NBA API failed ({exc}); continuing")
            continue
        if len(upcoming) >= 3:
            break
        time.sleep(API_DELAY_SECONDS)

    latest = sorted(
        latest,
        key=lambda game: str(game.get("game_datetime") or ""),
        reverse=True,
    )[:3]
    upcoming = sorted(
        upcoming,
        key=lambda game: str(game.get("game_datetime") or ""),
    )[:3]
    return latest, upcoming


def load_live_games(
    cache_dir: str | Path = LIVE_GAMES_CACHE_DIR,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    today: date | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    cache_path = Path(cache_dir) / "live_games.json"
    cached = _read_cache(cache_path) if cache_path.exists() else None
    if cached is not None:
        cache_ttl = _effective_cache_ttl(cached, ttl_seconds)
        if _cache_is_fresh(cache_path, cache_ttl):
            return cached

    today = today or date.today()
    try:
        latest, upcoming = _scan_scoreboards(today, debug=debug)
        payload = {
            "latest": latest,
            "upcoming": upcoming,
            "error": None,
            "fetched_at": datetime.utcnow().isoformat(),
        }
        _write_cache(cache_path, payload)
        return payload
    except Exception as exc:
        if cached is not None:
            cached["error"] = str(exc)
            cached["stale"] = True
            return cached
        return {
            "latest": [],
            "upcoming": [],
            "error": str(exc),
            "fetched_at": datetime.utcnow().isoformat(),
        }
