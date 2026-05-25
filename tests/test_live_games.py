import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pandas as pd

from src.live_games import (
    _scan_scoreboards,
    load_live_games,
    nba_season_from_date,
    parse_espn_scoreboard_games,
    parse_scoreboard_games,
)
from app import (
    PREDICTION_MODE_PLAYOFF,
    _upcoming_prediction_result,
    build_live_games_topbar_html,
    live_card_key,
    live_game_button_label,
    live_game_context_lines,
    live_game_payload,
    live_game_render_payloads,
    live_game_selection_state,
    safe_live_games_payload,
)


class LiveGamesTests(unittest.TestCase):
    def test_parse_scoreboard_games_latest_and_upcoming_fields(self):
        header = pd.DataFrame(
            [
                {
                    "GAME_ID": "001",
                    "GAME_STATUS_ID": 3,
                    "GAME_STATUS_TEXT": "Final",
                    "GAME_DATE_EST": "2026-05-24T20:00:00",
                    "HOME_TEAM_ID": 1,
                    "VISITOR_TEAM_ID": 2,
                },
                {
                    "GAME_ID": "002",
                    "GAME_STATUS_ID": 1,
                    "GAME_STATUS_TEXT": "7:30 PM ET",
                    "GAME_DATE_EST": "2026-05-25T19:30:00",
                    "HOME_TEAM_ID": 3,
                    "VISITOR_TEAM_ID": 4,
                },
            ]
        )
        lines = pd.DataFrame(
            [
                {"GAME_ID": "001", "TEAM_ID": 1, "TEAM_ABBREVIATION": "BOS", "PTS": 110},
                {"GAME_ID": "001", "TEAM_ID": 2, "TEAM_ABBREVIATION": "NYK", "PTS": 101},
                {"GAME_ID": "002", "TEAM_ID": 3, "TEAM_ABBREVIATION": "LAL", "PTS": None},
                {"GAME_ID": "002", "TEAM_ID": 4, "TEAM_ABBREVIATION": "GSW", "PTS": None},
            ]
        )

        games = parse_scoreboard_games(header, lines)

        self.assertEqual(len(games), 2)
        self.assertEqual(games[0]["home_abbr"], "BOS")
        self.assertEqual(games[0]["away_abbr"], "NYK")
        self.assertEqual(games[0]["home_score"], 110)
        self.assertEqual(games[0]["away_score"], 101)
        self.assertEqual(games[1]["status_text"], "7:30 PM ET")
        self.assertIsNone(games[1]["home_score"])

    def test_parse_espn_scoreboard_home_away(self):
        payload = _espn_payload(
            "401",
            "2026-05-25T23:30Z",
            completed=False,
            away_abbr="NYK",
            away_id="18",
            away_score="",
            home_abbr="CLE",
            home_id="5",
            home_score="",
        )

        games = parse_espn_scoreboard_games(payload)

        self.assertEqual(games[0]["away_abbr"], "NYK")
        self.assertEqual(games[0]["home_abbr"], "CLE")
        self.assertEqual(games[0]["away_team_id"], 18)
        self.assertEqual(games[0]["home_team_id"], 5)

    def test_parse_espn_series_status_and_if_necessary(self):
        payload = _espn_payload(
            "401",
            "2026-05-25T23:30Z",
            completed=False,
            away_abbr="NYK",
            away_id="18",
            home_abbr="CLE",
            home_id="5",
            series_status="NYK leads 3-0",
            note="Game 4 - If Necessary",
        )

        games = parse_espn_scoreboard_games(payload)

        self.assertEqual(games[0]["series_status"], "NYK leads 3-0")
        self.assertTrue(games[0]["if_necessary"])
        self.assertEqual(games[0]["away_series_wins"], 3)
        self.assertEqual(games[0]["home_series_wins"], 0)
        self.assertEqual(games[0]["game_number"], 4)

    def test_live_games_falls_back_to_empty_payload_without_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.live_games._scan_scoreboards", side_effect=RuntimeError("api down")):
                payload = load_live_games(cache_dir=Path(tmpdir), ttl_seconds=0, today=date(2026, 5, 25))

        self.assertEqual(payload["latest"], [])
        self.assertEqual(payload["upcoming"], [])
        self.assertIn("api down", payload["error"])

    def test_live_games_uses_stale_cache_when_api_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "live_games.json"
            cache_path.write_text('{"latest":[{"game_id":"1"}],"upcoming":[],"error":null}')
            with patch("src.live_games._scan_scoreboards", side_effect=RuntimeError("api down")):
                payload = load_live_games(cache_dir=Path(tmpdir), ttl_seconds=0, today=date(2026, 5, 25))

        self.assertEqual(payload["latest"], [{"game_id": "1"}])
        self.assertTrue(payload["stale"])
        self.assertIn("api down", payload["error"])

    def test_nba_season_from_date(self):
        self.assertEqual(nba_season_from_date("2026-05-25"), "2025-26")
        self.assertEqual(nba_season_from_date("2026-10-25"), "2026-27")

    def test_safe_live_games_payload_catches_fetch_failure(self):
        payload = safe_live_games_payload(fetcher=lambda: (_ for _ in ()).throw(RuntimeError("api down")))

        self.assertEqual(payload["latest"], [])
        self.assertEqual(payload["upcoming"], [])
        self.assertIn("api down", payload["error"])

    def test_safe_live_games_payload_times_out_without_blocking_dashboard(self):
        def slow_fetcher():
            import time

            time.sleep(0.25)
            return {"latest": [{"game_id": "late"}], "upcoming": []}

        payload = safe_live_games_payload(fetcher=slow_fetcher, timeout_seconds=0.01)

        self.assertEqual(payload["latest"], [])
        self.assertEqual(payload["upcoming"], [])
        self.assertIn("unavailable", payload["error"])

    def test_latest_scan_finds_yesterday_finals_when_today_has_none(self):
        def games_for_date(game_date):
            if game_date == date(2026, 5, 25):
                return []
            if game_date == date(2026, 5, 24):
                return parse_scoreboard_games(*_scoreboard_fixture(["001", "002", "003"]))
            return []

        with patch("src.live_games._games_for_date", side_effect=games_for_date):
            latest, _upcoming = _scan_scoreboards(date(2026, 5, 25))

        self.assertEqual([game["game_id"] for game in latest], ["003", "002", "001"])

    def test_latest_scan_returns_max_three_completed_games(self):
        with patch("src.live_games._games_for_date", return_value=parse_scoreboard_games(*_scoreboard_fixture(["001", "002", "003", "004"]))):
            latest, _upcoming = _scan_scoreboards(date(2026, 5, 25))

        self.assertEqual(len(latest), 3)

    def test_latest_scan_continues_after_one_date_api_failure(self):
        def games_for_date(game_date):
            if game_date == date(2026, 5, 25):
                raise RuntimeError("temporary scoreboard failure")
            return parse_scoreboard_games(*_scoreboard_fixture(["001", "002", "003"]))

        with patch("src.live_games._games_for_date", side_effect=games_for_date):
            latest, _upcoming = _scan_scoreboards(date(2026, 5, 25))

        self.assertEqual(len(latest), 3)

    def test_upcoming_scan_finds_future_games_and_keeps_time_label(self):
        def games_for_date(game_date):
            if game_date == date(2026, 5, 25):
                return []
            if game_date == date(2026, 5, 26):
                return parse_espn_scoreboard_games(
                    _espn_payload(
                        "401",
                        "2026-05-26T23:30Z",
                        completed=False,
                        away_abbr="NYK",
                        away_id="18",
                        home_abbr="CLE",
                        home_id="5",
                    )
                )
            return []

        with patch("src.live_games._games_for_date", side_effect=games_for_date):
            _latest, upcoming = _scan_scoreboards(date(2026, 5, 25))

        self.assertEqual(upcoming[0]["away_abbr"], "NYK")
        self.assertEqual(upcoming[0]["home_abbr"], "CLE")
        self.assertIn("May 26", upcoming[0]["time_label"])

    def test_today_game_still_shows_date_time(self):
        games = parse_espn_scoreboard_games(
            _espn_payload(
                "401",
                "2026-05-25T23:30Z",
                completed=False,
                away_abbr="NYK",
                away_id="18",
                home_abbr="CLE",
                home_id="5",
            )
        )

        self.assertIn("May 25", games[0]["time_label"])
        self.assertIn("ET", games[0]["time_label"])

    def test_live_games_topbar_builds_three_latest_and_three_upcoming_cards(self):
        live_games = {
            "latest": [
                _game("1", "NYK", "CLE", away_score=90, home_score=101),
                _game("2", "BOS", "MIA", away_score=110, home_score=99),
                _game("3", "LAL", "GSW", away_score=100, home_score=104),
                _game("4", "DAL", "DEN", away_score=100, home_score=90),
            ],
            "upcoming": [
                _game("5", "NYK", "CLE"),
                _game("6", "BOS", "MIA"),
                _game("7", "LAL", "GSW"),
                _game("8", "DAL", "DEN"),
            ],
        }

        output = build_live_games_topbar_html(live_games, model_available=False)

        self.assertEqual(output.count('class="live-game-card"'), 6)
        self.assertIn("NYK", output)
        self.assertIn("May 25, 7:30 PM ET", output)
        self.assertIn("Prediction unavailable", output)
        self.assertNotIn("\n        <div", output)

    def test_live_game_context_lines_include_series_and_if_necessary(self):
        game = _game("401", "NYK", "CLE")
        game["series_status"] = "NYK leads 3-0"
        game["if_necessary"] = True

        lines = live_game_context_lines(game)

        self.assertIn("NYK leads 3-0", lines)
        self.assertIn("IF NECESSARY", lines)

    def test_live_game_selection_state_loads_away_home_and_series(self):
        labels = ["Cleveland Cavaliers (CLE)", "New York Knicks (NYK)"]
        game = _game("401", "NYK", "CLE")
        game.update(
            {
                "game_datetime": "2026-05-25T23:30Z",
                "series_status": "NYK leads 3-0",
                "away_series_wins": 3,
                "home_series_wins": 0,
                "game_number": 4,
            }
        )

        state = live_game_selection_state(game, labels)

        self.assertEqual(state["selected_team_label"], "New York Knicks (NYK)")
        self.assertEqual(state["opponent_team_label"], "Cleveland Cavaliers (CLE)")
        self.assertEqual(state["home_team_label"], "Cleveland Cavaliers (CLE)")
        self.assertEqual(state["season"], "2025-26")
        self.assertEqual(state["prediction_context_mode"], PREDICTION_MODE_PLAYOFF)
        self.assertEqual(state["game_number"], 4)
        self.assertEqual(state["team_a_series_wins"], 3)

    def test_live_game_selection_state_loads_cle_at_nyk(self):
        labels = ["Cleveland Cavaliers (CLE)", "New York Knicks (NYK)"]
        game = _game("402", "CLE", "NYK")
        game.update({"game_datetime": "2026-05-27T00:00Z"})

        state = live_game_selection_state(live_game_payload(game), labels)

        self.assertEqual(state["selected_team_label"], "Cleveland Cavaliers (CLE)")
        self.assertEqual(state["opponent_team_label"], "New York Knicks (NYK)")
        self.assertEqual(state["home_team_label"], "New York Knicks (NYK)")

    def test_clicking_one_card_cannot_reuse_another_card_matchup(self):
        labels = [
            "Atlanta Hawks (ATL)",
            "Cleveland Cavaliers (CLE)",
            "New York Knicks (NYK)",
        ]
        nyk_at_cle = live_game_payload(_game("401", "NYK", "CLE"))
        cle_at_nyk = live_game_payload(_game("402", "CLE", "NYK"))

        first_state = live_game_selection_state(nyk_at_cle, labels)
        second_state = live_game_selection_state(cle_at_nyk, labels)

        self.assertEqual(first_state["selected_team_label"], "New York Knicks (NYK)")
        self.assertEqual(first_state["opponent_team_label"], "Cleveland Cavaliers (CLE)")
        self.assertEqual(second_state["selected_team_label"], "Cleveland Cavaliers (CLE)")
        self.assertEqual(second_state["opponent_team_label"], "New York Knicks (NYK)")
        self.assertNotEqual(first_state["selected_team_label"], second_state["selected_team_label"])

    def test_live_game_payload_has_immutable_click_fields(self):
        game = _game("401", "NYK", "CLE", away_score=99, home_score=100)
        game.update(
            {
                "game_datetime": "2026-05-25T23:30Z",
                "series_status": "NYK leads 3-0",
                "game_number": 4,
                "away_series_wins": 3,
                "home_series_wins": 0,
                "if_necessary": True,
            }
        )

        payload = live_game_payload(game)

        self.assertEqual(payload["game_id"], "401")
        self.assertEqual(payload["away_abbr"], "NYK")
        self.assertEqual(payload["home_abbr"], "CLE")
        self.assertEqual(payload["game_date"], "2026-05-25")
        self.assertEqual(payload["season"], "2025-26")
        self.assertEqual(payload["series_status"], "NYK leads 3-0")
        self.assertEqual(payload["game_number"], 4)
        self.assertEqual(payload["away_series_wins"], 3)
        self.assertEqual(payload["home_series_wins"], 0)
        self.assertTrue(payload["if_necessary"])

    def test_live_card_keys_are_stable_and_unique(self):
        first = live_game_payload(_game("401", "NYK", "CLE"))
        second = live_game_payload(_game("401", "CLE", "NYK"))

        self.assertEqual(live_card_key("upcoming", first, 0), "live_card_upcoming_401_0")
        self.assertNotEqual(live_card_key("upcoming", first, 0), live_card_key("upcoming", second, 1))

    def test_live_game_render_payloads_limit_to_three_per_section(self):
        live_games = {
            "latest": [
                _game("1", "NYK", "CLE"),
                _game("2", "BOS", "MIA"),
                _game("3", "LAL", "GSW"),
                _game("4", "DAL", "DEN"),
            ],
            "upcoming": [
                _game("5", "NYK", "CLE"),
                _game("6", "BOS", "MIA"),
                _game("7", "LAL", "GSW"),
                _game("8", "DAL", "DEN"),
            ],
        }

        self.assertEqual(len(live_game_render_payloads(live_games, "latest")), 3)
        self.assertEqual(len(live_game_render_payloads(live_games, "upcoming")), 3)

    def test_upcoming_prediction_failure_returns_debug_reason(self):
        def failing_predictor(**_kwargs):
            raise RuntimeError("model unavailable")

        prediction, debug = _upcoming_prediction_result(
            _game("401", "NYK", "CLE"),
            model_available=True,
            predictor=failing_predictor,
        )

        self.assertEqual(prediction, "Prediction unavailable")
        self.assertIn("model unavailable", debug)

    def test_live_game_button_label_uses_prediction_fallback_without_crashing(self):
        label, debug = live_game_button_label(_game("401", "NYK", "CLE"), is_upcoming=True, model_available=False)

        self.assertIn("NYK @ CLE", label)
        self.assertIn("Prediction unavailable", label)
        self.assertIn("Model artifact", debug)


def _scoreboard_fixture(game_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    header_rows = []
    line_rows = []
    for index, game_id in enumerate(game_ids, start=1):
        home_id = index * 10
        away_id = index * 10 + 1
        header_rows.append(
            {
                "GAME_ID": game_id,
                "GAME_STATUS_ID": 3,
                "GAME_STATUS_TEXT": "Final",
                "GAME_DATE_EST": f"2026-05-24T{18 + index:02d}:00:00",
                "HOME_TEAM_ID": home_id,
                "VISITOR_TEAM_ID": away_id,
            }
        )
        line_rows.extend(
            [
                {"GAME_ID": game_id, "TEAM_ID": home_id, "TEAM_ABBREVIATION": f"H{index}", "PTS": 100 + index},
                {"GAME_ID": game_id, "TEAM_ID": away_id, "TEAM_ABBREVIATION": f"A{index}", "PTS": 90 + index},
            ]
        )
    return pd.DataFrame(header_rows), pd.DataFrame(line_rows)


def _game(
    game_id: str,
    away_abbr: str,
    home_abbr: str,
    away_score: Optional[int] = None,
    home_score: Optional[int] = None,
) -> dict:
    return {
        "game_id": game_id,
        "away_abbr": away_abbr,
        "home_abbr": home_abbr,
        "game_datetime": "2026-05-25T23:30Z",
        "away_score": away_score,
        "home_score": home_score,
        "date_label": "May 25, 2026",
        "time_label": "May 25, 7:30 PM ET",
    }


def _espn_payload(
    game_id: str,
    game_date: str,
    *,
    completed: bool,
    away_abbr: str,
    away_id: str,
    home_abbr: str,
    home_id: str,
    away_score: str = "",
    home_score: str = "",
    series_status: str = "",
    note: str = "",
) -> dict:
    return {
        "events": [
            {
                "id": game_id,
                "date": game_date,
                "series": {"summary": series_status} if series_status else {},
                "notes": [{"headline": note}] if note else [],
                "status": {
                    "type": {
                        "completed": completed,
                        "state": "post" if completed else "pre",
                        "description": "Final" if completed else "Scheduled",
                        "shortDetail": "Final" if completed else "7:30 PM",
                    }
                },
                "competitions": [
                    {
                        "id": game_id,
                        "series": {"summary": series_status} if series_status else {},
                        "competitors": [
                            {
                                "homeAway": "away",
                                "score": away_score,
                                "team": {"id": away_id, "abbreviation": away_abbr},
                            },
                            {
                                "homeAway": "home",
                                "score": home_score,
                                "team": {"id": home_id, "abbreviation": home_abbr},
                            },
                        ],
                    }
                ],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
