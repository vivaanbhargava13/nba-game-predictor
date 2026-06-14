import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.advanced_boxscore_v3_cache_pilot import (
    PLAYER_FIELDS,
    TEAM_FIELDS,
    build_missingness_report,
    cache_game_payload,
    discover_playoff_games,
    load_cached_payload,
    parse_v3_payload,
    recommend_backfill,
    run_pilot,
    validate_payload,
)


def _statistics(player=False):
    values = {
        "minutes": "30:00",
        "offensiveRating": 115.0,
        "defensiveRating": 108.0,
        "netRating": 7.0,
        "pace": 96.0,
        "possessions": 60.0 if player else 96.0,
        "trueShootingPercentage": 0.61,
        "effectiveFieldGoalPercentage": 0.57,
        "assistRatio": 18.0,
        "offensiveReboundPercentage": 0.22,
        "defensiveReboundPercentage": 0.75,
        "reboundPercentage": 0.49,
        "turnoverRatio": 11.0,
        "PIE": 0.55,
        "usagePercentage": 0.24,
    }
    return values


def _team(team_id, tricode):
    return {
        "teamId": team_id,
        "teamCity": tricode,
        "teamName": tricode,
        "teamTricode": tricode,
        "players": [
            {
                "personId": team_id * 10,
                "firstName": "Test",
                "familyName": tricode,
                "position": "G",
                "comment": "",
                "statistics": _statistics(player=True),
            }
        ],
        "statistics": _statistics(),
    }


def _payload(game_id="0042400101"):
    return {
        "meta": {"version": 1},
        "boxScoreAdvanced": {
            "gameId": game_id,
            "awayTeamId": 1,
            "homeTeamId": 2,
            "awayTeam": _team(1, "AAA"),
            "homeTeam": _team(2, "BBB"),
        },
    }


class AdvancedBoxscoreV3CachePilotTests(unittest.TestCase):
    def _write_logs(self, directory, game_ids):
        for team_id in (1, 2):
            pd.DataFrame(
                {
                    "Team_ID": [team_id] * len(game_ids),
                    "Game_ID": game_ids,
                    "GAME_DATE": ["APR 20, 2025"] * len(game_ids),
                }
            ).to_csv(
                directory
                / f"team_game_log_2024-25_playoffs_{team_id}.csv",
                index=False,
            )

    def test_discovers_unique_playoff_games(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_logs(root, [42400101, 42400102])
            games = discover_playoff_games(root)
            self.assertEqual(games["GAME_ID"].tolist(), [
                "0042400101",
                "0042400102",
            ])
            self.assertTrue(games["TEAM_ROWS"].eq(2).all())

    def test_parser_uses_actual_nested_v3_fields(self):
        team_rows, player_rows = parse_v3_payload(_payload())
        self.assertEqual(len(team_rows), 2)
        self.assertEqual(len(player_rows), 2)
        self.assertIn("offensiveRating", team_rows)
        self.assertIn("usagePercentage", player_rows)
        self.assertNotIn("plusMinus", player_rows)

    def test_empty_and_malformed_responses_are_classified(self):
        self.assertEqual(validate_payload({})[1], "empty_response")
        self.assertEqual(
            validate_payload({"meta": {}})[1],
            "malformed_response",
        )

    def test_valid_cache_prevents_repeated_api_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                Path(temp_dir)
                / "2024-25"
                / "0042400101.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_payload()), encoding="utf-8")

            def unexpected_fetch(*_args):
                raise AssertionError("cached game should not be fetched")

            result = cache_game_payload(
                season="2024-25",
                game_id="0042400101",
                cache_dir=temp_dir,
                fetcher=unexpected_fetch,
            )
            self.assertEqual(result["status"], "already_cached")
            self.assertEqual(result["attempt_count"], 0)

    def test_retry_writes_valid_payload_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def fetcher(game_id, _timeout):
                calls.append(game_id)
                if len(calls) == 1:
                    raise TimeoutError("temporary")
                return _payload(game_id)

            result = cache_game_payload(
                season="2024-25",
                game_id="0042400101",
                cache_dir=temp_dir,
                fetcher=fetcher,
                sleeper=lambda _seconds: None,
                clock=iter([0.0, 1.0]).__next__,
            )
            self.assertEqual(result["status"], "fetched")
            self.assertEqual(result["attempt_count"], 2)
            path = Path(result["cache_path"])
            self.assertIsNotNone(load_cached_payload(path))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_missingness_reports_unavailable_plus_minus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "2024-25" / "0042400101.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            ledger = pd.DataFrame(
                [
                    {
                        "season": "2024-25",
                        "status": "already_cached",
                        "team_rows_count": 2,
                        "player_rows_count": 2,
                        "duplicate_team_rows": 0,
                        "duplicate_player_rows": 0,
                        "error_type": "",
                        "cache_path": str(path),
                    }
                ]
            )
            report = build_missingness_report(ledger)
            core = report[
                report["field_or_check"].isin(TEAM_FIELDS + PLAYER_FIELDS)
                & report["season"].eq("2024-25")
            ]
            self.assertTrue(core["missingness_pct"].eq(0).all())
            plus_minus = report[
                report["field_or_check"].eq("plusMinus")
                & report["season"].eq("2024-25")
            ]
            self.assertTrue(plus_minus["missingness_pct"].eq(100).all())

    def test_run_pilot_writes_outputs_and_recommends_playoff_backfill(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            self._write_logs(logs, [42400101])
            ledger, missingness = run_pilot(
                game_log_dir=logs,
                cache_dir=root / "cache",
                processed_dir=root / "processed",
                doc_path=root / "report.md",
                fetcher=lambda game_id, _timeout: _payload(game_id),
                sleeper=lambda _seconds: None,
                clock=iter([0.0, 1.0]).__next__,
            )
            self.assertEqual(ledger.iloc[0]["status"], "fetched")
            self.assertEqual(
                recommend_backfill(ledger, missingness)[0],
                "full playoff backfill",
            )
            self.assertTrue(
                (
                    root
                    / "processed"
                    / "advanced_boxscore_v3_cache_pilot.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    root
                    / "processed"
                    / "advanced_boxscore_v3_missingness.csv"
                ).exists()
            )
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("Sample Parsed Team Row", report)
            self.assertIn("full playoff backfill", report)

    def test_circuit_breaker_stops_systemic_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            self._write_logs(logs, [42400101, 42400102, 42400103])
            calls = []

            def failing_fetcher(game_id, _timeout):
                calls.append(game_id)
                return {}

            ledger, _ = run_pilot(
                game_log_dir=logs,
                cache_dir=root / "cache",
                processed_dir=root / "processed",
                doc_path=root / "report.md",
                fetcher=failing_fetcher,
                sleeper=lambda _seconds: None,
                systemic_failure_threshold=1,
            )
            self.assertEqual(len(calls), 3)
            self.assertEqual(ledger.iloc[0]["status"], "failed")
            self.assertTrue(
                ledger.iloc[1:]["status"]
                .eq("skipped_systemic_failure")
                .all()
            )


if __name__ == "__main__":
    unittest.main()
