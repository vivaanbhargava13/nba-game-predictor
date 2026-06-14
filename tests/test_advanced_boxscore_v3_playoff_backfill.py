import json
import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.advanced_boxscore_v3_playoff_backfill import (
    build_missingness_report,
    classify_result,
    materialize_rows,
    run_backfill,
)
import src.advanced_boxscore_v3_playoff_backfill as backfill_module


def _stats(player=False):
    return {
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


def _team(team_id, code):
    return {
        "teamId": team_id,
        "teamCity": code,
        "teamName": code,
        "teamTricode": code,
        "statistics": _stats(),
        "players": [
            {
                "personId": team_id * 100,
                "firstName": "Test",
                "familyName": code,
                "position": "G",
                "comment": "",
                "statistics": _stats(player=True),
            }
        ],
    }


def _payload(game_id):
    return {
        "meta": {"version": 1},
        "boxScoreAdvanced": {
            "gameId": game_id,
            "awayTeam": _team(1, "AAA"),
            "homeTeam": _team(2, "BBB"),
        },
    }


class AdvancedBoxscoreV3PlayoffBackfillTests(unittest.TestCase):
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

    def test_cached_games_are_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            cache = root / "cache"
            logs.mkdir()
            self._write_logs(logs, [42400101])
            path = cache / "2024-25" / "0042400101.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(_payload("0042400101")),
                encoding="utf-8",
            )

            def fail_fetch(*_args):
                raise AssertionError("cache hit must not refetch")

            ledger, _, _, _ = run_backfill(
                game_log_dir=logs,
                cache_dir=cache,
                processed_dir=root / "processed",
                doc_path=root / "report.md",
                fetcher=fail_fetch,
            )
            self.assertEqual(
                ledger.iloc[0]["quality_class"], "cache_hit"
            )

    def test_missing_game_is_fetched_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            self._write_logs(logs, [42400101])
            ledger, _, _, _ = run_backfill(
                game_log_dir=logs,
                cache_dir=root / "cache",
                processed_dir=root / "processed",
                doc_path=root / "report.md",
                fetcher=lambda game_id, _timeout: _payload(game_id),
                sleeper=lambda _seconds: None,
                clock=iter([0.0, 1.0]).__next__,
            )
            path = Path(ledger.iloc[0]["cache_path"])
            self.assertEqual(
                ledger.iloc[0]["quality_class"], "success"
            )
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_empty_and_malformed_classifications(self):
        self.assertEqual(
            classify_result(
                {
                    "status": "failed",
                    "error_type": "empty_response",
                    "error": "empty",
                }
            ),
            "empty_response",
        )
        self.assertEqual(
            classify_result(
                {
                    "status": "failed",
                    "error_type": "missing_team_rows",
                    "error": "one row",
                }
            ),
            "malformed_response",
        )

    def test_duplicate_checks_work(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "2024-25" / "0042400101.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(_payload("0042400101")),
                encoding="utf-8",
            )
            ledger = pd.DataFrame(
                [
                    {
                        "season": "2024-25",
                        "game_id": "0042400101",
                        "game_date": "2025-04-20",
                        "quality_class": "cache_hit",
                        "cache_path": str(path),
                    },
                    {
                        "season": "2024-25",
                        "game_id": "0042400101",
                        "game_date": "2025-04-20",
                        "quality_class": "cache_hit",
                        "cache_path": str(path),
                    },
                ]
            )
            teams, players = materialize_rows(ledger)
            report = build_missingness_report(
                ledger, teams, players
            )
            duplicate_team = report[
                report["field_or_check"].eq("duplicate_team_rows")
            ]
            duplicate_player = report[
                report["field_or_check"].eq(
                    "duplicate_player_rows"
                )
            ]
            self.assertGreater(
                int(duplicate_team.iloc[0]["observed_count"]), 0
            )
            self.assertGreater(
                int(duplicate_player.iloc[0]["observed_count"]), 0
            )

    def test_missingness_only_uses_present_seasons_and_plus_minus_unavailable(self):
        ledger = pd.DataFrame(
            [
                {
                    "season": "2024-25",
                    "quality_class": "api_error",
                    "team_rows_count": 0,
                    "player_rows_count": 0,
                }
            ]
        )
        report = build_missingness_report(
            ledger, pd.DataFrame(), pd.DataFrame()
        )
        self.assertEqual(set(report["season"]), {"2024-25"})
        plus_minus = report[
            report["field_or_check"].eq("plusMinus")
        ]
        self.assertTrue(
            plus_minus["availability"]
            .eq("unavailable_in_v3")
            .all()
        )
        self.assertTrue(plus_minus["missing_count"].eq(0).all())

    def test_circuit_breaker_stops_systemic_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            self._write_logs(logs, [42400101, 42400102])
            calls = []

            def failing_fetcher(game_id, _timeout):
                calls.append(game_id)
                return {}

            ledger, _, _, _ = run_backfill(
                game_log_dir=logs,
                cache_dir=root / "cache",
                processed_dir=root / "processed",
                doc_path=root / "report.md",
                fetcher=failing_fetcher,
                sleeper=lambda _seconds: None,
                systemic_failure_threshold=1,
            )
            self.assertEqual(len(calls), 3)
            self.assertEqual(
                ledger.iloc[0]["quality_class"], "empty_response"
            )
            self.assertEqual(
                ledger.iloc[1]["quality_class"],
                "circuit_breaker_skipped",
            )

    def test_backfill_module_isolated_from_streamlit_and_models(self):
        source = inspect.getsource(backfill_module)
        self.assertNotIn("streamlit", source)
        self.assertNotIn("playoff_predictor.joblib", source)
        self.assertNotIn("series_context_model.joblib", source)
        self.assertNotIn("src.model", source)
        self.assertNotIn("src.predictor", source)


if __name__ == "__main__":
    unittest.main()
