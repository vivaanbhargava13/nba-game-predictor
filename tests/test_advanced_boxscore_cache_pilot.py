import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.advanced_boxscore_cache_pilot import (
    PLAYER_FIELDS,
    TEAM_FIELDS,
    build_missingness_report,
    cache_game_payload,
    canonical_game_id,
    discover_playoff_games,
    load_cached_payload,
    parse_advanced_boxscore,
    run_pilot,
)


def _payload(game_id="0042400101"):
    team_headers = ["GAME_ID", "TEAM_ID", *TEAM_FIELDS]
    player_headers = ["GAME_ID", "TEAM_ID", "PLAYER_ID", *PLAYER_FIELDS]
    return {
        "resultSets": [
            {
                "name": "TeamStats",
                "headers": team_headers,
                "rowSet": [
                    [game_id, 1, *([1.0] * len(TEAM_FIELDS))],
                    [game_id, 2, *([2.0] * len(TEAM_FIELDS))],
                ],
            },
            {
                "name": "PlayerStats",
                "headers": player_headers,
                "rowSet": [
                    [game_id, 1, 11, *([1.0] * len(PLAYER_FIELDS))],
                    [game_id, 2, 22, *([2.0] * len(PLAYER_FIELDS))],
                ],
            },
        ]
    }


class AdvancedBoxscoreCachePilotTests(unittest.TestCase):
    def _write_team_log(self, raw_dir, season, team_id, game_ids):
        pd.DataFrame(
            {
                "Team_ID": [team_id] * len(game_ids),
                "Game_ID": game_ids,
                "GAME_DATE": ["APR 20, 2025"] * len(game_ids),
                "MATCHUP": ["AAA vs. BBB"] * len(game_ids),
            }
        ).to_csv(
            raw_dir
            / f"team_game_log_{season}_playoffs_{team_id}.csv",
            index=False,
        )

    def test_discovers_unique_canonical_playoff_game_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir)
            self._write_team_log(raw, "2024-25", 1, [42400101])
            self._write_team_log(raw, "2024-25", 2, [42400101])
            games = discover_playoff_games(raw)
            self.assertEqual(len(games), 1)
            self.assertEqual(games.iloc[0]["GAME_ID"], "0042400101")
            self.assertEqual(games.iloc[0]["TEAM_ROWS"], 2)
            self.assertEqual(canonical_game_id("42400101.0"), "0042400101")

    def test_valid_cache_skips_fetcher(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            path = cache / "2024-25" / "playoffs" / "0042400101.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_payload()), encoding="utf-8")

            def fail_fetcher(*_args):
                raise AssertionError("valid cache should skip the API")

            result = cache_game_payload(
                season="2024-25",
                game_id="0042400101",
                cache_dir=cache,
                fetcher=fail_fetcher,
            )
            self.assertEqual(result["status"], "already_cached")
            self.assertEqual(result["attempts"], 0)

    def test_retry_then_atomic_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            attempts = []

            def fetcher(game_id, timeout):
                attempts.append((game_id, timeout))
                if len(attempts) == 1:
                    raise TimeoutError("temporary")
                return _payload(game_id)

            result = cache_game_payload(
                season="2024-25",
                game_id="0042400101",
                cache_dir=temp_dir,
                fetcher=fetcher,
                sleeper=lambda _seconds: None,
                clock=iter([0.0, 1.5]).__next__,
            )
            self.assertEqual(result["status"], "fetched")
            self.assertEqual(result["attempts"], 2)
            self.assertIsNotNone(load_cached_payload(Path(result["cache_path"])))
            self.assertFalse(
                Path(result["cache_path"]).with_suffix(".json.tmp").exists()
            )

    def test_failed_refresh_does_not_overwrite_valid_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            path = cache / "2024-25" / "playoffs" / "0042400101.json"
            path.parent.mkdir(parents=True)
            original = json.dumps(_payload())
            path.write_text(original, encoding="utf-8")
            result = cache_game_payload(
                season="2024-25",
                game_id="0042400101",
                cache_dir=cache,
                fetcher=lambda *_args: (_ for _ in ()).throw(
                    TimeoutError("offline")
                ),
            )
            self.assertEqual(result["status"], "already_cached")
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_missingness_covers_team_and_player_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = (
                Path(temp_dir)
                / "2024-25"
                / "playoffs"
                / "0042400101.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            ledger = pd.DataFrame(
                [
                    {
                        "season": "2024-25",
                        "status": "already_cached",
                        "team_rows": 2,
                        "player_rows": 2,
                        "cache_path": str(path),
                    }
                ]
            )
            missingness = build_missingness_report(ledger)
            current = missingness[missingness["season"].eq("2024-25")]
            self.assertEqual(set(current["split"]), {"team", "player"})
            self.assertTrue(current["missingness_pct"].eq(0).all())

    def test_run_pilot_writes_requested_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            self._write_team_log(raw, "2024-25", 1, [42400101])
            self._write_team_log(raw, "2024-25", 2, [42400101])
            ledger, missingness = run_pilot(
                raw_dir=raw,
                cache_dir=root / "cache",
                processed_dir=root / "processed",
                doc_path=root / "pilot.md",
                fetcher=lambda game_id, _timeout: _payload(game_id),
                sleeper=lambda _seconds: None,
                clock=iter([0.0, 1.0]).__next__,
            )
            self.assertEqual(ledger.iloc[0]["status"], "fetched")
            self.assertFalse(missingness.empty)
            self.assertTrue(
                (root / "processed" / "advanced_boxscore_cache_pilot.csv").exists()
            )
            self.assertTrue(
                (
                    root
                    / "processed"
                    / "advanced_boxscore_cache_pilot_missingness.csv"
                ).exists()
            )
            self.assertIn(
                "Worth full ingestion",
                (root / "pilot.md").read_text(encoding="utf-8"),
            )

    def test_pilot_stops_repeated_calls_after_systemic_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            for game_id in [42400101, 42400102, 42400103]:
                self._write_team_log(raw, "2024-25", 1, [game_id])
                source = raw / "team_game_log_2024-25_playoffs_1.csv"
                frame = pd.read_csv(source)
                existing = (
                    pd.read_csv(raw / "combined.csv")
                    if (raw / "combined.csv").exists()
                    else pd.DataFrame()
                )
                pd.concat([existing, frame], ignore_index=True).to_csv(
                    raw / "combined.csv", index=False
                )
            combined = pd.read_csv(raw / "combined.csv")
            combined.to_csv(
                raw / "team_game_log_2024-25_playoffs_1.csv", index=False
            )
            calls = []

            def failing_fetcher(game_id, _timeout):
                calls.append(game_id)
                raise KeyError("resultSet")

            ledger, _ = run_pilot(
                raw_dir=raw,
                cache_dir=root / "cache",
                processed_dir=root / "processed",
                doc_path=root / "pilot.md",
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

    def test_parser_keeps_team_and_player_splits(self):
        team_stats, player_stats = parse_advanced_boxscore(_payload())
        self.assertEqual(len(team_stats), 2)
        self.assertEqual(len(player_stats), 2)
        self.assertIn("OFF_RATING", team_stats)
        self.assertIn("USG_PCT", player_stats)


if __name__ == "__main__":
    unittest.main()
