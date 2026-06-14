import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.nba_api_feature_source_audit import (
    FAMILY_CONFIG,
    build_family_priorities,
    build_feature_source_audit,
    run_feature_source_audit,
)


class NbaApiFeatureSourceAuditTests(unittest.TestCase):
    def _write_team_log(self, path: Path, game_ids):
        pd.DataFrame(
            {
                "Team_ID": [1] * len(game_ids),
                "Game_ID": game_ids,
                "GAME_DATE": ["APR 01, 2025"] * len(game_ids),
                "MATCHUP": ["AAA vs. BBB"] * len(game_ids),
                "WL": ["W"] * len(game_ids),
                "MIN": [240] * len(game_ids),
                "FGM": [40] * len(game_ids),
                "FGA": [80] * len(game_ids),
                "FG3M": [10] * len(game_ids),
                "FG3A": [30] * len(game_ids),
                "FTM": [20] * len(game_ids),
                "FTA": [25] * len(game_ids),
                "OREB": [10] * len(game_ids),
                "DREB": [30] * len(game_ids),
                "REB": [40] * len(game_ids),
                "AST": [25] * len(game_ids),
                "TOV": [12] * len(game_ids),
                "STL": [8] * len(game_ids),
                "BLK": [5] * len(game_ids),
                "PF": [20] * len(game_ids),
                "PTS": [110] * len(game_ids),
                "PLUS_MINUS": [5] * len(game_ids),
            }
        ).to_csv(path, index=False)

    def test_audit_contains_all_families_and_seasons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir)
            self._write_team_log(
                raw / "team_game_log_2024-25_regular_season_1.csv",
                ["1", "2"],
            )
            audit = build_feature_source_audit(raw, seasons=["2024-25"])
            self.assertEqual(set(audit["feature_family"]), set(FAMILY_CONFIG))
            self.assertEqual(set(audit["season"]), {"2024-25"})
            required = {
                "historical_game_coverage_pct",
                "key_field_missingness_pct",
                "api_cost_rate_limit_risk",
                "leakage_risk",
                "feature_ideas",
                "recommended_experiment_priority",
                "deployability_risk",
            }
            self.assertTrue(required <= set(audit.columns))

    def test_uncached_endpoints_do_not_claim_game_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = build_feature_source_audit(
                temp_dir, seasons=["2024-25"]
            )
            uncached = audit[
                audit["feature_family"].isin(
                    ["play_by_play_momentum", "shot_profile", "hustle_effort"]
                )
            ]
            self.assertTrue(
                uncached["historical_game_coverage_pct"].eq(0).all()
            )
            self.assertTrue(uncached["key_field_missingness_pct"].eq(100).all())

    def test_priorities_and_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            processed = root / "processed"
            doc = root / "research.md"
            raw.mkdir()
            self._write_team_log(
                raw / "team_game_log_2024-25_regular_season_1.csv",
                ["1"],
            )
            audit, priorities = run_feature_source_audit(
                raw_dir=raw,
                processed_dir=processed,
                doc_path=doc,
            )
            self.assertFalse(audit.empty)
            self.assertEqual(
                len(build_family_priorities(audit)), len(FAMILY_CONFIG)
            )
            self.assertTrue(
                (processed / "nba_api_feature_source_audit.csv").exists()
            )
            self.assertTrue(
                (processed / "nba_api_feature_family_priorities.csv").exists()
            )
            self.assertTrue(doc.exists())
            self.assertIn("No source should move to modeling", doc.read_text())


if __name__ == "__main__":
    unittest.main()
