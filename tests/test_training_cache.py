import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.nba_data import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, _processed_training_path, load_training_frame


def _cached_row(season: str, game_id: str, target: int) -> dict:
    row = {
        "FEATURE_SCHEMA_VERSION": FEATURE_SCHEMA_VERSION,
        "SEASON": season,
        "GAME_ID": game_id,
        "TEAM_A_WON": target,
        "GAME_DATE": "2024-04-20",
    }
    row.update({feature: 0.0 for feature in FEATURE_COLUMNS})
    return row


class TrainingCacheTests(unittest.TestCase):
    def test_loads_combined_processed_training_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir)
            cached = pd.DataFrame(
                [
                    _cached_row("2023-24", "1", 1),
                    _cached_row("2022-23", "2", 0),
                ]
            )
            cached.to_csv(_processed_training_path(processed_dir, "Regular Season"), index=False)

            result = load_training_frame(
                ["2023-24"],
                cache_dir=processed_dir / "raw",
                processed_dir=processed_dir,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["SEASON"], "2023-24")
            self.assertEqual(result.iloc[0]["GAME_ID"], 1)

    def test_old_feature_schema_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir)
            stale = pd.DataFrame([{"SEASON": "2023-24", "GAME_ID": "1", "TEAM_A_WON": 1, "GAME_DATE": "2024-04-20"}])
            stale.to_csv(_processed_training_path(processed_dir, "Regular Season"), index=False)

            rebuilt = pd.DataFrame([_cached_row("2023-24", "rebuilt", 1)])
            with patch("src.nba_data._build_training_frame_for_season", return_value=rebuilt) as builder:
                result = load_training_frame(
                    ["2023-24"],
                    cache_dir=processed_dir / "raw",
                    processed_dir=processed_dir,
                )

            builder.assert_called_once()
            self.assertEqual(result.iloc[0]["GAME_ID"], "rebuilt")


if __name__ == "__main__":
    unittest.main()
