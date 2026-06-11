import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.nba_data import (
    CURRENT_SEASON_CACHE_TTL_SECONDS,
    _read_or_fetch,
    raw_cache_ttl_seconds,
)


class RawDataCacheTests(unittest.TestCase):
    def test_current_season_expired_cache_refreshes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "current.csv"
            pd.DataFrame([{"value": 1}]).to_csv(path, index=False)
            now = path.stat().st_mtime + CURRENT_SEASON_CACHE_TTL_SECONDS + 1
            fetcher = Mock(return_value=pd.DataFrame([{"value": 2}]))

            with patch("src.nba_data.time.sleep"):
                result = _read_or_fetch(
                    path,
                    fetcher,
                    max_age_seconds=raw_cache_ttl_seconds("2025-26", date(2026, 6, 11)),
                    now=now,
                )

            fetcher.assert_called_once()
            self.assertEqual(int(result.iloc[0]["value"]), 2)
            self.assertEqual(int(pd.read_csv(path).iloc[0]["value"]), 2)

    def test_current_season_fresh_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "current.csv"
            pd.DataFrame([{"value": 1}]).to_csv(path, index=False)
            fetcher = Mock(return_value=pd.DataFrame([{"value": 2}]))

            result = _read_or_fetch(
                path,
                fetcher,
                max_age_seconds=raw_cache_ttl_seconds("2025-26", date(2026, 6, 11)),
                now=path.stat().st_mtime + CURRENT_SEASON_CACHE_TTL_SECONDS - 1,
            )

            fetcher.assert_not_called()
            self.assertEqual(int(result.iloc[0]["value"]), 1)

    def test_historical_cache_remains_long_lived(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "historical.csv"
            pd.DataFrame([{"value": 1}]).to_csv(path, index=False)
            os.utime(path, (1, 1))
            fetcher = Mock(return_value=pd.DataFrame([{"value": 2}]))

            result = _read_or_fetch(
                path,
                fetcher,
                max_age_seconds=raw_cache_ttl_seconds("2024-25", date(2026, 6, 11)),
                now=2_000_000_000,
            )

            fetcher.assert_not_called()
            self.assertEqual(int(result.iloc[0]["value"]), 1)


if __name__ == "__main__":
    unittest.main()
