import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app import (
    PREDICTION_MODE_PLAYOFF,
    _cached_default_matchup_prediction,
    _cached_processed_csv,
    _compute_matchup_prediction_impl,
    _read_processed_csv,
    raw_data_cache_version,
)


class AppCachingTests(unittest.TestCase):
    def test_processed_csv_loader_uses_path_version_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_text("value\n1\n")
            _cached_processed_csv.clear()

            with patch("app.pd.read_csv", wraps=pd.read_csv) as read_csv:
                first = _read_processed_csv(path)
                second = _read_processed_csv(path)

                self.assertEqual(read_csv.call_count, 1)
                pd.testing.assert_frame_equal(first, second)

                path.write_text("value\n2\n")
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns + 1_000_000_000, stat.st_mtime_ns + 1_000_000_000))
                third = _read_processed_csv(path)

                self.assertEqual(read_csv.call_count, 2)
                self.assertEqual(int(third.iloc[0]["value"]), 2)

    def test_cached_default_matchup_prediction_matches_uncached_result(self):
        def fake_predictor(**kwargs):
            probability = 0.64 if kwargs["team_a"] == "NYK" else 0.42
            features = {
                "home_team_A": 0.0 if kwargs["home_team"] == "team2" else 1.0,
                "series_score_diff": kwargs["team_a_series_wins"] - kwargs["team_b_series_wins"],
            }
            return probability, features, None, None, 5

        kwargs = {
            "team_a": "NYK",
            "team_b": "CLE",
            "season": "2025-26",
            "prediction_date": "2026-05-25",
            "home_team": "team2",
            "feature_season_type": "Regular Season",
            "prediction_context_mode": PREDICTION_MODE_PLAYOFF,
            "game_number": 4,
            "team_a_series_wins": 3,
            "team_b_series_wins": 0,
        }
        expected = _compute_matchup_prediction_impl(**kwargs, predictor=fake_predictor)
        _cached_default_matchup_prediction.clear()

        with patch("app._predict_probability", side_effect=fake_predictor):
            actual = _cached_default_matchup_prediction(
                **kwargs,
                model_version=(1, 100),
                raw_data_version=(1, ()),
            )

        self.assertEqual(actual["team_a_probability"], expected["team_a_probability"])
        self.assertEqual(actual["team_b_probability"], expected["team_b_probability"])
        self.assertEqual(actual["team_a_series_probability"], expected["team_a_series_probability"])
        self.assertEqual(actual["features"], expected["features"])

    def test_selected_prediction_cache_invalidates_when_raw_data_version_changes(self):
        calls = []

        def fake_predictor(**kwargs):
            calls.append(kwargs)
            return 0.6, {"home_team_A": 0.0}, None, None, 5

        kwargs = {
            "team_a": "NYK",
            "team_b": "CLE",
            "season": "2025-26",
            "prediction_date": "2026-05-25",
            "home_team": "team2",
            "feature_season_type": "Regular Season",
            "prediction_context_mode": PREDICTION_MODE_PLAYOFF,
            "game_number": 4,
            "team_a_series_wins": 3,
            "team_b_series_wins": 0,
            "model_version": (1, 100),
        }
        _cached_default_matchup_prediction.clear()

        with patch("app._predict_probability", side_effect=fake_predictor):
            first = _cached_default_matchup_prediction(**kwargs, raw_data_version=(1, ()))
            second = _cached_default_matchup_prediction(**kwargs, raw_data_version=(1, ()))
            refreshed = _cached_default_matchup_prediction(**kwargs, raw_data_version=(2, ()))

        self.assertEqual(first["team_a_probability"], second["team_a_probability"])
        self.assertEqual(first["team_a_probability"], refreshed["team_a_probability"])
        self.assertEqual(len(calls), 4)

    def test_raw_data_version_changes_when_relevant_file_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            path = cache_dir / "team_base_2025-26_regular_season.csv"
            path.write_text("value\n1\n")
            first = raw_data_cache_version("2025-26", "Regular Season", cache_dir, now=100)

            path.write_text("value\n100\n")
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns + 1_000_000_000, stat.st_mtime_ns + 1_000_000_000))
            second = raw_data_cache_version("2025-26", "Regular Season", cache_dir, now=100)

        self.assertNotEqual(first, second)

    def test_current_season_raw_data_version_changes_at_ttl_boundary(self):
        with patch("app.is_current_nba_season", return_value=True):
            first = raw_data_cache_version(
                "2025-26",
                "Regular Season",
                cache_dir=Path("/missing"),
                now=1,
            )
            second = raw_data_cache_version(
                "2025-26",
                "Regular Season",
                cache_dir=Path("/missing"),
                now=1 + 90 * 60,
            )

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
