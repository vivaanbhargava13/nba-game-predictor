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
            actual = _cached_default_matchup_prediction(**kwargs, model_version=(1, 100))

        self.assertEqual(actual["team_a_probability"], expected["team_a_probability"])
        self.assertEqual(actual["team_b_probability"], expected["team_b_probability"])
        self.assertEqual(actual["team_a_series_probability"], expected["team_a_series_probability"])
        self.assertEqual(actual["features"], expected["features"])


if __name__ == "__main__":
    unittest.main()
