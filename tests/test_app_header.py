import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app


class AppHeaderTests(unittest.TestCase):
    def test_render_header_includes_static_model_status_card(self):
        with (
            patch.object(app.st, "markdown") as markdown,
            patch.object(app.st, "code") as code,
            patch.object(app.st, "text") as text,
            patch.object(app.st, "write") as write,
            patch.object(app.st, "caption") as caption,
        ):
            app.render_header()

        self.assertEqual(markdown.call_count, 2)
        rendered = markdown.call_args_list[0].args[0]
        popup = markdown.call_args_list[1].args[0]

        self.assertIn('class="dashboard-header"', rendered)
        self.assertIn("Game Predictor Dashboard", rendered)
        self.assertIn("MODEL STATUS", rendered)
        self.assertIn('class="model-status-pill" data-tone="warm">Calibrated RF', rendered)
        self.assertIn('class="model-status-pill" data-tone="cool">Order-symmetric', rendered)
        self.assertIn("No odds • injuries • lineup/news", rendered)
        self.assertIn("Details", rendered)
        self.assertTrue(markdown.call_args_list[0].kwargs["unsafe_allow_html"])
        self.assertTrue(markdown.call_args_list[1].kwargs["unsafe_allow_html"])
        self.assertNotIn("&lt;div class=", rendered)

        for streamlit_text_call in (
            code.call_args_list + text.call_args_list + write.call_args_list + caption.call_args_list
        ):
            self.assertNotIn('<div class="model-status-card"', str(streamlit_text_call))
        self.assertIn(
            '<button type="button" class="model-details-button" popovertarget="model-details-popover">Details</button>',
            rendered,
        )
        self.assertNotIn('id="model-details-popover" class="model-details-popover" popover', rendered)
        self.assertNotIn("About this model", rendered)
        self.assertNotIn("Model facts", rendered)
        self.assertNotIn("model-hero-metric", rendered)

        self.assertIn('id="model-details-popover" class="model-details-popover" popover', popup)
        self.assertIn("About this model", popup)
        self.assertIn("Model facts", popup)
        self.assertIn('<span class="model-detail-label">Model</span><span class="model-detail-value">Calibrated RF</span>', popup)
        self.assertIn('<span class="model-detail-label">Method</span><span class="model-detail-value">Order-symmetric probabilities</span>', popup)
        self.assertIn('<span class="model-detail-label">Excludes</span><span class="model-detail-value">odds, injuries, lineup/news</span>', popup)
        self.assertIn("Pure basketball-stat matchup/series forecast", popup)
        self.assertIn("model-hero-metric", popup)
        self.assertIn("Top feature importances", popup)
        self.assertIn("Calibration", popup)
        self.assertNotIn("Model Overview", popup)
        self.assertNotIn("model-overview", popup)

        status_card_index = rendered.index('class="model-status-card"')
        status_top_start = rendered.index('<div class="model-status-top">', status_card_index)
        status_top = rendered[status_top_start:]
        self.assertIn('class="model-status-title">MODEL STATUS</div>', status_top)
        self.assertIn('class="model-details-button"', status_top)

    def test_render_header_model_status_card_html_is_not_visible_text(self):
        visible_output = []

        def capture_markdown(body, *args, **kwargs):
            if not kwargs.get("unsafe_allow_html"):
                visible_output.append(str(body))

        with patch.object(app.st, "markdown", side_effect=capture_markdown) as markdown:
            app.render_header()

        self.assertEqual(markdown.call_count, 2)
        rendered = markdown.call_args_list[0].args[0]
        popup = markdown.call_args_list[1].args[0]

        self.assertIn("MODEL STATUS", rendered)
        self.assertIn("Calibrated RF", rendered)
        self.assertIn("Order-symmetric", rendered)
        self.assertIn('id="model-details-popover" class="model-details-popover" popover', popup)
        self.assertTrue(markdown.call_args_list[0].kwargs["unsafe_allow_html"])
        self.assertTrue(markdown.call_args_list[1].kwargs["unsafe_allow_html"])
        self.assertNotIn("<div class=", "\n".join(visible_output))

    def test_dashboard_css_keeps_status_card_in_right_column(self):
        with patch.object(app.st, "markdown") as markdown:
            app.inject_dashboard_css()

        markdown.assert_called_once()
        rendered = markdown.call_args.args[0]

        self.assertIn(".dashboard-header {", rendered)
        self.assertIn("grid-template-columns: minmax(0, 1fr) minmax(240px, 320px);", rendered)
        self.assertIn(".model-status-card {", rendered)
        self.assertIn("align-self: center;", rendered)
        self.assertIn("width: min(840px, calc(100vw - 2rem));", rendered)
        self.assertIn("background:", rendered)
        self.assertIn("#0B1220", rendered)
        self.assertIn(".model-details-grid {", rendered)
        self.assertTrue(markdown.call_args.kwargs["unsafe_allow_html"])

    def test_model_details_metrics_are_clean_and_rounded(self):
        with TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "model_calibration.csv"
            metrics_path.write_text(
                "\n".join(
                    [
                        "model,calibration_method,feature_set,prediction_context_mode,accuracy,roc_auc,brier_score,log_loss,precision,recall,f1,expected_calibration_error",
                        "Random Forest,isotonic,baseline_plus_corrected_signs,Current Hypothetical,0.91234,0.82345,0.12345,0.54321,0.73456,0.84567,0.78901,0.06789",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(app, "MODEL_CALIBRATION_PATH", metrics_path):
                rendered = app._render_model_metrics_html()

        self.assertIn('<span class="model-hero-label">ROC AUC</span>', rendered)
        self.assertIn('<span class="model-hero-value">0.823</span>', rendered)
        self.assertIn("Validation performance", rendered)
        self.assertIn('<span class="model-detail-label">Accuracy</span><span class="model-detail-value">0.912</span>', rendered)
        self.assertIn('<span class="model-detail-label">Brier Score</span><span class="model-detail-value">0.123</span>', rendered)
        self.assertIn('<span class="model-detail-label">Log Loss</span><span class="model-detail-value">0.543</span>', rendered)
        self.assertIn('<span class="model-detail-label">Precision</span><span class="model-detail-value">0.735</span>', rendered)
        self.assertIn('<span class="model-detail-label">Recall</span><span class="model-detail-value">0.846</span>', rendered)
        self.assertIn('<span class="model-detail-label">F1</span><span class="model-detail-value">0.789</span>', rendered)
        self.assertIn('<span class="model-detail-label">ECE</span><span class="model-detail-value">0.068</span>', rendered)
        self.assertIn("model-metric-card", rendered)
        self.assertNotIn("roc_auc", rendered)
        self.assertNotIn("brier_score", rendered)

    def test_model_details_feature_importances_render_when_available(self):
        with TemporaryDirectory() as tmpdir:
            importances_path = Path(tmpdir) / "feature_importances.csv"
            importances_path.write_text(
                "\n".join(
                    [
                        "model,feature,importance,prediction_context_mode",
                        "Random Forest,NET_RATING_DIFF,0.600,Current Hypothetical",
                        "Random Forest,OFF_RATING_DIFF,0.500,Current Hypothetical",
                        "Random Forest,DEF_RATING_DIFF,0.400,Current Hypothetical",
                        "Random Forest,W_PCT_DIFF,0.300,Current Hypothetical",
                        "Random Forest,PLUS_MINUS_DIFF,0.200,Current Hypothetical",
                        "Random Forest,PACE_DIFF,0.100,Current Hypothetical",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(app, "FEATURE_IMPORTANCE_PATH", importances_path):
                rendered = app._render_feature_importance_html()

        self.assertIn("model-feature-list", rendered)
        self.assertIn("Net rating", rendered)
        self.assertIn("Offensive rating", rendered)
        self.assertIn("Defensive rating", rendered)
        self.assertIn("Season win percentage", rendered)
        self.assertIn("Point differential", rendered)
        self.assertIn("Pace", rendered)
        self.assertIn("0.600", rendered)
        self.assertEqual(rendered.count('class="model-feature-row"'), 6)
        self.assertNotIn("Feature importance data is unavailable", rendered)

    def test_model_details_calibration_renders_when_available(self):
        with TemporaryDirectory() as tmpdir:
            calibration_path = Path(tmpdir) / "model_calibration.csv"
            calibration_path.write_text(
                "\n".join(
                    [
                        "model,calibration_method,feature_set,prediction_context_mode,bin_lower,bin_upper,bin_count,mean_predicted_probability,observed_win_rate",
                        "Random Forest,isotonic,baseline_plus_corrected_signs,Current Hypothetical,0.0,0.1,12,0.0555,0.0833",
                        "Random Forest,isotonic,baseline_plus_corrected_signs,Current Hypothetical,0.1,0.2,20,0.1555,0.2500",
                        "Random Forest,isotonic,baseline_plus_corrected_signs,Current Hypothetical,0.2,0.3,18,0.2555,0.2777",
                        "Random Forest,isotonic,baseline_plus_corrected_signs,Current Hypothetical,0.3,0.4,14,0.3555,0.3888",
                        "Random Forest,isotonic,baseline_plus_corrected_signs,Current Hypothetical,0.4,0.5,10,0.4555,0.5000",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(app, "MODEL_CALIBRATION_PATH", calibration_path):
                rendered = app._render_calibration_html()

        self.assertIn("model-calibration-table", rendered)
        self.assertIn("Predicted probabilities are compared with observed win rates across bins.", rendered)
        self.assertIn("<span>Predicted</span><span>Observed</span><span>Count</span>", rendered)
        self.assertIn("0.056", rendered)
        self.assertIn("0.083", rendered)
        self.assertIn("<span>12</span>", rendered)
        self.assertEqual(rendered.count('class="model-calibration-row"'), 5)
        self.assertNotIn("0.456", rendered)
        self.assertNotIn("model-calibration-track", rendered)
        self.assertNotIn("Calibration bins are unavailable", rendered)

    def test_model_details_missing_files_are_safe(self):
        with TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.csv"
            with (
                patch.object(app, "MODEL_CALIBRATION_PATH", missing),
                patch.object(app, "MODEL_COMPARISON_PATH", missing),
                patch.object(app, "FEATURE_IMPORTANCE_PATH", missing),
                patch.object(app, "load_model", side_effect=RuntimeError("missing artifact")),
            ):
                rendered = app._render_model_details_popover()

        self.assertIn("About this model", rendered)
        self.assertIn("Validation metrics are unavailable", rendered)
        self.assertIn("Feature importance data is unavailable", rendered)
        self.assertIn("Calibration bins are unavailable", rendered)


if __name__ == "__main__":
    unittest.main()
