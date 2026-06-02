import unittest
from unittest.mock import patch

import app


class AppHeaderTests(unittest.TestCase):
    def test_render_header_includes_static_model_status_card(self):
        with patch.object(app.st, "markdown") as markdown:
            app.render_header()

        markdown.assert_called_once()
        rendered = markdown.call_args.args[0]
        self.assertIn('class="dashboard-header"', rendered)
        self.assertIn("Game Predictor Dashboard", rendered)
        self.assertIn("Model Status", rendered)
        self.assertIn('class="model-status-pill" data-tone="warm">Calibrated RF', rendered)
        self.assertIn('class="model-status-pill" data-tone="cool">Order-symmetric', rendered)
        self.assertIn("No odds • injuries • lineup/news", rendered)
        self.assertIn("Details", rendered)
        self.assertTrue(markdown.call_args.kwargs["unsafe_allow_html"])


if __name__ == "__main__":
    unittest.main()
