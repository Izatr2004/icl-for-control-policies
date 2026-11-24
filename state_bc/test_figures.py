"""Tests for make_figures.py (needs only numpy and matplotlib).

Run: python -m unittest state_bc/test_figures.py -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

RUN = Path(__file__).resolve().parent / "results" / "transformer_v1_seed0"


class WilsonTest(unittest.TestCase):
    def test_known_values(self):
        lo, hi = mf.wilson(0.5, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_edges_stay_in_unit_interval(self):
        self.assertEqual(mf.wilson(0.0, 512)[0], 0.0)
        self.assertGreater(mf.wilson(0.0, 512)[1], 0.0)
        self.assertEqual(mf.wilson(1.0, 512)[1], 1.0)
        self.assertLess(mf.wilson(1.0, 512)[0], 1.0)


class CommittedResultsTest(unittest.TestCase):
    def test_training_loss_parsed_for_full_run_only(self):
        rows = mf.parse_train_loss(RUN / "notebook.log", "transformer_v1_seed0")
        self.assertEqual(rows[0][0], 500)
        self.assertEqual(rows[-1][0], 100_000)
        self.assertEqual(len(rows), 200)  # every 500 steps; the 3k-step pilot (logged every 250) is excluded

    def test_regenerates_every_output(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sys, "argv", ["make_figures.py", "--out", d]):
            mf.main()
            names = {p.name for p in Path(d).iterdir()}
            for stem in ("fig_training", "fig_success_vs_k", "fig_action_error_vs_k"):
                self.assertTrue({f"{stem}.pdf", f"{stem}.png"} <= names)
            md = (Path(d) / "tables.md").read_text(encoding="utf-8")
            self.assertIn("| Test, Q1–Q3 goals | PPO | once | 91.2 [88.4, 93.4]", md)
            self.assertIn("| Test, Q4 goals (held out) | PPO | at end | 67.0 [62.8, 70.9]", md)


if __name__ == "__main__":
    unittest.main()
