"""Offline regression checks for the Hyperball source-only analysis."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.a2.plot_p1d import BUDGETS, fit_sources, plot_p1d, source_summary


class P1dPlotTests(unittest.TestCase):
    def test_all_source_measurements_and_bracketed_optima(self):
        fits = fit_sources()
        self.assertEqual([f.tokens for f in fits["Hyperball"]], list(BUDGETS))
        self.assertEqual(sum(len(f.losses) for f in fits["Hyperball"]), 24)
        self.assertEqual(sum(len(f.losses) for f in fits["AdamW"]), 9)
        for curves in fits.values():
            for fit in curves:
                self.assertGreater(fit.coefficients[0], 0)
                self.assertGreater(fit.optimal_learning_rate, fit.learning_rates.min())
                self.assertLess(fit.optimal_learning_rate, fit.learning_rates.max())
        np.testing.assert_allclose(
            [f.optimal_learning_rate for f in fits["Hyperball"]],
            [0.0156100843709, 0.0122132751585, 0.0104158267297], rtol=1e-9,
        )

    def test_scaling_rules_and_fit_sensitivity(self):
        summary = source_summary(fit_sources())
        hyperball = summary["optimizers"]["Hyperball"]
        adamw = summary["optimizers"]["AdamW"]
        self.assertAlmostEqual(hyperball["eta_ref"], 0.0102672616031, places=12)
        self.assertAlmostEqual(hyperball["beta"], -0.2918504903976, places=10)
        self.assertAlmostEqual(hyperball["multiplier_per_doubling"], 0.816853639, places=8)
        self.assertAlmostEqual(adamw["beta"], 0.212507174509, places=10)
        self.assertAlmostEqual(
            summary["hyperball_local_sensitivity"]["beta"], -0.133473871859, places=10
        )

    def test_plot_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            output = plot_p1d(Path(folder) / "p1d.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
