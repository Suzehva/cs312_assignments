"""Offline checks for the P1(a) data-size plot."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.a2.plot_p1a import (
    body_compute_pflop,
    d8_parameter_counts,
    fit_learning_rate_curve,
    fit_optimal_lr_rule,
    plot_p1a,
    plot_p1a_analysis,
    tokens_from_body_compute_pflop,
)
from experiments.a2.provided_sweeps import load


class P1aPlotTests(unittest.TestCase):
    def test_d8_counts_and_compute_conversion(self):
        total, body = d8_parameter_counts()
        self.assertEqual(total, 34_612_736)
        self.assertEqual(body, 30_418_432)
        budgets = np.array([153_600_000, 307_200_000, 614_400_000])
        compute = body_compute_pflop(budgets, body)
        np.testing.assert_allclose(compute, [28.0336269312, 56.0672538624, 112.1345077248])
        np.testing.assert_allclose(tokens_from_body_compute_pflop(compute, body), budgets)

    def test_plot_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            output = plot_p1a(Path(folder) / "p1a.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_learning_rate_fits_and_scaling_rule(self):
        rows = load("P1a")
        budgets = [153_600_000, 307_200_000, 614_400_000]
        fits = [fit_learning_rate_curve(rows, tokens) for tokens in budgets]
        np.testing.assert_allclose(
            [fit.optimal_learning_rate for fit in fits],
            [0.0023775575, 0.0029785011, 0.0031920751],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            [fit.coefficients for fit in fits],
            [
                [0.0457844734, 0.0307197571, 3.2291338444],
                [0.0149586201, 0.0003104210, 3.0562233925],
                [0.0184122324, -0.0032969713, 2.9256360531],
            ],
            rtol=1e-7,
        )
        eta_ref, beta = fit_optimal_lr_rule(fits, reference_tokens=614_400_000)
        self.assertAlmostEqual(eta_ref, 0.0032762095, places=10)
        self.assertAlmostEqual(beta, 0.2125071745, places=9)

    def test_analysis_plot_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            output = plot_p1a_analysis(Path(folder) / "p1a_analysis.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
