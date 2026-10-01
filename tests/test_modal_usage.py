"""Offline tests for assignment-scoped Modal GPU usage reporting."""

from argparse import Namespace
from datetime import date
from decimal import Decimal
import unittest
from unittest.mock import patch

from scripts.modal_usage import (
    ASSIGNMENT_START,
    format_duration,
    run_modal_billing_command,
    summarize_usage,
)


class ModalUsageTests(unittest.TestCase):
    def test_converts_gpu_cost_to_billed_time_and_groups_apps(self):
        report = [
            {
                "object_id": "ap-smoke",
                "description": "smoke",
                "resource": "H100",
                "cost": "0.90849990",
            },
            {
                "object_id": "ap-smoke",
                "description": "smoke",
                "resource": "CPU",
                "cost": "0.01156191",
            },
            {
                "object_id": "ap-other",
                "description": "other",
                "resource": "L40S",
                "cost": "0.975",
            },
        ]
        rates = {
            "gpu_hour_cost_h100": "3.95",
            "gpu_hour_cost_l40s": "1.95",
        }

        summary = summarize_usage(report, rates)

        self.assertEqual(format_duration(summary["gpu_hours"]["H100"]), "13m 48s")
        self.assertEqual(format_duration(summary["gpu_hours"]["L40S"]), "30m")
        self.assertEqual(
            format_duration(summary["app_gpu_hours"][("ap-smoke", "smoke")]["H100"]),
            "13m 48s",
        )
        self.assertEqual(summary["total_cost"], Decimal("1.89506181"))

    def test_default_window_starts_on_tuesday_september_29(self):
        args = Namespace(env="cs312-test", start=ASSIGNMENT_START)
        responses = [
            [{"object_id": "old"}],
            [{"object_id": "today"}],
        ]
        with patch("scripts.modal_usage.run_json_command", side_effect=responses) as run:
            report = run_modal_billing_command(args, today=date(2026, 9, 30))

        self.assertEqual([row["object_id"] for row in report], ["old", "today"])
        historical = run.call_args_list[0].args[0]
        current = run.call_args_list[1].args[0]
        self.assertEqual(historical[historical.index("--start") + 1], "2026-09-29")
        self.assertEqual(historical[historical.index("--end") + 1], "2026-09-30")
        self.assertEqual(current[current.index("--for") + 1], "today")

    def test_duration_rounds_to_nearest_second(self):
        self.assertEqual(format_duration(Decimal("48")), "48h")
        self.assertEqual(format_duration(Decimal("0")), "0s")
        self.assertEqual(format_duration(Decimal("749") / 3600), "12m 29s")


if __name__ == "__main__":
    unittest.main()
