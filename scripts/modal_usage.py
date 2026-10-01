"""Print Modal GPU usage for the current assignment window."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from utils import config_str


ASSIGNMENT_START = "2026-09-29"
ASSIGNMENT_GPU_HOUR_BUDGET = Decimal("48")
REPORT_TIMEZONE = "America/Los_Angeles"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Modal GPU usage since the assignment started."
    )
    parser.add_argument(
        "--env",
        default=config_str(
            "CONFIG_MODAL_ENVIRONMENT",
            "DL_ALCHEMY_MODAL_ENVIRONMENT",
            "MODAL_ENVIRONMENT",
        ),
        help=(
            "Modal environment to inspect. Defaults to utils.py, then "
            "Modal's active profile or workspace default."
        ),
    )
    parser.add_argument(
        "--start",
        default=ASSIGNMENT_START,
        help=f"Assignment start date in YYYY-MM-DD form (default: {ASSIGNMENT_START}).",
    )
    parser.add_argument(
        "--budget-hours",
        type=Decimal,
        default=ASSIGNMENT_GPU_HOUR_BUDGET,
        help=f"GPU-hour budget (default: {ASSIGNMENT_GPU_HOUR_BUDGET}).",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Print the combined billing report and current rates as JSON.",
    )
    return parser.parse_args(argv)


def run_json_command(command: list[str]) -> dict | list:
    result = subprocess.run(command, capture_output=True, check=True, text=True)
    return json.loads(result.stdout)


def _billing_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "modal",
        "environment",
        "billing",
        "report",
        "--json",
        "--show-resources",
        "--resolution",
        "h",
        "--tz",
        REPORT_TIMEZONE,
    ]
    if args.env:
        command.append(args.env)
    return command


def run_modal_billing_command(
    args: argparse.Namespace, *, today: date | None = None
) -> list[dict[str, object]]:
    """Fetch complete prior days plus today's still-updating billing interval."""
    today = today or date.today()
    try:
        start = date.fromisoformat(args.start)
    except ValueError as exc:
        raise SystemExit(f"Invalid --start date {args.start!r}; use YYYY-MM-DD.") from exc
    if start > today:
        raise SystemExit(f"Assignment start {start} is after today ({today}).")

    report: list[dict[str, object]] = []
    base_command = _billing_command(args)
    if start < today:
        historical = run_json_command(
            base_command
            + ["--start", start.isoformat(), "--end", today.isoformat()]
        )
        if not isinstance(historical, list):
            raise SystemExit("Expected a JSON list from Modal's historical report.")
        report.extend(historical)

    current = run_json_command(base_command + ["--for", "today"])
    if not isinstance(current, list):
        raise SystemExit("Expected a JSON list from Modal's current-day report.")
    report.extend(current)
    return report


def run_modal_rates_command() -> dict[str, object]:
    payload = run_json_command(
        [sys.executable, "-m", "modal", "billing", "rates", "--json"]
    )
    if not isinstance(payload, Mapping):
        raise SystemExit("Expected a JSON object from Modal's rates command.")
    return dict(payload)


def decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SystemExit(f"Modal returned a non-numeric cost or rate: {value!r}.") from exc


def money(value: object) -> str:
    return f"${decimal(value):.4f}"


def normalize_resource_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def gpu_rates_by_resource(rates: Mapping[str, object]) -> dict[str, Decimal]:
    prefix = "gpu_hour_cost_"
    return {
        normalize_resource_name(key.removeprefix(prefix)): decimal(value)
        for key, value in rates.items()
        if key.startswith(prefix)
    }


def format_duration(hours: Decimal) -> str:
    seconds = int(
        (hours * Decimal(3600)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    hours_part, remainder = divmod(max(0, seconds), 3600)
    minutes_part, seconds_part = divmod(remainder, 60)
    parts = []
    if hours_part:
        parts.append(f"{hours_part}h")
    if minutes_part:
        parts.append(f"{minutes_part}m")
    if seconds_part or not parts:
        parts.append(f"{seconds_part}s")
    return " ".join(parts)


def summarize_usage(
    report: Sequence[Mapping[str, object]], rates: Mapping[str, object]
) -> dict[str, object]:
    resource_costs: defaultdict[str, Decimal] = defaultdict(Decimal)
    app_resource_costs: defaultdict[tuple[str, str], defaultdict[str, Decimal]] = (
        defaultdict(lambda: defaultdict(Decimal))
    )
    for row in report:
        resource = str(row.get("resource", "Unknown"))
        cost = decimal(row.get("cost", 0))
        resource_costs[resource] += cost
        app_key = (
            str(row.get("object_id", "unknown")),
            str(row.get("description", "unknown")),
        )
        app_resource_costs[app_key][resource] += cost

    gpu_rates = gpu_rates_by_resource(rates)
    gpu_hours: dict[str, Decimal] = {}
    for resource, cost in resource_costs.items():
        rate = gpu_rates.get(normalize_resource_name(resource))
        if rate:
            gpu_hours[resource] = cost / rate

    app_gpu_hours: dict[tuple[str, str], dict[str, Decimal]] = {}
    for app_key, costs in app_resource_costs.items():
        converted = {}
        for resource, cost in costs.items():
            rate = gpu_rates.get(normalize_resource_name(resource))
            if rate:
                converted[resource] = cost / rate
        if converted:
            app_gpu_hours[app_key] = converted

    return {
        "total_cost": sum(resource_costs.values(), Decimal()),
        "resource_costs": dict(resource_costs),
        "gpu_rates": gpu_rates,
        "gpu_hours": gpu_hours,
        "app_gpu_hours": app_gpu_hours,
    }


def print_summary(
    summary: Mapping[str, object], args: argparse.Namespace, *, today: date | None = None
) -> None:
    today = today or date.today()
    gpu_hours = summary["gpu_hours"]
    resource_costs = summary["resource_costs"]
    gpu_rates = summary["gpu_rates"]
    app_gpu_hours = summary["app_gpu_hours"]
    assert isinstance(gpu_hours, Mapping)
    assert isinstance(resource_costs, Mapping)
    assert isinstance(gpu_rates, Mapping)
    assert isinstance(app_gpu_hours, Mapping)

    total_gpu_hours = sum(gpu_hours.values(), Decimal())
    remaining = max(Decimal(), args.budget_hours - total_gpu_hours)
    percent = (
        Decimal() if not args.budget_hours else total_gpu_hours / args.budget_hours * 100
    )

    env_text = args.env or "Modal default environment"
    print(f"Modal usage for {env_text}")
    print(f"Assignment window: {args.start} through {today.isoformat()}")
    print(f"Total metered cost: {money(summary['total_cost'])}")
    print(
        "Estimated billed GPU time: "
        f"{format_duration(total_gpu_hours)} / {format_duration(args.budget_hours)} "
        f"({percent:.2f}%)"
    )
    print(f"Estimated GPU time remaining: {format_duration(remaining)}")

    if gpu_hours:
        print("GPU breakdown:")
        for resource, hours in sorted(gpu_hours.items()):
            cost = resource_costs[resource]
            rate = gpu_rates[normalize_resource_name(resource)]
            print(
                f"  {resource}: {format_duration(hours)} "
                f"({money(cost)} at {money(rate)}/GPU-hour)"
            )

    if app_gpu_hours:
        print("GPU time by app:")
        sorted_apps = sorted(
            app_gpu_hours.items(),
            key=lambda item: sum(item[1].values(), Decimal()),
            reverse=True,
        )
        for (object_id, description), resources in sorted_apps:
            details = ", ".join(
                f"{resource} {format_duration(hours)}"
                for resource, hours in sorted(resources.items())
            )
            print(f"  {description} ({object_id}): {details}")

    print(
        "Note: billed GPU time includes container startup and shutdown, so it can "
        "be slightly longer than the training loop's elapsed time."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_modal_billing_command(args)
    rates = run_modal_rates_command()
    if args.raw_json:
        print(
            json.dumps(
                {
                    "assignment_start": args.start,
                    "budget_gpu_hours": str(args.budget_hours),
                    "report": report,
                    "rates": rates,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print_summary(summarize_usage(report, rates), args)


if __name__ == "__main__":
    main()
