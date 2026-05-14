#!/usr/bin/env python3
"""Send optional notification transport for the latest signal drift alert."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.drift_notifications import (  # noqa: E402
    NOTIFY_COMMAND_ENV_VAR,
    SUCCESS_STATUSES,
    WEBHOOK_ENV_VAR,
    default_alert_json_path,
    default_delivery_ledger_path,
    send_signal_drift_alert_notification,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send optional notification for signal drift alerts.")
    parser.add_argument("--alert-json", type=Path, default=default_alert_json_path())
    parser.add_argument("--delivery-ledger", type=Path, default=default_delivery_ledger_path())
    parser.add_argument(
        "--webhook-url",
        default="",
        help=f"Optional webhook URL. Can also be set with {WEBHOOK_ENV_VAR}.",
    )
    parser.add_argument(
        "--notify-command",
        default="",
        help=(
            "Optional local shell command. Environment includes SIGNAL_DRIFT_OPS_STATUS, "
            "SIGNAL_DRIFT_ALERT_JSON, and SIGNAL_DRIFT_REPORT_JSON. "
            f"Can also be set with {NOTIFY_COMMAND_ENV_VAR}."
        ),
    )
    parser.add_argument("--webhook-timeout-s", type=float, default=8.0)
    parser.add_argument("--include-watch", action="store_true", help="Notify for WATCH as well as DETERIORATING.")
    parser.add_argument("--dry-run", action="store_true", help="Record intended notification without sending.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when a notify-worthy alert is not delivered.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    webhook_url = args.webhook_url or os.getenv(WEBHOOK_ENV_VAR, "")
    notify_command = args.notify_command or os.getenv(NOTIFY_COMMAND_ENV_VAR, "")
    artifact = send_signal_drift_alert_notification(
        alert_json_path=args.alert_json,
        delivery_ledger_path=args.delivery_ledger,
        webhook_url=webhook_url,
        notify_command=notify_command,
        include_watch=args.include_watch,
        dry_run=args.dry_run,
        timeout_s=args.webhook_timeout_s,
    )
    payload = {
        "alert_json_path": artifact["alert_json_path"],
        "delivery_ledger_path": artifact["delivery_ledger_path"],
        "should_notify": artifact["should_notify"],
        "delivery_status": artifact["delivery_status"],
        "ops_status": artifact["notification_payload"].get("ops_status"),
        "trend_assessment": artifact["notification_payload"].get("trend_assessment"),
        "webhook": artifact["webhook"],
        "notify_command": artifact["notify_command"],
    }
    print(json.dumps(payload, indent=2, default=str))
    if args.strict and artifact["should_notify"] and artifact["delivery_status"] not in SUCCESS_STATUSES:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
