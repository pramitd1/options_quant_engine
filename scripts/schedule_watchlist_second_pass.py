import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def watchlist_realized_ready(db_path: str, target_date: str) -> dict:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        SELECT
          COUNT(*) total_watchlist,
          SUM(CASE WHEN signed_return_60m_bps IS NOT NULL THEN 1 ELSE 0 END) populated_60m,
          SUM(CASE WHEN signed_return_session_close_bps IS NOT NULL THEN 1 ELSE 0 END) populated_session
        FROM signals
        WHERE trade_status='WATCHLIST'
          AND signal_timestamp LIKE ?
        """,
        (f"{target_date}%",),
    )
    total_watchlist, populated_60m, populated_session = c.fetchone()
    conn.close()
    return {
        "total_watchlist": int(total_watchlist or 0),
        "populated_60m": int(populated_60m or 0),
        "populated_session": int(populated_session or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll until watchlist realized outcomes are available, then run second-pass evaluation")
    parser.add_argument("--target-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--check-every-minutes", type=int, default=15)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--db-path", default="research/signal_evaluation/signals_dataset_cumul.sqlite")
    parser.add_argument("--evaluation-script", default="scripts/watchlist_realized_evaluation.py")
    args = parser.parse_args()

    start_ts = time.time()
    timeout_seconds = max(args.timeout_hours * 3600.0, 60.0)
    check_seconds = max(args.check_every_minutes * 60, 60)

    sched_dir = Path("research/reviews/watchlist_realized_evaluation") / "scheduler_runs"
    sched_dir.mkdir(parents=True, exist_ok=True)
    status_file = sched_dir / f"second_pass_{args.target_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    status_payload = {
        "target_date": args.target_date,
        "started_at": datetime.now().isoformat(),
        "check_every_minutes": args.check_every_minutes,
        "timeout_hours": args.timeout_hours,
        "db_path": args.db_path,
        "evaluation_script": args.evaluation_script,
        "poll_history": [],
        "completed": False,
    }

    while (time.time() - start_ts) <= timeout_seconds:
        readiness = watchlist_realized_ready(args.db_path, args.target_date)
        poll_record = {
            "ts": datetime.now().isoformat(),
            **readiness,
        }
        status_payload["poll_history"].append(poll_record)

        if readiness["total_watchlist"] > 0 and readiness["populated_60m"] > 0:
            run = subprocess.run(
                [sys.executable, args.evaluation_script],
                capture_output=True,
                text=True,
                check=False,
            )
            status_payload["completed"] = True
            status_payload["trigger_reason"] = "populated_60m_available"
            status_payload["completed_at"] = datetime.now().isoformat()
            status_payload["evaluation_return_code"] = run.returncode
            status_payload["evaluation_stdout"] = run.stdout
            status_payload["evaluation_stderr"] = run.stderr
            status_file.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
            print(f"Second-pass evaluation triggered for {args.target_date}")
            print(status_file)
            return

        status_file.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        time.sleep(check_seconds)

    status_payload["completed"] = False
    status_payload["completed_at"] = datetime.now().isoformat()
    status_payload["trigger_reason"] = "timeout"
    status_file.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    print(f"Timeout reached without populated 60m outcomes for {args.target_date}")
    print(status_file)


if __name__ == "__main__":
    main()
