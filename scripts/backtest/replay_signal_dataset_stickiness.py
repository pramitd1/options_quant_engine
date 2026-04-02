from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine_runner import run_preloaded_engine_snapshot
from data.replay_loader import load_option_chain_snapshot, load_spot_snapshot, resolve_nearest_replay_snapshot_paths
from news.models import HeadlineIngestionState


DATASET_PATH = REPO_ROOT / "research" / "signal_evaluation" / "signals_dataset_cumul_backfilled.csv"
OUTPUT_DIR = REPO_ROOT / "research" / "reviews" / "signal_dataset_stickiness_2026-04-02_cumul_replay"
LARGE_MOVE_THRESHOLDS = (150.0, 200.0)
FLIP_BACK_LOOKAHEAD = 2
MAX_SPOT_DELTA_SECONDS = 7200.0
MAX_CHAIN_DELTA_SECONDS = 14400.0


@dataclass(frozen=True)
class ReplayConfig:
    dataset_path: Path = DATASET_PATH
    output_dir: Path = OUTPUT_DIR


def _resolve_saved_path(raw_path: Any, *, kind: str) -> Path | None:
    if raw_path is None or (isinstance(raw_path, float) and pd.isna(raw_path)):
        return None
    text = str(raw_path).strip()
    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.exists():
        return candidate

    subdir = "spot_snapshots" if kind == "spot" else "option_chain_snapshots"
    fallback = REPO_ROOT / "debug_samples" / subdir / candidate.name
    if fallback.exists():
        return fallback

    fallback = REPO_ROOT / text
    if fallback.exists():
        return fallback

    return None


def _neutral_headline_state(as_of: Any) -> HeadlineIngestionState:
    fetched_at = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(fetched_at):
        fetched_at = pd.Timestamp.now(tz="Asia/Kolkata")
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.tz_localize("Asia/Kolkata")
    return HeadlineIngestionState(
        records=[],
        provider_name="REPLAY_NEUTRAL",
        fetched_at=fetched_at,
        latest_headline_at=None,
        is_stale=True,
        data_available=False,
        neutral_fallback=True,
        stale_after_minutes=15,
        warnings=["headline_replay_neutralized"],
    )


def _neutral_global_market_snapshot(as_of: Any) -> dict[str, Any]:
    ts = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.now(tz="Asia/Kolkata")
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    return {
        "provider": "REPLAY_NEUTRAL",
        "as_of": ts.isoformat(),
        "data_available": False,
        "neutral_fallback": True,
        "issues": [],
        "warnings": ["global_market_replay_neutralized"],
        "stale": False,
        "latest_market_timestamp": ts.isoformat(),
        "market_inputs": {},
    }


def _macro_event_state(row: pd.Series) -> dict[str, Any]:
    return {
        "macro_event_risk_score": int(pd.to_numeric(row.get("macro_event_risk_score"), errors="coerce") or 0),
        "event_window_status": "NO_EVENT_DATA",
        "event_lockdown_flag": False,
        "event_data_available": False,
    }


def _prepare_dataset_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["symbol"] == "NIFTY") & (df["mode"] == "LIVE")].copy()
    df["signal_timestamp"] = pd.to_datetime(df["signal_timestamp"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.sort_values(["signal_timestamp", "created_at"]).reset_index(drop=True)
    collapsed = (
        df.groupby("signal_timestamp", as_index=False)
        .tail(1)
        .sort_values("signal_timestamp")
        .reset_index(drop=True)
    )
    return collapsed


def replay_current_engine(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_chain = None

    for row in frame.itertuples(index=False):
        spot_path = _resolve_saved_path(getattr(row, "saved_spot_snapshot_path", None), kind="spot")
        chain_path = _resolve_saved_path(getattr(row, "saved_chain_snapshot_path", None), kind="chain")
        if spot_path is None or chain_path is None:
            nearest = resolve_nearest_replay_snapshot_paths(
                str(getattr(row, "symbol", "NIFTY") or "NIFTY").upper().strip(),
                target_timestamp=getattr(row, "signal_timestamp", None),
                replay_dir=str(REPO_ROOT / "debug_samples"),
                source_label=str(getattr(row, "source", "ICICI") or "ICICI").upper().strip(),
                max_spot_delta_seconds=MAX_SPOT_DELTA_SECONDS,
                max_chain_delta_seconds=MAX_CHAIN_DELTA_SECONDS,
            )
            if spot_path is None and nearest.get("spot_path"):
                spot_path = Path(nearest["spot_path"])
            if chain_path is None and nearest.get("chain_path"):
                chain_path = Path(nearest["chain_path"])
        if spot_path is None or chain_path is None:
            rows.append(
                {
                    "signal_timestamp": getattr(row, "signal_timestamp", None),
                    "captured_direction": getattr(row, "direction", None),
                    "captured_trade_status": getattr(row, "trade_status", None),
                    "replay_ok": False,
                    "replay_error": "missing_saved_snapshot",
                    "direction": None,
                    "trade_status": None,
                    "expansion_mode": False,
                    "breakout_evidence": None,
                    "signed_return_30m_bps": getattr(row, "signed_return_30m_bps", None),
                    "spot_at_signal": getattr(row, "spot_at_signal", None),
                }
            )
            continue

        try:
            spot_snapshot = load_spot_snapshot(str(spot_path))
            option_chain = load_option_chain_snapshot(str(chain_path))
            result = run_preloaded_engine_snapshot(
                symbol=str(getattr(row, "symbol", "NIFTY") or "NIFTY").upper().strip(),
                mode="REPLAY",
                source="DATASET_REPLAY",
                spot_snapshot=spot_snapshot,
                option_chain=option_chain,
                previous_chain=previous_chain,
                apply_budget_constraint=False,
                requested_lots=1,
                lot_size=1,
                max_capital=float("inf"),
                capture_signal_evaluation=False,
                enable_shadow_logging=False,
                headline_state=_neutral_headline_state(getattr(row, "signal_timestamp", None)),
                macro_event_state=_macro_event_state(pd.Series(row._asdict())),
                global_market_snapshot=_neutral_global_market_snapshot(getattr(row, "signal_timestamp", None)),
            )
            previous_chain = option_chain.copy()
        except Exception as exc:
            rows.append(
                {
                    "signal_timestamp": getattr(row, "signal_timestamp", None),
                    "captured_direction": getattr(row, "direction", None),
                    "captured_trade_status": getattr(row, "trade_status", None),
                    "replay_ok": False,
                    "replay_error": f"{type(exc).__name__}:{exc}",
                    "direction": None,
                    "trade_status": None,
                    "expansion_mode": False,
                    "breakout_evidence": None,
                    "signed_return_30m_bps": getattr(row, "signed_return_30m_bps", None),
                    "spot_at_signal": getattr(row, "spot_at_signal", None),
                }
            )
            continue

        trade = (result.get("execution_trade") or result.get("trade") or {}) if isinstance(result, dict) else {}
        rows.append(
            {
                "signal_timestamp": getattr(row, "signal_timestamp", None),
                "captured_direction": getattr(row, "direction", None),
                "captured_trade_status": getattr(row, "trade_status", None),
                "replay_ok": bool(result.get("ok", False)) if isinstance(result, dict) else False,
                "replay_error": None if bool(result.get("ok", False)) else (result.get("error") if isinstance(result, dict) else "unknown_error"),
                "direction": trade.get("direction"),
                "trade_status": trade.get("trade_status"),
                "direction_source": trade.get("direction_source"),
                "expansion_mode": bool(trade.get("expansion_mode", False)),
                "expansion_direction": trade.get("expansion_direction"),
                "breakout_evidence": trade.get("breakout_evidence"),
                "reversal_stage": trade.get("reversal_stage"),
                "signed_return_30m_bps": getattr(row, "signed_return_30m_bps", None),
                "spot_at_signal": getattr(row, "spot_at_signal", None),
            }
        )

    replayed = pd.DataFrame(rows)
    replayed["signal_timestamp"] = pd.to_datetime(replayed["signal_timestamp"], errors="coerce")
    replayed["spot_at_signal"] = pd.to_numeric(replayed.get("spot_at_signal"), errors="coerce")
    replayed["signed_return_30m_bps"] = pd.to_numeric(replayed.get("signed_return_30m_bps"), errors="coerce")
    return replayed.sort_values("signal_timestamp").reset_index(drop=True)


def _directional_frame(frame: pd.DataFrame, *, direction_col: str, trade_status_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["dir"] = out[direction_col].astype(str).str.upper()
    out["trade_status_eval"] = out[trade_status_col].astype(str).str.upper()
    out = out[out["dir"].isin(["CALL", "PUT"])].copy().reset_index(drop=True)
    return out


def _metric_block(frame: pd.DataFrame, *, direction_col: str, trade_status_col: str) -> dict[str, Any]:
    df = _directional_frame(frame, direction_col=direction_col, trade_status_col=trade_status_col)
    if df.empty:
        metrics = {
            "directional_rows": 0,
            "reversal_events": 0,
            "reversal_latency_mean_snapshots": None,
            "reversal_latency_median_snapshots": None,
            f"flip_back_rate_{FLIP_BACK_LOOKAHEAD}snap": None,
        }
        for threshold in LARGE_MOVE_THRESHOLDS:
            metrics[f"large_move_events_30m_ge_{int(threshold)}pts"] = 0
            metrics[f"missed_large_move_30m_ge_{int(threshold)}pts"] = 0
            metrics[f"captured_large_move_30m_ge_{int(threshold)}pts"] = 0
        return metrics

    df["prev_dir"] = df["dir"].shift(1)
    df["is_reversal"] = df["prev_dir"].notna() & df["dir"].ne(df["prev_dir"])
    reversal_idx = df.index[df["is_reversal"]].tolist()

    latencies = []
    run_length = 1
    for i in range(1, len(df)):
        if df.loc[i, "dir"] == df.loc[i - 1, "dir"]:
            run_length += 1
        else:
            latencies.append(run_length)
            run_length = 1

    flip_back_flags = []
    for idx in reversal_idx:
        curr_dir = df.loc[idx, "dir"]
        future_dirs = [
            df.loc[j, "dir"]
            for j in range(idx + 1, min(len(df), idx + 1 + FLIP_BACK_LOOKAHEAD))
        ]
        flip_back_flags.append(any(future_dir != curr_dir for future_dir in future_dirs))

    metrics = {
        "directional_rows": int(len(df)),
        "reversal_events": int(len(reversal_idx)),
        "reversal_latency_mean_snapshots": round(float(pd.Series(latencies).mean()), 3) if latencies else None,
        "reversal_latency_median_snapshots": round(float(pd.Series(latencies).median()), 3) if latencies else None,
        f"flip_back_rate_{FLIP_BACK_LOOKAHEAD}snap": round(float(pd.Series(flip_back_flags).mean()), 4) if flip_back_flags else None,
    }

    for threshold in LARGE_MOVE_THRESHOLDS:
        large_events = 0
        missed = 0
        captured = 0
        for idx in reversal_idx:
            spot = pd.to_numeric(df.loc[idx, "spot_at_signal"], errors="coerce")
            move_bps = pd.to_numeric(df.loc[idx, "signed_return_30m_bps"], errors="coerce")
            if pd.isna(spot) or pd.isna(move_bps):
                continue
            move_points = float(spot) * float(move_bps) / 10000.0
            if abs(move_points) < float(threshold):
                continue

            large_events += 1
            realized_direction = "CALL" if move_points > 0 else "PUT"
            captured_flag = bool(
                df.loc[idx, "trade_status_eval"] == "TRADE"
                and df.loc[idx, "dir"] == realized_direction
            )
            if captured_flag:
                captured += 1
            else:
                missed += 1

        metrics[f"large_move_events_30m_ge_{int(threshold)}pts"] = int(large_events)
        metrics[f"missed_large_move_30m_ge_{int(threshold)}pts"] = int(missed)
        metrics[f"captured_large_move_30m_ge_{int(threshold)}pts"] = int(captured)

    return metrics


def build_summary(captured: pd.DataFrame, replayed: pd.DataFrame) -> pd.DataFrame:
    latest_ts = replayed["signal_timestamp"].max()
    captured_eval = captured.copy()
    captured_eval["captured_direction"] = captured_eval["direction"]
    captured_eval["captured_trade_status"] = captured_eval["trade_status"]
    windows = {
        "all_directional_rows": replayed,
        "trade_only_rows": replayed[replayed["trade_status"].astype(str).str.upper().eq("TRADE")].copy(),
        "last_14d_all": replayed[replayed["signal_timestamp"] >= latest_ts - pd.Timedelta(days=14)].copy(),
        "last_30d_all": replayed[replayed["signal_timestamp"] >= latest_ts - pd.Timedelta(days=30)].copy(),
    }
    rows = []
    for scope_name, scope_frame in windows.items():
        rows.append({"stream": "captured_engine_stream", "scope": scope_name, **_metric_block(captured_eval.loc[scope_frame.index], direction_col="captured_direction", trade_status_col="captured_trade_status")})
        rows.append({"stream": "replayed_current_engine", "scope": scope_name, **_metric_block(scope_frame, direction_col="direction", trade_status_col="trade_status")})
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, replayed: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "signal_dataset_cumul_replay_stickiness_summary.csv"
    report_json_path = output_dir / "signal_dataset_cumul_replay_stickiness_report.json"
    report_md_path = output_dir / "signal_dataset_cumul_replay_stickiness_report.md"
    rows_path = output_dir / "signal_dataset_cumul_replay_rows.csv"

    summary.to_csv(summary_path, index=False)
    replayed.to_csv(rows_path, index=False)

    payload = {
        "dataset_path": str(DATASET_PATH),
        "report_type": "cumulative_dataset_snapshot_replay",
        "notes": [
            "Saved spot and chain snapshots were replayed through the current engine.",
            "Historical real-time headlines and historical global-market snapshots were neutralized during replay to avoid present-time data leakage.",
            "30-minute move outcomes were taken from the captured cumulative dataset labels.",
        ],
        "summary_rows": summary.to_dict(orient="records"),
        "replay_success_rate": round(float(replayed["replay_ok"].fillna(False).mean()), 4) if not replayed.empty else None,
        "replay_failures": replayed[replayed["replay_ok"].fillna(False) == False][["signal_timestamp", "replay_error"]].head(25).to_dict(orient="records"),
    }
    report_json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# Cumulative Dataset Stickiness Replay Report",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        f"- Summary CSV: `{summary_path}`",
        f"- Replay rows: `{rows_path}`",
        "- Historical headline and global-market feeds were neutralized during replay to prevent real-time leakage.",
        "",
        "## Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
    ]
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    cfg = ReplayConfig()
    captured = _prepare_dataset_frame(cfg.dataset_path)
    replayed = replay_current_engine(captured)
    summary = build_summary(captured, replayed)
    write_report(summary, replayed, cfg.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()