from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine_runner import run_preloaded_engine_snapshot
from backtest.holistic_backtest_runner import build_realized_spot_path, evaluate_eod_outcomes
from config.settings import (
    BACKTEST_ENABLE_BUDGET,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
    STOP_LOSS_PERCENT,
    TARGET_PROFIT_PERCENT,
)
from data.expiry_resolver import filter_option_chain_by_expiry, ordered_expiries
from data.historical_snapshot import get_available_dates, replay_historical_snapshot
from research.signal_evaluation.evaluator import build_signal_evaluation_row
from tuning.runtime import temporary_parameter_pack


BASELINE_LIKE_OVERRIDES = {
    "trade_strength.runtime_thresholds.min_composite_score": 0,
    "trade_strength.runtime_thresholds.max_intraday_hold_minutes": 240,
    "trade_strength.runtime_thresholds.toxic_regime_hold_cap_minutes": 240,
    "trade_strength.runtime_thresholds.provider_health_caution_blocks_trade": 0,
    "trade_strength.runtime_thresholds.at_flip_trade_strength_penalty": 0,
    "trade_strength.runtime_thresholds.at_flip_size_cap": 1.0,
    "trade_strength.runtime_thresholds.at_flip_toxic_size_cap": 1.0,
    "trade_strength.runtime_thresholds.regime_strength_add_at_flip": 0,
    "trade_strength.runtime_thresholds.regime_strength_add_toxic": 0,
    "trade_strength.runtime_thresholds.regime_composite_add_at_flip": 0,
    "trade_strength.runtime_thresholds.regime_composite_add_toxic": 0,
}

EXTRA_FIELDS = [
    "runtime_composite_score",
    "min_trade_strength_threshold",
    "min_composite_score_threshold",
    "provider_health_summary",
    "call_put_alignment",
    "call_put_imbalance_score",
    "call_put_imbalance_severity",
    "at_flip_trade_strength_penalty",
    "at_flip_size_cap",
    "at_flip_toxic_context",
    "regime_toxic_context",
    "regime_threshold_adjustments",
    "effective_size_cap",
    "recommended_hold_minutes",
    "max_hold_minutes",
]


@dataclass
class RunConfig:
    label: str
    symbol: str
    start_date: date
    end_date: date
    max_expiries: int = 3
    prediction_method: str = "blended"
    target_profit_percent: float = TARGET_PROFIT_PERCENT
    stop_loss_percent: float = STOP_LOSS_PERCENT
    compute_iv: bool = True
    include_global_market: bool = True
    include_macro_events: bool = True
    min_quality_score: float = 40.0
    evaluate_outcomes: bool = True
    overrides: dict[str, Any] | None = None


def _coerce_serializable(value: Any) -> Any:
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(_coerce_serializable)
    return out


def _activate_prediction_method(prediction_method: str | None):
    if not prediction_method:
        return nullcontext()

    class _PredictionMethodContext:
        def __enter__(self):
            from engine.predictors import factory as _pf

            self._factory = _pf
            self._saved_predictor = _pf._ACTIVE_PREDICTOR
            self._saved_override = _pf._METHOD_OVERRIDE
            registry = _pf._ensure_registry()
            cls = registry.get(prediction_method)
            if cls is None:
                raise ValueError(
                    f"Unknown prediction_method: {prediction_method!r}. "
                    f"Available: {', '.join(sorted(registry))}"
                )
            _pf._ACTIVE_PREDICTOR = cls()
            _pf._METHOD_OVERRIDE = prediction_method
            return self

        def __exit__(self, exc_type, exc, tb):
            self._factory._ACTIVE_PREDICTOR = self._saved_predictor
            self._factory._METHOD_OVERRIDE = self._saved_override
            return False

    return _PredictionMethodContext()


def _run_capture(config: RunConfig) -> pd.DataFrame:
    available = get_available_dates(config.symbol)
    available = [d for d in available if config.start_date <= d <= config.end_date]
    if not available:
        raise ValueError("No available dates in requested window")

    rows: list[dict[str, Any]] = []
    previous_chain = None
    runtime_context = (
        temporary_parameter_pack("baseline_v1", overrides=config.overrides)
        if config.overrides
        else nullcontext()
    )

    with runtime_context, _activate_prediction_method(config.prediction_method):
        for trade_date in available:
            snap = replay_historical_snapshot(
                trade_date,
                config.symbol,
                compute_iv=config.compute_iv,
                include_global_market=config.include_global_market,
                include_macro_events=config.include_macro_events,
            )
            if not snap.get("ok"):
                continue
            if snap.get("quality_score", 0) < config.min_quality_score:
                continue

            option_chain = snap["option_chain"]
            spot_snapshot = snap["spot_snapshot"]
            expiries = ordered_expiries(option_chain)
            if not expiries:
                continue

            for expiry in expiries[: config.max_expiries]:
                expiry_chain = filter_option_chain_by_expiry(option_chain, expiry)
                if expiry_chain is None or expiry_chain.empty:
                    continue

                signal_result = run_preloaded_engine_snapshot(
                    symbol=config.symbol,
                    mode="BACKTEST",
                    source="HISTORICAL_HOLISTIC",
                    spot_snapshot=spot_snapshot,
                    option_chain=expiry_chain,
                    previous_chain=previous_chain,
                    apply_budget_constraint=BACKTEST_ENABLE_BUDGET,
                    requested_lots=NUMBER_OF_LOTS,
                    lot_size=LOT_SIZE,
                    max_capital=MAX_CAPITAL_PER_TRADE,
                    capture_signal_evaluation=False,
                    enable_shadow_logging=False,
                    global_market_snapshot=snap["global_market_snapshot"],
                    macro_event_state=snap["macro_event_state"],
                    target_profit_percent=config.target_profit_percent,
                    stop_loss_percent=config.stop_loss_percent,
                )
                if not signal_result.get("ok"):
                    continue

                trade = signal_result.get("trade")
                if not trade:
                    continue

                row = build_signal_evaluation_row(
                    signal_result,
                    notes=f"recent_policy_impact|label={config.label}|expiry={expiry}",
                    captured_at=f"{trade_date}T15:30:00+05:30",
                )
                for field in EXTRA_FIELDS:
                    row[field] = _coerce_serializable(trade.get(field))
                row["comparison_label"] = config.label
                row["prediction_method"] = config.prediction_method

                if config.evaluate_outcomes:
                    expiry_date = None
                    try:
                        expiry_date = pd.to_datetime(expiry).date()
                    except Exception:
                        expiry_date = None
                    spot_path = build_realized_spot_path(trade_date, expiry_date)
                    if not spot_path.empty:
                        row = evaluate_eod_outcomes(row, spot_path, available)

                rows.append(row)

            previous_chain = option_chain.copy()

    return _normalize_frame(pd.DataFrame(rows))


def _direction_balance(frame: pd.DataFrame) -> dict[str, Any]:
    trade_frame = frame[frame["trade_status"] == "TRADE"].copy()
    calls = int((trade_frame["direction"] == "CALL").sum())
    puts = int((trade_frame["direction"] == "PUT").sum())
    return {
        "trade_signals": int(len(trade_frame)),
        "calls": calls,
        "puts": puts,
        "call_put_ratio": round(calls / max(puts, 1), 3),
    }


def _series_distribution(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0}
    return {
        "count": int(numeric.count()),
        "mean": round(float(numeric.mean()), 3),
        "median": round(float(numeric.median()), 3),
        "p25": round(float(numeric.quantile(0.25)), 3),
        "p75": round(float(numeric.quantile(0.75)), 3),
        "p90": round(float(numeric.quantile(0.90)), 3),
    }


def _hit_rates_by_regime(frame: pd.DataFrame) -> list[dict[str, Any]]:
    trade_frame = frame[frame["trade_status"] == "TRADE"].copy()
    if trade_frame.empty:
        return []
    trade_frame["correct_60m"] = pd.to_numeric(trade_frame.get("correct_60m"), errors="coerce")
    trade_frame["correct_120m"] = pd.to_numeric(trade_frame.get("correct_120m"), errors="coerce")

    rows = []
    for regime, group in trade_frame.groupby("signal_regime", dropna=False):
        rows.append(
            {
                "signal_regime": regime,
                "signal_count": int(len(group)),
                "hit_rate_60m": round(float(group["correct_60m"].mean()), 4) if group["correct_60m"].notna().any() else None,
                "hit_rate_120m": round(float(group["correct_120m"].mean()), 4) if group["correct_120m"].notna().any() else None,
            }
        )
    rows.sort(key=lambda row: row["signal_count"], reverse=True)
    return rows


def _summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    status_counts = frame["trade_status"].value_counts(dropna=False).to_dict() if "trade_status" in frame else {}
    trade_frame = frame[frame["trade_status"] == "TRADE"].copy()
    return {
        "signal_count": int(len(frame)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "direction_balance": _direction_balance(frame),
        "runtime_composite_distribution": _series_distribution(trade_frame.get("runtime_composite_score", pd.Series(dtype=float))),
        "recommended_hold_distribution": _series_distribution(trade_frame.get("recommended_hold_minutes", pd.Series(dtype=float))),
        "max_hold_distribution": _series_distribution(trade_frame.get("max_hold_minutes", pd.Series(dtype=float))),
        "hit_rates_by_regime": _hit_rates_by_regime(frame),
    }


def _build_transition_frame(baseline: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    baseline_cols = [
        "signal_id",
        "trade_status",
        "direction",
        "signal_regime",
        "runtime_composite_score",
        "recommended_hold_minutes",
        "max_hold_minutes",
        "provider_health_summary",
        "call_put_alignment",
    ]
    current_cols = baseline_cols
    baseline_frame = baseline[baseline_cols].rename(columns={col: f"baseline_{col}" for col in baseline_cols if col != "signal_id"})
    current_frame = current[current_cols].rename(columns={col: f"current_{col}" for col in current_cols if col != "signal_id"})
    merged = baseline_frame.merge(current_frame, on="signal_id", how="outer")
    merged["status_transition"] = (
        merged["baseline_trade_status"].fillna("MISSING")
        + " -> "
        + merged["current_trade_status"].fillna("MISSING")
    )
    return merged.sort_values(["status_transition", "signal_id"]).reset_index(drop=True)


def _build_comparison_summary(baseline: pd.DataFrame, current: pd.DataFrame, transitions: pd.DataFrame) -> dict[str, Any]:
    baseline_trade_count = int((baseline["trade_status"] == "TRADE").sum())
    rerouted_trade_to_watchlist = int(
        ((transitions["baseline_trade_status"] == "TRADE") & (transitions["current_trade_status"] == "WATCHLIST")).sum()
    )
    blocked_trade_to_nontrade = int(
        ((transitions["baseline_trade_status"] == "TRADE") & (transitions["current_trade_status"] != "TRADE")).sum()
    )
    return {
        "baseline": _summarize_frame(baseline),
        "current": _summarize_frame(current),
        "status_transition_counts": {
            str(key): int(value)
            for key, value in transitions["status_transition"].value_counts(dropna=False).to_dict().items()
        },
        "trade_to_watchlist_reroute_rate": round(rerouted_trade_to_watchlist / max(baseline_trade_count, 1), 4),
        "trade_to_nontrade_block_rate": round(blocked_trade_to_nontrade / max(baseline_trade_count, 1), 4),
        "rerouted_trade_to_watchlist_count": rerouted_trade_to_watchlist,
        "blocked_trade_to_nontrade_count": blocked_trade_to_nontrade,
    }


def _render_markdown(configs: list[RunConfig], summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    current = summary["current"]
    baseline_balance = baseline["direction_balance"]
    current_balance = current["direction_balance"]
    lines = [
        "# Recent Policy Impact Analysis",
        "",
        f"Window: {configs[0].start_date} to {configs[0].end_date}",
        f"Prediction method: {configs[0].prediction_method}",
        f"Max expiries/day: {configs[0].max_expiries}",
        "",
        "## Topline",
        "",
        "| Metric | Baseline-like | Current |",
        "| --- | ---: | ---: |",
        f"| Signal count | {baseline['signal_count']} | {current['signal_count']} |",
        f"| Trade signals | {baseline_balance['trade_signals']} | {current_balance['trade_signals']} |",
        f"| Call/put ratio (TRADE only) | {baseline_balance['call_put_ratio']} | {current_balance['call_put_ratio']} |",
        f"| TRADE -> WATCHLIST reroute rate | - | {summary['trade_to_watchlist_reroute_rate']:.2%} |",
        f"| TRADE -> non-TRADE block rate | - | {summary['trade_to_nontrade_block_rate']:.2%} |",
        "",
        "## Runtime Composite Distribution (TRADE only)",
        "",
        f"Baseline-like: {json.dumps(baseline['runtime_composite_distribution'], indent=2)}",
        "",
        f"Current: {json.dumps(current['runtime_composite_distribution'], indent=2)}",
        "",
        "## Recommended Hold Distribution (TRADE only)",
        "",
        f"Baseline-like: {json.dumps(baseline['recommended_hold_distribution'], indent=2)}",
        "",
        f"Current: {json.dumps(current['recommended_hold_distribution'], indent=2)}",
        "",
        "## Hit Rates By Signal Regime",
        "",
        "### Baseline-like",
        "",
    ]
    for row in baseline["hit_rates_by_regime"]:
        lines.append(
            f"- {row['signal_regime']}: n={row['signal_count']}, 60m={row['hit_rate_60m']}, 120m={row['hit_rate_120m']}"
        )
    lines.extend(["", "### Current", ""])
    for row in current["hit_rates_by_regime"]:
        lines.append(
            f"- {row['signal_regime']}: n={row['signal_count']}, 60m={row['hit_rate_60m']}, 120m={row['hit_rate_120m']}"
        )
    lines.extend(["", "## Status Transition Counts", ""])
    for key, value in summary["status_transition_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline-like and current runtime policy behavior.")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--prediction-method", default="blended")
    parser.add_argument("--trading-days", type=int, default=20)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--max-expiries", type=int, default=3)
    parser.add_argument("--report-tag", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    symbol = args.symbol
    trading_days = args.trading_days
    prediction_method = args.prediction_method
    available = get_available_dates(symbol)

    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("start_date must be on or before end_date")

    if args.start_date or args.end_date:
        start_date = args.start_date or min(available)
        end_date = args.end_date or max(available)
        window_dates = [trade_date for trade_date in available if start_date <= trade_date <= end_date]
        if not window_dates:
            raise ValueError("No available dates in requested date range")
        start_date = window_dates[0]
        end_date = window_dates[-1]
    else:
        if len(available) < trading_days:
            raise ValueError(f"Need at least {trading_days} dates for analysis")
        start_date = available[-trading_days]
        end_date = available[-1]

    report_name = args.report_tag or f"policy_impact_recent_{end_date.strftime('%Y%m%d')}"
    report_root = Path("research/signal_evaluation/reports") / report_name
    report_root.mkdir(parents=True, exist_ok=True)

    baseline_cfg = RunConfig(
        label="baseline_like",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        prediction_method=prediction_method,
        max_expiries=args.max_expiries,
        overrides=BASELINE_LIKE_OVERRIDES,
    )
    current_cfg = RunConfig(
        label="current",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        prediction_method=prediction_method,
        max_expiries=args.max_expiries,
        overrides=None,
    )

    baseline = _run_capture(baseline_cfg)
    current = _run_capture(current_cfg)
    transitions = _build_transition_frame(baseline, current)
    summary = _build_comparison_summary(baseline, current, transitions)
    summary["configs"] = [asdict(baseline_cfg), asdict(current_cfg)]

    baseline_path = report_root / "baseline_like_signals.csv"
    current_path = report_root / "current_signals.csv"
    transitions_path = report_root / "status_transitions.csv"
    summary_path = report_root / "summary.json"
    report_path = report_root / "report.md"

    _normalize_frame(baseline).to_csv(baseline_path, index=False)
    _normalize_frame(current).to_csv(current_path, index=False)
    _normalize_frame(transitions).to_csv(transitions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    report_path.write_text(_render_markdown([baseline_cfg, current_cfg], summary))

    print(json.dumps({
        "report_dir": str(report_root),
        "baseline_signals": len(baseline),
        "current_signals": len(current),
        "trade_to_watchlist_reroute_rate": summary["trade_to_watchlist_reroute_rate"],
        "trade_to_nontrade_block_rate": summary["trade_to_nontrade_block_rate"],
        "baseline_call_put_ratio": summary["baseline"]["direction_balance"]["call_put_ratio"],
        "current_call_put_ratio": summary["current"]["direction_balance"]["call_put_ratio"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())