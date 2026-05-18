#!/usr/bin/env python3
"""Mine historical NIFTY spot, options, global, and macro data for edge clues.

The output is a daily feature panel plus ranked conditional tables.  The goal is
to understand the data before wiring anything into live engine rules.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.market_data_policy import IST_TIMEZONE  # noqa: E402


SPOT_PATH = PROJECT_ROOT / "data_store" / "historical" / "spot" / "NIFTY_spot_daily.parquet"
OPTIONS_PATH = PROJECT_ROOT / "data_store" / "historical" / "merged" / "NIFTY_option_chain_historical.parquet"
GLOBAL_PATH = PROJECT_ROOT / "data_store" / "historical" / "global_market" / "features" / "global_market_features.parquet"
MACRO_PATH = PROJECT_ROOT / "data_store" / "historical" / "macro_events" / "india_macro_events_historical.json"
OUT_DIR = PROJECT_ROOT / "research" / "ml_research" / "historical_insights"


def _now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST_TIMEZONE)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return round(val, digits)


def _safe_div(num: Any, den: Any) -> float | None:
    try:
        n = float(num)
        d = float(den)
    except Exception:
        return None
    if not math.isfinite(n) or not math.isfinite(d) or d == 0:
        return None
    return n / d


def _norm_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _pct_change_bps(series: pd.Series, periods: int = 1) -> pd.Series:
    return (series / series.shift(periods) - 1.0) * 10000.0


def _forward_return_bps(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) / close - 1.0) * 10000.0


def _ann_realized_vol(ret_bps: pd.Series, window: int) -> pd.Series:
    return (ret_bps / 10000.0).rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252.0)


def _summary_stats(series: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return {"n": 0}
    quantiles = vals.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(len(vals)),
        "mean": _round(vals.mean(), 4),
        "median": _round(vals.median(), 4),
        "std": _round(vals.std(), 4),
        "skew": _round(vals.skew(), 4),
        "kurtosis": _round(vals.kurtosis(), 4),
        "p01": _round(quantiles.loc[0.01], 4),
        "p05": _round(quantiles.loc[0.05], 4),
        "p25": _round(quantiles.loc[0.25], 4),
        "p75": _round(quantiles.loc[0.75], 4),
        "p95": _round(quantiles.loc[0.95], 4),
        "p99": _round(quantiles.loc[0.99], 4),
    }


def _performance_summary(frame: pd.DataFrame, return_col: str = "fwd_ret_1d_bps") -> dict[str, Any]:
    vals = pd.to_numeric(frame.get(return_col), errors="coerce").dropna()
    if vals.empty:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "mean_bps": _round(vals.mean(), 2),
        "median_bps": _round(vals.median(), 2),
        "hit_rate_up": _round((vals > 0).mean(), 4),
        "abs_mean_bps": _round(vals.abs().mean(), 2),
        "p05_bps": _round(vals.quantile(0.05), 2),
        "p95_bps": _round(vals.quantile(0.95), 2),
    }


def _bucket_performance(
    frame: pd.DataFrame,
    feature: str,
    *,
    target_cols: list[str],
    buckets: int = 5,
    min_count: int = 80,
) -> pd.DataFrame:
    if feature not in frame.columns:
        return pd.DataFrame()
    values = pd.to_numeric(frame[feature], errors="coerce")
    valid = frame.loc[values.notna()].copy()
    if valid.empty or values.nunique(dropna=True) < 3:
        return pd.DataFrame()
    try:
        valid["bucket"] = pd.qcut(values.loc[valid.index], buckets, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bucket, sub in valid.groupby("bucket", observed=False):
        item: dict[str, Any] = {
            "feature": feature,
            "bucket": str(bucket),
            "n": int(len(sub)),
            "feature_min": _round(pd.to_numeric(sub[feature], errors="coerce").min(), 4),
            "feature_max": _round(pd.to_numeric(sub[feature], errors="coerce").max(), 4),
        }
        for target in target_cols:
            vals = pd.to_numeric(sub.get(target), errors="coerce").dropna()
            item[f"{target}_n"] = int(len(vals))
            item[f"{target}_mean"] = _round(vals.mean(), 3) if len(vals) >= min_count else None
            item[f"{target}_median"] = _round(vals.median(), 3) if len(vals) >= min_count else None
            item[f"{target}_hit_up"] = _round((vals > 0).mean(), 4) if len(vals) >= min_count else None
            item[f"{target}_abs_mean"] = _round(vals.abs().mean(), 3) if len(vals) >= min_count else None
        rows.append(item)
    return pd.DataFrame(rows)


def _categorical_performance(
    frame: pd.DataFrame,
    feature: str,
    *,
    target_cols: list[str],
    min_count: int = 50,
) -> pd.DataFrame:
    if feature not in frame.columns:
        return pd.DataFrame()
    rows = []
    for value, sub in frame.groupby(feature, dropna=False):
        item: dict[str, Any] = {"feature": feature, "bucket": str(value), "n": int(len(sub))}
        if len(sub) < min_count:
            continue
        for target in target_cols:
            vals = pd.to_numeric(sub.get(target), errors="coerce").dropna()
            item[f"{target}_n"] = int(len(vals))
            item[f"{target}_mean"] = _round(vals.mean(), 3) if len(vals) >= min_count else None
            item[f"{target}_median"] = _round(vals.median(), 3) if len(vals) >= min_count else None
            item[f"{target}_hit_up"] = _round((vals > 0).mean(), 4) if len(vals) >= min_count else None
            item[f"{target}_abs_mean"] = _round(vals.abs().mean(), 3) if len(vals) >= min_count else None
        rows.append(item)
    return pd.DataFrame(rows)


def _spearman_table(frame: pd.DataFrame, features: list[str], targets: list[str], min_count: int = 250) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in frame.columns:
            continue
        x = pd.to_numeric(frame[feature], errors="coerce")
        for target in targets:
            if target not in frame.columns:
                continue
            y = pd.to_numeric(frame[target], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < min_count or x.loc[mask].nunique() < 3 or y.loc[mask].nunique() < 3:
                continue
            corr = x.loc[mask].corr(y.loc[mask], method="spearman")
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": n,
                    "spearman": _round(corr, 4),
                    "abs_spearman": _round(abs(corr), 4),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_spearman", "n"], ascending=[False, False]).reset_index(drop=True)
    return out


def build_spot_panel() -> pd.DataFrame:
    spot = pd.read_parquet(SPOT_PATH).copy()
    spot["date"] = _norm_date(spot["date"])
    spot = spot.sort_values("date").drop_duplicates("date", keep="last")
    for col in ["open", "high", "low", "close"]:
        spot[col] = pd.to_numeric(spot[col], errors="coerce")

    close = spot["close"]
    prev_close = close.shift(1)
    spot["ret_1d_bps"] = _pct_change_bps(close)
    spot["gap_bps"] = (spot["open"] / prev_close - 1.0) * 10000.0
    spot["intraday_bps"] = (spot["close"] / spot["open"] - 1.0) * 10000.0
    spot["range_bps"] = (spot["high"] / spot["low"] - 1.0) * 10000.0
    spot["close_to_high_bps"] = (spot["high"] / spot["close"] - 1.0) * 10000.0
    spot["close_to_low_bps"] = (spot["close"] / spot["low"] - 1.0) * 10000.0
    spot["ret_5d_bps"] = _pct_change_bps(close, 5)
    spot["ret_20d_bps"] = _pct_change_bps(close, 20)
    spot["realized_vol_5d"] = _ann_realized_vol(spot["ret_1d_bps"], 5)
    spot["realized_vol_20d"] = _ann_realized_vol(spot["ret_1d_bps"], 20)
    spot["realized_vol_60d"] = _ann_realized_vol(spot["ret_1d_bps"], 60)
    spot["fwd_ret_1d_bps"] = _forward_return_bps(close, 1)
    spot["fwd_ret_3d_bps"] = _forward_return_bps(close, 3)
    spot["fwd_ret_5d_bps"] = _forward_return_bps(close, 5)
    spot["fwd_abs_ret_1d_bps"] = spot["fwd_ret_1d_bps"].abs()
    spot["fwd_abs_ret_3d_bps"] = spot["fwd_ret_3d_bps"].abs()
    spot["fwd_abs_ret_5d_bps"] = spot["fwd_ret_5d_bps"].abs()
    spot["next_day_range_bps"] = spot["range_bps"].shift(-1)
    spot["weekday"] = spot["date"].dt.day_name()
    spot["month"] = spot["date"].dt.month_name()
    spot["year"] = spot["date"].dt.year
    spot["is_expiry_weekday_thursday"] = spot["date"].dt.weekday.eq(3)
    return spot


def build_global_panel() -> pd.DataFrame:
    global_df = pd.read_parquet(GLOBAL_PATH).copy()
    global_df["date"] = _norm_date(global_df["date"])
    global_df = global_df.sort_values("date").drop_duplicates("date", keep="last")
    return global_df


def _severity_rank(value: str | None) -> int:
    mapping = {"MINOR": 1, "MEDIUM": 2, "MAJOR": 3, "HIGH": 3, "CRITICAL": 4}
    return mapping.get(str(value or "").upper(), 0)


def build_macro_panel() -> pd.DataFrame:
    if not MACRO_PATH.exists():
        return pd.DataFrame(columns=["date", "macro_event_count", "macro_max_severity_rank"])
    payload = json.loads(MACRO_PATH.read_text(encoding="utf-8"))
    events = payload.get("events") if isinstance(payload, dict) else payload
    rows = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        ts = pd.to_datetime(event.get("timestamp") or event.get("date"), errors="coerce")
        if pd.isna(ts):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST_TIMEZONE)
        else:
            ts = ts.tz_convert(IST_TIMEZONE)
        rows.append(
            {
                "date": ts.normalize().tz_localize(None),
                "macro_event_name": event.get("name"),
                "macro_event_source": event.get("source"),
                "macro_event_severity": str(event.get("severity") or "UNKNOWN").upper(),
                "macro_severity_rank": _severity_rank(event.get("severity")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "macro_event_count", "macro_max_severity_rank"])
    events_df = pd.DataFrame(rows)
    grouped = events_df.groupby("date").agg(
        macro_event_count=("macro_event_name", "count"),
        macro_max_severity_rank=("macro_severity_rank", "max"),
        macro_major_event=("macro_severity_rank", lambda s: int((s >= 3).any())),
        macro_sources=("macro_event_source", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))[:200]),
        macro_severities=("macro_event_severity", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
    )
    return grouped.reset_index()


def _max_pain(strikes: np.ndarray, call_oi: np.ndarray, put_oi: np.ndarray) -> float | None:
    if len(strikes) == 0:
        return None
    strikes = strikes.astype(float)
    call_oi = np.nan_to_num(call_oi.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    put_oi = np.nan_to_num(put_oi.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    settle = strikes.reshape(-1, 1)
    strike_row = strikes.reshape(1, -1)
    call_pain = np.maximum(settle - strike_row, 0.0) * call_oi.reshape(1, -1)
    put_pain = np.maximum(strike_row - settle, 0.0) * put_oi.reshape(1, -1)
    total_pain = call_pain.sum(axis=1) + put_pain.sum(axis=1)
    if total_pain.size == 0:
        return None
    return float(strikes[int(np.nanargmin(total_pain))])


def build_option_features(spot_panel: pd.DataFrame, *, max_days: int | None = None) -> pd.DataFrame:
    columns = [
        "trade_date",
        "expiry_date",
        "instrument",
        "strike_price",
        "option_type",
        "close",
        "contracts",
        "open_interest",
        "change_in_oi",
    ]
    options = pd.read_parquet(OPTIONS_PATH, columns=columns)
    # NSE instrument labels changed in the normalized history
    # (OPTIDX historically, IDO in newer files).  The option_type field is the
    # stable contract discriminator, so use it as the primary filter.
    options = options[options["option_type"].astype(str).str.upper().isin({"CE", "PE"})].copy()
    options["date"] = _norm_date(options["trade_date"])
    options["expiry_date"] = _norm_date(options["expiry_date"])
    options["dte"] = (options["expiry_date"] - options["date"]).dt.days
    options = options[options["dte"].ge(0)]
    if max_days is not None:
        recent_dates = sorted(options["date"].dropna().unique())[-max_days:]
        options = options[options["date"].isin(recent_dates)]

    front_expiry = options.groupby("date")["dte"].min().rename("front_dte").reset_index()
    options = options.merge(front_expiry, on="date", how="inner")
    front = options[options["dte"].eq(options["front_dte"])].copy()
    for col in ["strike_price", "close", "contracts", "open_interest", "change_in_oi"]:
        front[col] = pd.to_numeric(front[col], errors="coerce")

    spot_lookup = spot_panel[["date", "close"]].rename(columns={"close": "spot_close"})
    front = front.merge(spot_lookup, on="date", how="left")
    front = front.dropna(subset=["spot_close", "strike_price"])

    rows: list[dict[str, Any]] = []
    for date_value, group in front.groupby("date", sort=True):
        spot = float(group["spot_close"].iloc[0])
        if not math.isfinite(spot) or spot <= 0:
            continue
        ce = group[group["option_type"].astype(str).str.upper().eq("CE")]
        pe = group[group["option_type"].astype(str).str.upper().eq("PE")]
        total_call_oi = ce["open_interest"].sum(min_count=1)
        total_put_oi = pe["open_interest"].sum(min_count=1)
        total_call_vol = ce["contracts"].sum(min_count=1)
        total_put_vol = pe["contracts"].sum(min_count=1)
        total_call_chg = ce["change_in_oi"].sum(min_count=1)
        total_put_chg = pe["change_in_oi"].sum(min_count=1)

        strikes = np.array(sorted(group["strike_price"].dropna().unique()), dtype=float)
        by_strike = group.pivot_table(
            index="strike_price",
            columns="option_type",
            values=["open_interest", "contracts", "close", "change_in_oi"],
            aggfunc="sum",
        )
        by_strike.columns = [f"{str(a).lower()}_{str(b).upper()}" for a, b in by_strike.columns]
        by_strike = by_strike.reset_index().sort_values("strike_price")
        call_oi_arr = by_strike.get("open_interest_CE", pd.Series(0.0, index=by_strike.index)).to_numpy(dtype=float)
        put_oi_arr = by_strike.get("open_interest_PE", pd.Series(0.0, index=by_strike.index)).to_numpy(dtype=float)
        strike_arr = by_strike["strike_price"].to_numpy(dtype=float)
        max_pain = _max_pain(strike_arr, call_oi_arr, put_oi_arr)

        atm_idx = int(np.argmin(np.abs(strike_arr - spot))) if len(strike_arr) else None
        atm_strike = float(strike_arr[atm_idx]) if atm_idx is not None else None
        atm_row = by_strike.iloc[atm_idx] if atm_idx is not None else None
        atm_ce = float(atm_row.get("close_CE", np.nan)) if atm_row is not None else np.nan
        atm_pe = float(atm_row.get("close_PE", np.nan)) if atm_row is not None else np.nan
        atm_straddle = atm_ce + atm_pe if math.isfinite(atm_ce) and math.isfinite(atm_pe) else np.nan

        call_above = ce[ce["strike_price"].ge(spot)]
        put_below = pe[pe["strike_price"].le(spot)]
        call_wall = None
        put_wall = None
        if not call_above.empty and call_above["open_interest"].notna().any():
            call_wall = float(call_above.loc[call_above["open_interest"].idxmax(), "strike_price"])
        if not put_below.empty and put_below["open_interest"].notna().any():
            put_wall = float(put_below.loc[put_below["open_interest"].idxmax(), "strike_price"])

        all_oi = group["open_interest"].dropna().sort_values(ascending=False)
        top5_conc = float(all_oi.head(5).sum() / all_oi.sum()) if all_oi.sum() else np.nan
        near_atm = group[group["strike_price"].sub(spot).abs().le(max(100.0, spot * 0.01))]
        near_ce_oi = near_atm[near_atm["option_type"].astype(str).str.upper().eq("CE")]["open_interest"].sum(min_count=1)
        near_pe_oi = near_atm[near_atm["option_type"].astype(str).str.upper().eq("PE")]["open_interest"].sum(min_count=1)
        near_ce_vol = near_atm[near_atm["option_type"].astype(str).str.upper().eq("CE")]["contracts"].sum(min_count=1)
        near_pe_vol = near_atm[near_atm["option_type"].astype(str).str.upper().eq("PE")]["contracts"].sum(min_count=1)

        rows.append(
            {
                "date": pd.Timestamp(date_value),
                "front_dte": int(group["dte"].min()),
                "front_expiry": group["expiry_date"].iloc[0],
                "option_rows_front": int(len(group)),
                "strike_count_front": int(len(strikes)),
                "pcr_oi": _safe_div(total_put_oi, total_call_oi),
                "pcr_volume": _safe_div(total_put_vol, total_call_vol),
                "pcr_chg_oi": _safe_div(total_put_chg, total_call_chg),
                "total_call_oi": _round(total_call_oi, 2),
                "total_put_oi": _round(total_put_oi, 2),
                "total_call_volume": _round(total_call_vol, 2),
                "total_put_volume": _round(total_put_vol, 2),
                "net_put_minus_call_oi": _round(total_put_oi - total_call_oi, 2),
                "net_put_minus_call_chg_oi": _round(total_put_chg - total_call_chg, 2),
                "oi_top5_concentration": _round(top5_conc, 6),
                "near_atm_pcr_oi": _safe_div(near_pe_oi, near_ce_oi),
                "near_atm_pcr_volume": _safe_div(near_pe_vol, near_ce_vol),
                "atm_strike": atm_strike,
                "atm_straddle_close": _round(atm_straddle, 4),
                "atm_straddle_pct": _round(atm_straddle / spot * 100.0, 4) if math.isfinite(atm_straddle) else None,
                "max_pain": max_pain,
                "max_pain_dist_pct": _round((max_pain / spot - 1.0) * 100.0, 4) if max_pain else None,
                "max_pain_abs_dist_pct": _round(abs(max_pain / spot - 1.0) * 100.0, 4) if max_pain else None,
                "call_wall_above": call_wall,
                "call_wall_dist_pct": _round((call_wall / spot - 1.0) * 100.0, 4) if call_wall else None,
                "put_wall_below": put_wall,
                "put_wall_dist_pct": _round((put_wall / spot - 1.0) * 100.0, 4) if put_wall else None,
                "wall_width_pct": _round((call_wall - put_wall) / spot * 100.0, 4) if call_wall and put_wall else None,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def add_derived_feature_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["gap_fade_bps"] = -np.sign(out["gap_bps"]) * out["intraday_bps"]
    out["same_day_gap_follow_through"] = (np.sign(out["gap_bps"]) * out["intraday_bps"] > 0).astype(float)
    out["prev_day_follow_through_bps"] = np.sign(out["ret_1d_bps"]) * out["fwd_ret_1d_bps"]
    out["max_pain_pull_1d_bps"] = np.sign(out["max_pain_dist_pct"]) * out["fwd_ret_1d_bps"]
    out["max_pain_pull_5d_bps"] = np.sign(out["max_pain_dist_pct"]) * out["fwd_ret_5d_bps"]
    out["near_call_wall"] = pd.to_numeric(out.get("call_wall_dist_pct"), errors="coerce").between(0.0, 0.5)
    out["near_put_wall"] = pd.to_numeric(out.get("put_wall_dist_pct"), errors="coerce").between(-0.5, 0.0)
    out["expiry_bucket"] = pd.cut(
        pd.to_numeric(out.get("front_dte"), errors="coerce"),
        bins=[-1, 1, 3, 7, 14, 45],
        labels=["0-1d", "2-3d", "4-7d", "8-14d", "15d+"],
    ).astype("object")
    out["india_vix_bucket"] = _qcut_label(out, "india_vix_level", labels=["low", "q2", "q3", "q4", "high"])
    out["pcr_oi_bucket"] = _qcut_label(out, "pcr_oi", labels=["low", "q2", "q3", "q4", "high"])
    out["trend_20d_bucket"] = pd.cut(
        pd.to_numeric(out.get("ret_20d_bps"), errors="coerce"),
        bins=[-np.inf, -500, -150, 150, 500, np.inf],
        labels=["selloff", "weak", "flat", "strong", "surge"],
    ).astype("object")
    return out


def _qcut_label(frame: pd.DataFrame, column: str, labels: list[str]) -> pd.Series:
    values = pd.to_numeric(frame.get(column), errors="coerce")
    try:
        return pd.qcut(values, len(labels), labels=labels, duplicates="drop").astype("object")
    except ValueError:
        return pd.Series(pd.NA, index=frame.index, dtype="object")


def build_daily_feature_panel(*, max_option_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot = build_spot_panel()
    global_df = build_global_panel()
    macro = build_macro_panel()
    option_features = build_option_features(spot, max_days=max_option_days)

    panel = spot.merge(global_df, on="date", how="left", suffixes=("", "_global"))
    panel = panel.merge(macro, on="date", how="left")
    panel = panel.merge(option_features, on="date", how="left")
    panel["macro_event_count"] = panel["macro_event_count"].fillna(0).astype(int)
    panel["macro_max_severity_rank"] = panel["macro_max_severity_rank"].fillna(0).astype(int)
    panel["macro_major_event"] = panel["macro_major_event"].fillna(0).astype(int)
    panel = add_derived_feature_flags(panel)
    return panel.sort_values("date").reset_index(drop=True), option_features


def _top_decile_study(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = frame.loc[values.notna()].copy()
        if len(valid) < 250 or values.nunique(dropna=True) < 10:
            continue
        low = values.quantile(0.1)
        high = values.quantile(0.9)
        for label, mask in [("bottom_decile", values <= low), ("top_decile", values >= high)]:
            sub = frame.loc[mask & values.notna()]
            if len(sub) < 50:
                continue
            item = {"feature": feature, "bucket": label, "threshold": _round(low if label == "bottom_decile" else high, 4)}
            item.update({f"fwd1_{k}": v for k, v in _performance_summary(sub, "fwd_ret_1d_bps").items()})
            item.update({f"fwd5_{k}": v for k, v in _performance_summary(sub, "fwd_ret_5d_bps").items()})
            item["next_range_mean_bps"] = _round(pd.to_numeric(sub["next_day_range_bps"], errors="coerce").mean(), 2)
            rows.append(item)
    return pd.DataFrame(rows)


def _event_studies(panel: pd.DataFrame) -> dict[str, Any]:
    studies: dict[str, Any] = {}
    macro_day = panel["macro_event_count"].fillna(0).gt(0)
    major_day = panel["macro_major_event"].fillna(0).gt(0)
    studies["macro_event_day"] = {
        "event_day": _performance_summary(panel.loc[macro_day], "ret_1d_bps"),
        "non_event_day": _performance_summary(panel.loc[~macro_day], "ret_1d_bps"),
        "next_day_after_event": _performance_summary(panel.loc[macro_day], "fwd_ret_1d_bps"),
        "major_event_day": _performance_summary(panel.loc[major_day], "ret_1d_bps"),
    }
    near_call = panel["near_call_wall"].fillna(False).astype(bool)
    near_put = panel["near_put_wall"].fillna(False).astype(bool)
    studies["wall_proximity"] = {
        "near_call_wall_next": _performance_summary(panel.loc[near_call], "fwd_ret_1d_bps"),
        "near_put_wall_next": _performance_summary(panel.loc[near_put], "fwd_ret_1d_bps"),
    }
    mp = panel[pd.to_numeric(panel["max_pain_dist_pct"], errors="coerce").notna()].copy()
    studies["max_pain_pull"] = {
        "all_n": int(len(mp)),
        "pull_1d_mean_bps": _round(pd.to_numeric(mp["max_pain_pull_1d_bps"], errors="coerce").mean(), 2),
        "pull_1d_hit_rate": _round((pd.to_numeric(mp["max_pain_pull_1d_bps"], errors="coerce") > 0).mean(), 4),
        "pull_5d_mean_bps": _round(pd.to_numeric(mp["max_pain_pull_5d_bps"], errors="coerce").mean(), 2),
        "pull_5d_hit_rate": _round((pd.to_numeric(mp["max_pain_pull_5d_bps"], errors="coerce") > 0).mean(), 4),
    }
    return studies


def _interaction_table(panel: pd.DataFrame, row_feature: str, col_feature: str, target: str) -> pd.DataFrame:
    if row_feature not in panel.columns or col_feature not in panel.columns:
        return pd.DataFrame()
    rows = []
    for row_value, row_sub in panel.groupby(row_feature, dropna=False):
        for col_value, sub in row_sub.groupby(col_feature, dropna=False):
            vals = pd.to_numeric(sub[target], errors="coerce").dropna()
            if len(vals) < 40:
                continue
            rows.append(
                {
                    "row_feature": row_feature,
                    "row_value": str(row_value),
                    "col_feature": col_feature,
                    "col_value": str(col_value),
                    "target": target,
                    "n": int(len(vals)),
                    "mean_bps": _round(vals.mean(), 2),
                    "hit_up": _round((vals > 0).mean(), 4),
                    "abs_mean_bps": _round(vals.abs().mean(), 2),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_bps", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def run_insight_mining(*, max_option_days: int | None = None) -> dict[str, Any]:
    run_id = _now_ist().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    panel, option_features = build_daily_feature_panel(max_option_days=max_option_days)
    panel_path = run_dir / "historical_daily_feature_panel.csv"
    option_path = run_dir / "historical_option_daily_features.csv"
    _atomic_write_csv(panel_path, panel)
    _atomic_write_csv(option_path, option_features)

    target_cols = ["fwd_ret_1d_bps", "fwd_ret_5d_bps", "fwd_abs_ret_1d_bps", "fwd_abs_ret_5d_bps", "next_day_range_bps"]
    candidate_features = [
        "ret_1d_bps",
        "ret_5d_bps",
        "ret_20d_bps",
        "gap_bps",
        "intraday_bps",
        "range_bps",
        "realized_vol_5d",
        "realized_vol_20d",
        "realized_vol_60d",
        "oil_change_24h",
        "gold_change_24h",
        "copper_change_24h",
        "vix_change_24h",
        "india_vix_change_24h",
        "india_vix_level",
        "sp500_change_24h",
        "nasdaq_change_24h",
        "us10y_change_bp",
        "usdinr_change_24h",
        "nifty50_realized_vol_5d",
        "nifty50_realized_vol_30d",
        "banknifty_realized_vol_5d",
        "banknifty_realized_vol_30d",
        "front_dte",
        "pcr_oi",
        "pcr_volume",
        "pcr_chg_oi",
        "near_atm_pcr_oi",
        "near_atm_pcr_volume",
        "oi_top5_concentration",
        "atm_straddle_pct",
        "max_pain_dist_pct",
        "max_pain_abs_dist_pct",
        "call_wall_dist_pct",
        "put_wall_dist_pct",
        "wall_width_pct",
    ]
    correlations = _spearman_table(panel, candidate_features, target_cols)
    bucket_tables = []
    for feature in candidate_features:
        table = _bucket_performance(panel, feature, target_cols=target_cols)
        if not table.empty:
            bucket_tables.append(table)
    bucket_performance = pd.concat(bucket_tables, ignore_index=True, sort=False) if bucket_tables else pd.DataFrame()

    categorical_tables = []
    for feature in [
        "weekday",
        "month",
        "expiry_bucket",
        "india_vix_bucket",
        "pcr_oi_bucket",
        "trend_20d_bucket",
        "is_expiry_weekday_thursday",
        "near_call_wall",
        "near_put_wall",
        "macro_major_event",
    ]:
        table = _categorical_performance(panel, feature, target_cols=target_cols)
        if not table.empty:
            categorical_tables.append(table)
    categorical_performance = pd.concat(categorical_tables, ignore_index=True, sort=False) if categorical_tables else pd.DataFrame()

    decile_study = _top_decile_study(panel, candidate_features)
    interactions = {
        "india_vix_x_trend": _interaction_table(panel, "india_vix_bucket", "trend_20d_bucket", "fwd_ret_1d_bps"),
        "expiry_x_pcr": _interaction_table(panel, "expiry_bucket", "pcr_oi_bucket", "fwd_ret_1d_bps"),
        "weekday_x_vix": _interaction_table(panel, "weekday", "india_vix_bucket", "next_day_range_bps"),
    }

    tables = {
        "spearman_correlations.csv": correlations,
        "numeric_bucket_performance.csv": bucket_performance,
        "categorical_performance.csv": categorical_performance,
        "top_bottom_decile_study.csv": decile_study,
    }
    for name, table in interactions.items():
        tables[f"interaction_{name}.csv"] = table
    table_paths = {}
    for filename, table in tables.items():
        path = run_dir / filename
        _atomic_write_csv(path, table)
        table_paths[filename] = _rel(path)

    date_min = panel["date"].min()
    date_max = panel["date"].max()
    option_coverage = panel["pcr_oi"].notna()
    report: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": _now_ist().isoformat(),
        "paths": {
            "run_dir": _rel(run_dir),
            "daily_feature_panel": _rel(panel_path),
            "option_daily_features": _rel(option_path),
            **table_paths,
        },
        "coverage": {
            "panel_rows": int(len(panel)),
            "date_min": date_min.date().isoformat() if pd.notna(date_min) else None,
            "date_max": date_max.date().isoformat() if pd.notna(date_max) else None,
            "option_feature_rows": int(option_coverage.sum()),
            "option_date_min": panel.loc[option_coverage, "date"].min().date().isoformat() if option_coverage.any() else None,
            "option_date_max": panel.loc[option_coverage, "date"].max().date().isoformat() if option_coverage.any() else None,
            "global_feature_non_null_rows": int(panel["india_vix_level"].notna().sum()) if "india_vix_level" in panel else 0,
            "macro_event_days": int(panel["macro_event_count"].gt(0).sum()),
        },
        "distributions": {
            "daily_return_bps": _summary_stats(panel["ret_1d_bps"]),
            "daily_abs_return_bps": _summary_stats(panel["ret_1d_bps"].abs()),
            "daily_range_bps": _summary_stats(panel["range_bps"]),
            "gap_bps": _summary_stats(panel["gap_bps"]),
            "intraday_bps": _summary_stats(panel["intraday_bps"]),
            "india_vix_level": _summary_stats(panel["india_vix_level"]) if "india_vix_level" in panel else {"n": 0},
            "pcr_oi": _summary_stats(panel["pcr_oi"]) if "pcr_oi" in panel else {"n": 0},
            "atm_straddle_pct": _summary_stats(panel["atm_straddle_pct"]) if "atm_straddle_pct" in panel else {"n": 0},
        },
        "event_studies": _event_studies(panel),
        "top_correlations": correlations.head(30).to_dict(orient="records") if not correlations.empty else [],
        "notable_deciles": decile_study.sort_values("fwd1_abs_mean_bps", ascending=False).head(30).to_dict(orient="records") if not decile_study.empty else [],
        "weekday_counts": dict(Counter(panel["weekday"].dropna().astype(str))),
    }

    report["insights"] = _derive_insights(report, correlations, bucket_performance, categorical_performance, decile_study, interactions)
    json_path = run_dir / "historical_insight_report.json"
    md_path = run_dir / "historical_insight_report.md"
    _atomic_write_json(json_path, report)
    _atomic_write_text(md_path, _render_markdown(report, correlations, bucket_performance, categorical_performance, decile_study, interactions))
    _atomic_write_json(OUT_DIR / "latest_historical_insight_report.json", report)
    _atomic_write_text(OUT_DIR / "latest_historical_insight_report.md", _render_markdown(report, correlations, bucket_performance, categorical_performance, decile_study, interactions))
    return report


def _derive_insights(
    report: dict[str, Any],
    correlations: pd.DataFrame,
    bucket_performance: pd.DataFrame,
    categorical_performance: pd.DataFrame,
    decile_study: pd.DataFrame,
    interactions: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    if not correlations.empty:
        for _, row in correlations.head(10).iterrows():
            insights.append(
                {
                    "type": "rank_correlation",
                    "feature": row["feature"],
                    "target": row["target"],
                    "n": int(row["n"]),
                    "value": _round(row["spearman"], 4),
                    "interpretation": "Candidate conditioning feature; validate with walk-forward tests before use.",
                }
            )
    max_pain = report.get("event_studies", {}).get("max_pain_pull", {})
    if max_pain.get("all_n", 0) > 250:
        insights.append(
            {
                "type": "option_structure",
                "feature": "max_pain_dist_pct",
                "value": max_pain,
                "interpretation": "Positive pull metrics mean next returns tend to move toward max pain; negative means max pain is not a magnet in this sample.",
            }
        )
    macro = report.get("event_studies", {}).get("macro_event_day", {})
    if macro:
        insights.append(
            {
                "type": "macro_event",
                "feature": "macro_event_count",
                "value": macro,
                "interpretation": "Compare event-day and non-event-day range/return behavior before event lockdown tuning.",
            }
        )
    if not decile_study.empty:
        top_abs = decile_study.sort_values("fwd1_abs_mean_bps", ascending=False).head(5)
        for _, row in top_abs.iterrows():
            insights.append(
                {
                    "type": "extreme_decile",
                    "feature": row["feature"],
                    "bucket": row["bucket"],
                    "n": int(row["fwd1_n"]),
                    "next_abs_mean_bps": _round(row["fwd1_abs_mean_bps"], 2),
                    "interpretation": "Extreme feature bucket associated with larger next-day absolute movement.",
                }
            )
    return insights


def _format_table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    subset = frame.loc[:, [c for c in columns if c in frame.columns]].head(limit).copy()
    lines = ["|" + "|".join(subset.columns) + "|", "|" + "|".join(["---"] * len(subset.columns)) + "|"]
    for _, row in subset.iterrows():
        lines.append("|" + "|".join(str(row.get(c, "")) for c in subset.columns) + "|")
    return lines


def _render_markdown(
    report: dict[str, Any],
    correlations: pd.DataFrame,
    bucket_performance: pd.DataFrame,
    categorical_performance: pd.DataFrame,
    decile_study: pd.DataFrame,
    interactions: dict[str, pd.DataFrame],
) -> str:
    coverage = report.get("coverage", {})
    dist = report.get("distributions", {})
    event = report.get("event_studies", {})
    lines = [
        "# Historical Insight Mining Report",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Panel rows: `{coverage.get('panel_rows')}` from `{coverage.get('date_min')}` to `{coverage.get('date_max')}`",
        f"- Option feature rows: `{coverage.get('option_feature_rows')}` from `{coverage.get('option_date_min')}` to `{coverage.get('option_date_max')}`",
        f"- Macro event days: `{coverage.get('macro_event_days')}`",
        "",
        "## Distribution Baseline",
        f"- Daily return bps: {dist.get('daily_return_bps')}",
        f"- Daily absolute return bps: {dist.get('daily_abs_return_bps')}",
        f"- Daily range bps: {dist.get('daily_range_bps')}",
        f"- Gap bps: {dist.get('gap_bps')}",
        f"- India VIX level: {dist.get('india_vix_level')}",
        f"- Option PCR OI: {dist.get('pcr_oi')}",
        f"- ATM straddle pct: {dist.get('atm_straddle_pct')}",
        "",
        "## Strongest Rank Associations",
    ]
    lines.extend(_format_table(correlations, ["feature", "target", "n", "spearman"], limit=20))
    lines.extend(["", "## Largest Next-Day Movement Deciles"])
    lines.extend(
        _format_table(
            decile_study.sort_values("fwd1_abs_mean_bps", ascending=False) if not decile_study.empty else decile_study,
            ["feature", "bucket", "threshold", "fwd1_n", "fwd1_abs_mean_bps", "fwd1_mean_bps", "next_range_mean_bps"],
            limit=20,
        )
    )
    lines.extend(["", "## Categorical Performance"])
    lines.extend(
        _format_table(
            categorical_performance,
            ["feature", "bucket", "n", "fwd_ret_1d_bps_mean", "fwd_ret_1d_bps_hit_up", "next_day_range_bps_mean"],
            limit=30,
        )
    )
    lines.extend(["", "## Event And Option Structure Studies"])
    lines.append(f"- Macro event study: `{event.get('macro_event_day')}`")
    lines.append(f"- Wall proximity study: `{event.get('wall_proximity')}`")
    lines.append(f"- Max pain pull study: `{event.get('max_pain_pull')}`")
    lines.extend(["", "## Engine Candidate Ideas"])
    for item in report.get("insights", [])[:20]:
        lines.append(f"- `{item.get('type')}` `{item.get('feature')}`: {item.get('value', item.get('next_abs_mean_bps'))}. {item.get('interpretation')}")
    lines.extend(["", "## Output Artifacts"])
    for key, path in report.get("paths", {}).items():
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine historical NIFTY data for conditional edge insights.")
    parser.add_argument(
        "--max-option-days",
        type=int,
        default=None,
        help="Optional development limiter for the option-chain feature build.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_insight_mining(max_option_days=args.max_option_days)
    print(f"Historical insight run: {report['run_id']}")
    print(f"Panel rows: {report['coverage']['panel_rows']}")
    print(f"Option feature rows: {report['coverage']['option_feature_rows']}")
    print(f"Report: {OUT_DIR / 'latest_historical_insight_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
