from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/Users/pramitdutta/Desktop/Quant Engines/options_quant_engine')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.historical_snapshot import get_available_dates
from scripts.backtest.recent_policy_impact_analysis import RunConfig, _run_capture

out_dir = ROOT / 'research/reviews/breakout_direction_sweep_2026-04-02_v2'
out_dir.mkdir(parents=True, exist_ok=True)

symbol = 'NIFTY'
avail = get_available_dates(symbol)
selected = avail[-140:-40] if len(avail) > 160 else avail[:-20]
start_d, end_d = selected[0], selected[-1]

variants = [
    ('baseline', {}),
    ('buffer_12', {'trade_strength.runtime_thresholds.direction_breakout_buffer_points': 12}),
    ('buffer_28', {'trade_strength.runtime_thresholds.direction_breakout_buffer_points': 28}),
    ('evidence_1p6', {'trade_strength.runtime_thresholds.direction_breakout_evidence_threshold': 1.6}),
    ('evidence_2p4', {'trade_strength.runtime_thresholds.direction_breakout_evidence_threshold': 2.4}),
    ('override_loose', {
        'trade_strength.runtime_thresholds.reversal_breakout_override_move_probability_floor': 0.56,
        'trade_strength.runtime_thresholds.reversal_breakout_override_range_pct_floor': 0.28,
    }),
    ('override_strict', {
        'trade_strength.runtime_thresholds.reversal_breakout_override_move_probability_floor': 0.68,
        'trade_strength.runtime_thresholds.reversal_breakout_override_range_pct_floor': 0.42,
    }),
]

rows = []
all_frames = []
# Intraday 30m labels are intentionally disabled on synthetic daily paths.
# Use 1-day signed move (bps) as the breakout proxy for this runner.
BREAKOUT_MOVE_BPS_1D = 50.0

for label, overrides in variants:
    cfg = RunConfig(
        label=label,
        symbol=symbol,
        start_date=start_d,
        end_date=end_d,
        max_expiries=1,
        prediction_method='blended',
        compute_iv=True,
        include_global_market=True,
        include_macro_events=True,
        min_quality_score=40.0,
        evaluate_outcomes=True,
        overrides=overrides if overrides else None,
    )
    frame = _run_capture(cfg)
    if frame.empty:
        rows.append({'variant': label, 'rows': 0, 'trade_rows': 0, 'reversal_rows': 0, 'false_flip_rate': None, 'breakout_opportunities': 0, 'breakout_capture_rate': None, 'missed_breakout_capture_rate': None})
        continue

    f = frame.copy()
    for c in ['return_1d_bps', 'correct_1d', 'correct_2d', 'eod_mfe_bps', 'eod_mae_bps']:
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors='coerce')

    if 'signal_timestamp' in f.columns:
        f = f.sort_values('signal_timestamp').reset_index(drop=True)

    is_trade = f['trade_status'].astype(str).str.upper().eq('TRADE') if 'trade_status' in f.columns else pd.Series(False, index=f.index)
    is_dir = f['direction'].astype(str).str.upper().isin(['CALL', 'PUT']) if 'direction' in f.columns else pd.Series(False, index=f.index)
    t = f[is_trade & is_dir].copy().reset_index(drop=True)

    prev_dir = t['direction'].shift(1) if not t.empty else pd.Series(dtype=object)
    reversals = t[prev_dir.notna() & (t['direction'] != prev_dir)].copy() if not t.empty else t

    false_flip_rate = None
    if not reversals.empty:
        idx = reversals.index.tolist()
        false_flags = []
        for i in idx:
            curr = t.loc[i, 'direction']
            nxt1 = t.loc[i + 1, 'direction'] if i + 1 < len(t) else None
            nxt2 = t.loc[i + 2, 'direction'] if i + 2 < len(t) else None
            false_flags.append((nxt1 is not None and nxt1 != curr) or (nxt2 is not None and nxt2 != curr))
        if false_flags:
            false_flip_rate = float(pd.Series(false_flags).mean())

    b = pd.DataFrame()
    if 'return_1d_bps' in f.columns:
        breakout_base = f[pd.to_numeric(f['return_1d_bps'], errors='coerce').notna()].copy()
        if not breakout_base.empty:
            breakout_base['abs_move_1d_bps'] = pd.to_numeric(breakout_base['return_1d_bps'], errors='coerce').abs()
            b = breakout_base[breakout_base['abs_move_1d_bps'] >= BREAKOUT_MOVE_BPS_1D].copy()

    breakout_capture_rate = None
    missed_rate = None
    if not b.empty:
        captured = (
            b['trade_status'].astype(str).str.upper().eq('TRADE')
            & pd.to_numeric(b.get('correct_1d'), errors='coerce').eq(1)
        )
        breakout_capture_rate = float(captured.mean())
        missed_rate = float(1.0 - breakout_capture_rate)

    rows.append({
        'variant': label,
        'rows': int(len(f)),
        'trade_rows': int(len(t)),
        'reversal_rows': int(len(reversals)),
        'false_flip_rate': round(false_flip_rate, 4) if false_flip_rate is not None else None,
        'breakout_opportunities': int(len(b)),
        'breakout_capture_rate': round(breakout_capture_rate, 4) if breakout_capture_rate is not None else None,
        'missed_breakout_capture_rate': round(missed_rate, 4) if missed_rate is not None else None,
    })
    f['variant'] = label
    all_frames.append(f)

summary = pd.DataFrame(rows)
summary.to_csv(out_dir / 'sweep_summary.csv', index=False)
if all_frames:
    pd.concat(all_frames, ignore_index=True).to_csv(out_dir / 'sweep_rows_combined.csv', index=False)

s = summary.dropna(subset=['missed_breakout_capture_rate', 'false_flip_rate']).copy()
best = None
if not s.empty:
    s['score'] = (1.0 - s['missed_breakout_capture_rate']) - s['false_flip_rate']
    best = s.sort_values('score', ascending=False).iloc[0].to_dict()

report = {
    'symbol': symbol,
    'start_date': str(start_d),
    'end_date': str(end_d),
    'days_evaluated': len(selected),
    'breakout_proxy': 'abs(return_1d_bps) >= threshold_bps',
    'breakout_threshold_bps': BREAKOUT_MOVE_BPS_1D,
    'best_variant_balanced_score': best,
}
(out_dir / 'sweep_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

print(summary.to_string(index=False))
print('OUTPUT_DIR', out_dir)
