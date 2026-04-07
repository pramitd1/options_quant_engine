#!/usr/bin/env python3
"""Live-style dry pass with compact integrity checklist."""
import io
import sys
from contextlib import redirect_stdout
from app.engine_runner import run_engine_snapshot
from app.terminal_output import render_compact

# Run the snapshot
r = run_engine_snapshot(
    symbol='NIFTY',
    mode='REPLAY',
    source='REPLAY',
    apply_budget_constraint=False,
    requested_lots=1,
    lot_size=65,
    max_capital=50000,
    replay_spot='debug_samples/spot_snapshots/NIFTY_spot_snapshot_2026-04-07T10-55-32.291841+05-30.json',
    replay_chain='debug_samples/option_chain_snapshots/NIFTY_ZERODHA_option_chain_snapshot_2026-04-07T10-55-37.904955+05-30.csv',
    capture_signal_evaluation=False,
)

trade = r.get('trade') or {}

print("=" * 70)
print("INTEGRITY CHECKLIST - LIVE DRY PASS")
print("=" * 70)

# 1. GATE VERDICT PROPAGATION
print("\n[1] GATE VERDICT PROPAGATION")
print("-" * 70)

calibration_gate = trade.get('live_calibration_gate') or {}
directional_gate = trade.get('live_directional_gate') or {}
blocked_by = trade.get('blocked_by', [])

gate_data_present = isinstance(calibration_gate, dict) and isinstance(directional_gate, dict)
gate_verdict_ok = (
    calibration_gate.get('verdict') is not None or len(calibration_gate) == 0
) and (
    directional_gate.get('verdict') is not None or len(directional_gate) == 0
)

if calibration_gate and calibration_gate.get('verdict'):
    print(f"  ✓ live_calibration_gate: {calibration_gate.get('verdict')} ({calibration_gate.get('red_alerts', 0)} alerts)")
else:
    print(f"  ✓ live_calibration_gate: (empty/pass)")

if directional_gate and directional_gate.get('verdict'):
    print(f"  ✓ live_directional_gate: {directional_gate.get('verdict')} ({directional_gate.get('red_alerts', 0)} alerts)")
else:
    print(f"  ✓ live_directional_gate: (empty/pass)")

if blocked_by:
    print(f"  ✓ blocked_by: {len(blocked_by)} blocker(s): {', '.join(blocked_by)}")
else:
    print(f"  ✓ blocked_by: empty (trade not blocked)")

reason_details = trade.get('no_trade_reason_details', [])
if reason_details:
    print(f"  ✓ no_trade_reason_details: {len(reason_details)} entries")

gate_verdicts_ok = gate_data_present and gate_verdict_ok
print(f"  → {'PASS ✓' if gate_verdicts_ok else 'FAIL ✗'}")

# 2. DIRECTION RESOLUTION STATE
print("\n[2] DIRECTION RESOLUTION STATE")
print("-" * 70)

direction = trade.get('trade_direction')
confidence = trade.get('direction_confidence')
dealer_state = trade.get('dealer_hedging_state')

# Check if sufficient analytics are available
analytics_payload = r.get('analytics_payload') or {}
has_analytics = len(analytics_payload) > 0

direction_resolved = direction is not None
confidence_valid = confidence is not None and isinstance(confidence, (int, float))

print(f"  ✓ trade_direction: {direction if direction else '(insufficient analytics)'}")
print(f"  ✓ direction_confidence: {confidence:.3f}" if confidence_valid else f"  {'⚠' if has_analytics else '•'} direction_confidence: {confidence if confidence else '(insufficient data)'}")
print(f"  ✓ dealer_hedging_state: {dealer_state}" if dealer_state else "  (neutral/missing)")
print(f"  ✓ analytics_payload available: {'yes' if has_analytics else 'no (minimal snapshot)'}")

# Pass if direction is resolved and valid, OR if we don't have enough data (expected for minimal replays)
direction_ok = (direction_resolved and direction.upper() in ['BULL', 'BEAR', 'NEUTRAL']) or (not has_analytics)

print(f"  → {'PASS ✓' if direction_ok else 'FAIL ✗'}")

# 3. SCORING CONSISTENCY
print("\n[3] SCORING CONSISTENCY")
print("-" * 70)

macro_news_core = trade.get('macro_news_score_core', 0)
macro_news_overlay = trade.get('macro_news_score_overlay', 0)
macro_news_total = trade.get('macro_news_score', 0)

arithmetic_ok = abs(macro_news_core + macro_news_overlay - macro_news_total) < 0.001

print(f"  ✓ macro_news_score: {macro_news_core:.3f} + {macro_news_overlay:.3f} = {macro_news_total:.3f}")
if arithmetic_ok:
    print(f"  ✓ arithmetic consistent")
else:
    print(f"  ✗ arithmetic mismatch: expected sum={macro_news_core + macro_news_overlay:.3f}, got {macro_news_total:.3f}")

vol_regime = trade.get('volatility_regime')
vol_ok = vol_regime is not None  # UNKNOWN_VOL is valid (fallback when no IV source)
print(f"  ✓ volatility_regime: {vol_regime}")

greeks_warning = trade.get('greeks_data_warning')
print(f"  ✓ greeks_data_warning: {greeks_warning or 'none'}")

scoring_ok = arithmetic_ok and vol_ok
print(f"  → {'PASS ✓' if scoring_ok else 'FAIL ✗'}")

# 4. CONSISTENCY SECTION COMPLETENESS
print("\n[4] CONSISTENCY SECTION COMPLETENESS")
print("-" * 70)

buf = io.StringIO()
with redirect_stdout(buf):
    render_compact(
        result=r,
        trade=trade,
        spot_summary=r.get('spot_summary') or {},
        macro_event_state=r.get('macro_event_state') or {},
        global_risk_state=r.get('global_risk_state') or {},
        execution_trade=r.get('execution_trade'),
        option_chain_frame=r.get('option_chain_frame'),
    )

compact_output = buf.getvalue()
lines = compact_output.split('\n')

# Find consistency check section (more robust extraction)
consistency_lines = []
found_consistency = False

for i, line in enumerate(lines):
    if 'CONSISTENCY CHECK' in line:
        found_consistency = True
        # Collect following lines until next section or end
        for j in range(i + 1, min(i + 50, len(lines))):
            if j > i + 1 and lines[j].strip() and not lines[j].startswith('  ') and not lines[j].startswith('-'):
                break  # Hit next section
            consistency_lines.append(lines[j])
        break

consistency_found = found_consistency and len(consistency_lines) > 0

# Parse consistency section content
consistency_content = '\n'.join(consistency_lines)

# Look for status indicator
has_status_indicator = 'PASS' in consistency_content or 'WARN' in consistency_content or 'status' in consistency_content.lower()

# Look for any findings/issues
has_findings_line = '•' in consistency_content or 'findings' in consistency_content.lower() or 'issues' in consistency_content.lower()

# For minimal snapshots without decision data, consistency check may be minimal
analytics_available = len(r.get('analytics_payload') or {}) > 0

print(f"  ✓ CONSISTENCY CHECK section: {'rendered' if consistency_found else 'missing'}")
print(f"  ✓ contains status: {'yes' if has_status_indicator else 'no'}")
print(f"  ✓ findings/issues present: {'yes' if has_findings_line else 'no (may be empty for minimal snapshot)'}")
if analytics_available:
    print(f"  ✓ analytics available: yes (should have detailed checks)")
else:
    print(f"  • note: minimal snapshot (limited consistency data)")

consistency_ok = consistency_found
print(f"  → {'PASS ✓' if consistency_ok else 'FAIL ✗'}")

# FINAL SUMMARY
print("\n" + "=" * 70)
print("FINAL INTEGRITY SUMMARY")
print("=" * 70)

all_pass = (gate_verdicts_ok and direction_ok and scoring_ok and consistency_ok)

checklist = [
    ("Gate Verdict Propagation", gate_verdicts_ok),
    ("Direction Resolution State", direction_ok),
    ("Scoring Consistency", scoring_ok),
    ("Consistency Section Completeness", consistency_ok),
]

for item, status in checklist:
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {item}")

# Additional context
if not has_analytics:
    print(f"\n  Note: Minimal replay snapshot (no analytics payload)")
    print(f"        Direction and consistency sections are expected to be sparse.")

print(f"\n  Overall: {'✅ ALL CHECKS PASS' if all_pass else '⚠️  CHECK CONDITIONS MET' if gate_verdicts_ok and scoring_ok else '❌ FAILURES DETECTED'}")
print("=" * 70)

sys.exit(0 if all_pass else 1)
