from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SHADOW_MODE_SCRIPT = ROOT / "scripts" / "ops" / "run_threshold_shadow_mode.py"


def _shadow_mode_dataset(days: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01 09:20:00+05:30")
    for idx in range(days):
        high_score = idx >= 30
        rows.append(
            {
                "signal_id": f"shadow-mode-{idx}",
                "signal_timestamp": (base + pd.Timedelta(days=idx)).isoformat(),
                "symbol": "NIFTY",
                "source": "unit",
                "mode": "TEST",
                "direction": "CALL" if idx % 2 == 0 else "PUT",
                "trade_status": "TRADE",
                "signal_regime": "EXPANSION_BIAS" if high_score else "CONFLICTED",
                "macro_regime": "RISK_ON" if idx % 3 else "RISK_OFF",
                "gamma_regime": "SHORT_GAMMA_ZONE" if idx % 2 else "LONG_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "global_risk_state": "CALM",
                "composite_signal_score": 82.0 if high_score else 52.0,
                "tradeability_score": 78.0 if high_score else 45.0,
                "move_probability": 0.72 if high_score else 0.48,
                "ml_confidence_score": 0.74 if high_score else 0.42,
                "correct_60m": 1.0 if high_score else 0.0,
                "signed_return_60m_bps": 24.0 if high_score else -8.0,
                "calibration_label": 1.0 if high_score else 0.0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 24.0 if high_score else -8.0,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_threshold_shadow_mode_runs_full_chain_from_one_command(tmp_path: Path):
    dataset_path = tmp_path / "signals_dataset.csv"
    output_dir = tmp_path / "shadow_mode_output"
    _shadow_mode_dataset().to_csv(dataset_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            str(SHADOW_MODE_SCRIPT),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "unit_shadow_mode",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["status"] == "SUCCESS"
    assert payload["runtime_config_changed"] is False
    assert payload["governance_status"] == "PROMOTE_TO_REVIEW"
    assert payload["policy_experiment_status"] == "APPROVED_FOR_POLICY_EXPERIMENT"
    assert payload["shadow_status"] == "SHADOW_SIMULATION_READY"
    assert payload["shadow_review_status"] == "PROMOTION_READY"
    assert payload["promotion_review_status"] == "PROMOTION_REVIEW_READY"
    assert payload["post_promotion_monitor_status"] == "POST_PROMOTION_SKIPPED_NO_APPROVAL"
    assert payload["adoption_reconciliation_status"] == "UNKNOWN_ADOPTION_STATE"
    assert payload["manual_promotion_review_required"] is True

    for artifact_group in [
        "threshold_governance",
        "threshold_policy_experiment",
        "threshold_shadow_simulation",
        "threshold_shadow_review",
        "threshold_promotion_review",
        "threshold_post_promotion_monitor",
        "threshold_adoption_reconciliation",
    ]:
        latest_json = Path(payload[artifact_group]["latest_json_path"])
        latest_markdown = Path(payload[artifact_group]["latest_markdown_path"])
        assert latest_json.exists()
        assert latest_markdown.exists()

    assert Path(payload["checkpoint_json"]).exists()
    assert Path(payload["latest_checkpoint_json"]).exists()
