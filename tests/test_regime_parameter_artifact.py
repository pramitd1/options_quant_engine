from __future__ import annotations

import json
from pathlib import Path

from research.signal_evaluation.regime_parameter_artifact import (
    build_regime_parameter_artifact,
    write_regime_parameter_artifact,
)


def _counterfactual_report() -> dict:
    return {
        "assessment_status": "COUNTERFACTUAL_POSITIVE",
        "assessment_basis": "conservative_research_replay_no_watchlist_promotions",
        "cell_summary": [
            {
                "impact_status": "REPLAY_SUPPORTS_SUPPRESSION",
                "matched_cell": "gamma_regime=NEGATIVE_GAMMA<br>volatility_regime=VOL_EXPANSION<br>direction=PUT<br>macro_risk_bucket=RISK_OFF",
                "matched_signal_count": 853,
                "baseline_trade_count": 340,
                "suppressed_trade_count": 340,
                "suppressed_label_count": 340,
                "suppressed_avg_return_60m_bps": -17.9694,
                "avoided_suppressed_return_60m_bps": 6109.58,
            },
            {
                "impact_status": "REPLAY_REJECTS_SUPPRESSION",
                "matched_cell": "gamma_regime=POSITIVE_GAMMA<br>volatility_regime=VOL_EXPANSION<br>direction=CALL<br>macro_risk_bucket=RISK_OFF",
                "suppressed_label_count": 141,
            },
            {
                "impact_status": "REPLAY_SUPPORTS_SUPPRESSION",
                "matched_cell": "gamma_regime=NEUTRAL_GAMMA<br>volatility_regime=NORMAL_VOL<br>direction=PUT<br>macro_risk_bucket=RISK_OFF",
                "suppressed_label_count": 5,
            },
        ],
    }


def test_build_regime_parameter_artifact_includes_only_supported_suppression_rules():
    artifact = build_regime_parameter_artifact(
        _counterfactual_report(),
        source_counterfactual_path="counterfactual.json",
        min_suppressed_labels=20,
    )

    assert artifact["artifact_version"] == "regime_parameter_candidate_v1"
    assert artifact["status"] == "CANDIDATE_RESEARCH_ONLY"
    assert artifact["live_activation"]["enabled"] is False
    assert artifact["live_activation"]["runtime_config_changed"] is False
    assert artifact["rule_count"] == 1
    assert artifact["excluded_cell_count"] == 2

    rule = artifact["rules"][0]
    assert rule["action"] == "SUPPRESS_OR_AVOID"
    assert rule["match"]["gamma_regime"] == "NEGATIVE_GAMMA"
    assert rule["match"]["direction"] == "PUT"
    assert rule["research_adjustments"]["allow_trade"] is False
    assert rule["evidence"]["suppressed_label_count"] == 340


def test_write_regime_parameter_artifact_outputs_json_and_markdown(tmp_path: Path):
    counterfactual_path = tmp_path / "counterfactual.json"
    output_path = tmp_path / "candidate.json"
    markdown_path = tmp_path / "candidate.md"
    docs_path = tmp_path / "docs.md"
    counterfactual_path.write_text(json.dumps(_counterfactual_report()), encoding="utf-8")

    result = write_regime_parameter_artifact(
        counterfactual_path=counterfactual_path,
        output_path=output_path,
        markdown_path=markdown_path,
        documentation_path=docs_path,
        min_suppressed_labels=20,
    )

    assert Path(result["artifact_path"]).exists()
    assert Path(result["markdown_path"]).exists()
    assert Path(result["documentation_markdown_path"]).exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["rule_count"] == 1
