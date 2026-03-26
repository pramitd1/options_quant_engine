EVENT_EXTRACTION_PROMPT = """
Extract only JSON for options event intelligence.
Required fields:
- event_type
- instrument_scope
- expected_direction
- directional_confidence
- vol_impact
- vol_confidence
- event_strength
- uncertainty_score
- gap_risk_score
- time_horizon
- catalyst_quality
- risk_flag
- summary
Use strict labels only. No prose outside JSON.
""".strip()
