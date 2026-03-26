from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_policy.event_overlay import apply_event_overlay
from features.event_features import aggregate_event_features
from nlp.extraction.structured_extractor import extract_structured_event


def run_demo() -> dict:
    samples = [
        "RELIANCE reports earnings beat, raises guidance and wins large order from PSU client",
        "SEBI probes midcap infra company after alleged disclosure violations",
        "Large private bank sees CEO transition; management says business momentum stable",
        "Unconfirmed report suggests possible merger talks in auto ancillary space",
    ]

    structured = []
    for text in samples:
        item = extract_structured_event(text=text, source="demo")
        if item is not None:
            structured.append(item)

    event_features = aggregate_event_features(structured, direction_hint="CALL")
    overlay = apply_event_overlay(
        direction="CALL",
        event_features=event_features.to_dict(),
        enabled=True,
    )

    payload = {
        "sample_events": samples,
        "structured_events": [item.to_dict() for item in structured],
        "event_features": event_features.to_dict(),
        "overlay_decision": overlay.to_dict(),
        "explanations": event_features.explanation_lines,
    }

    artifact_path = Path("debug_samples") / "event_intelligence_demo_output.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_demo()
    print(json.dumps(result, indent=2))
