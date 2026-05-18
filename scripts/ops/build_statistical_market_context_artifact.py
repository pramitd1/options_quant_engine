#!/usr/bin/env python3
"""Build the compact statistical-market-context artifact for live engine use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.signal_evaluation.statistical_market_context_artifact import (  # noqa: E402
    DEFAULT_STATISTICAL_CONTEXT_ARTIFACT_PATH,
    DEFAULT_STATISTICAL_MARKET_STRUCTURE_REPORT_PATH,
    write_statistical_market_context_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill the statistical market-structure study into a compact engine artifact."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_STATISTICAL_MARKET_STRUCTURE_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATISTICAL_CONTEXT_ARTIFACT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = write_statistical_market_context_artifact(report_path=args.report, artifact_path=args.output)
    artifact = result["artifact"]
    payload = {
        "artifact_path": result["artifact_path"],
        "artifact_version": artifact.get("artifact_version"),
        "source_run_id": artifact.get("source_run_id"),
        "numeric_prior_features": sorted((artifact.get("numeric_bucket_priors") or {}).keys()),
        "categorical_prior_features": sorted((artifact.get("categorical_bucket_priors") or {}).keys()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
