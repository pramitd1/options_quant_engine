#!/usr/bin/env python3
"""Build the disabled regime-parameter candidate artifact from replay evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.signal_evaluation.regime_cell_counterfactual import (  # noqa: E402
    DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR,
    LATEST_COUNTERFACTUAL_JSON_FILENAME,
)
from research.signal_evaluation.regime_parameter_artifact import (  # noqa: E402
    DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR,
    DEFAULT_REGIME_PARAMETER_ARTIFACT_PATH,
    DEFAULT_REGIME_PARAMETER_MARKDOWN_PATH,
    write_regime_parameter_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counterfactual",
        type=Path,
        default=DEFAULT_REGIME_CELL_COUNTERFACTUAL_DIR / LATEST_COUNTERFACTUAL_JSON_FILENAME,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REGIME_PARAMETER_ARTIFACT_PATH)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_REGIME_PARAMETER_MARKDOWN_PATH)
    parser.add_argument(
        "--documentation",
        type=Path,
        default=DEFAULT_DOCUMENTATION_REGIME_REVIEW_DIR / "latest_regime_parameter_candidate.md",
    )
    parser.add_argument("--min-suppressed-labels", type=int, default=20)
    parser.add_argument("--no-doc-copy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_regime_parameter_artifact(
        counterfactual_path=args.counterfactual,
        output_path=args.output,
        markdown_path=args.markdown,
        documentation_path=None if args.no_doc_copy else args.documentation,
        min_suppressed_labels=args.min_suppressed_labels,
    )
    artifact = result["artifact"]
    print(
        json.dumps(
            {
                "artifact_version": artifact.get("artifact_version"),
                "status": artifact.get("status"),
                "rule_count": artifact.get("rule_count"),
                "excluded_cell_count": artifact.get("excluded_cell_count"),
                "source_assessment_status": artifact.get("source_assessment_status"),
                "artifact_path": result.get("artifact_path"),
                "markdown_path": result.get("markdown_path"),
                "documentation_markdown_path": result.get("documentation_markdown_path"),
                "live_activation": artifact.get("live_activation"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
