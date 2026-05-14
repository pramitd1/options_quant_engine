from __future__ import annotations

import pandas as pd

from research.signal_evaluation.confidence import (
    confidence_intervals_overlap,
    mean_confidence_interval,
    outcome_confidence_fields,
    sample_quality,
    wilson_interval,
)


def test_sample_quality_labels_sparse_and_reliable_samples():
    assert sample_quality(0) == "NO_EVIDENCE"
    assert sample_quality(12) == "INSUFFICIENT_EVIDENCE"
    assert sample_quality(40) == "LOW_CONFIDENCE"
    assert sample_quality(120) == "RELIABLE"


def test_outcome_confidence_fields_include_wilson_and_return_intervals():
    fields = outcome_confidence_fields(
        pd.Series([1, 1, 0, 1, 0, 1]),
        pd.Series([15.0, 12.0, -8.0, 20.0, -5.0, 18.0]),
        min_sample=3,
        strong_sample=6,
    )

    assert fields["sample_quality"] == "RELIABLE"
    assert fields["hit_rate_label_count"] == 6
    assert fields["return_label_count"] == 6
    assert fields["hit_rate_ci_low"] is not None
    assert fields["hit_rate_ci_high"] is not None
    assert fields["return_ci_low_bps"] is not None
    assert fields["return_ci_high_bps"] is not None


def test_intervals_overlap_handles_missing_values():
    low, high = wilson_interval(8, 10)
    ret_low, ret_high = mean_confidence_interval([1.0, 2.0, 3.0, 4.0])

    assert confidence_intervals_overlap(low, high, low, high) is True
    assert confidence_intervals_overlap(ret_low, ret_high, None, 5.0) is None
