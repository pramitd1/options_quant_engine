"""
EVSizingPredictor
=================
Extends the research dual-model pathway with Expected Value (EV) based
sizing derived from conditional return tables.

This predictor:
  1. Runs the full engine pipeline (rule + ML blend)
  2. Overlays GBT ranking + LogReg calibration (research dual-model)
  3. Looks up the conditional return table for the signal bucket
  4. Computes per-signal EV and maps it to a size multiplier
  5. Adjusts hybrid_move_probability: blocks negative-EV signals,
     scales positive-EV signals proportionally

The conditional return table and normalizer bounds are lazy-loaded once
from the backtest dataset and cached for the process lifetime.

RESEARCH / OPTIONAL - set OQE_PREDICTION_METHOD=ev_sizing to activate.
Default production method (blended) is unchanged.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import numpy as np

from engine.predictors.inference_row import build_inference_row
from engine.predictors.protocol import PredictionResult

logger = logging.getLogger(__name__)

# Module-level lazy caches
_CRT_CACHE = None  # ConditionalReturnTable | None
_EV_BOUNDS = None  # tuple[float, float] | None
_CRT_LOAD_ATTEMPTED = False

_BACKTEST_PARQUET = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "signal_evaluation"
    / "backtest_signals_dataset.parquet"
)


def _load_crt_and_bounds() -> None:
    """
    Build the conditional return table and EV normalizer bounds from the
    backtest dataset. Called once; results cached at module level.

    Requires ml_rank_bucket / ml_confidence_bucket columns to already exist
    in the parquet. infer_batch() is intentionally not run here because it can
    be slow and brittle under sklearn version skew.
    """
    global _CRT_CACHE, _EV_BOUNDS, _CRT_LOAD_ATTEMPTED
    _CRT_LOAD_ATTEMPTED = True

    try:
        import pandas as pd
        from research.ml_evaluation.ev_and_regime_policy.conditional_return_tables import (
            build_conditional_return_table,
        )
        from research.ml_evaluation.ev_and_regime_policy.ev_sizing_model import (
            build_ev_normalizer,
            compute_ev,
        )
    except ImportError:
        logger.warning("EV research modules not available; EV sizing disabled.")
        return

    if not _BACKTEST_PARQUET.exists():
        logger.warning(
            "Backtest dataset not found at %s; EV sizing disabled.", _BACKTEST_PARQUET
        )
        return

    try:
        df = pd.read_parquet(_BACKTEST_PARQUET)
        missing = [
            c
            for c in ("ml_rank_bucket", "ml_confidence_bucket")
            if c not in df.columns
        ]
        if missing:
            logger.warning(
                "EV sizing disabled; backtest parquet is missing columns %s. "
                "Re-run scripts/signal_evaluation_report.py to add ML columns.",
                missing,
            )
            return

        table = build_conditional_return_table(df)
        _CRT_CACHE = table

        ev_values = []
        for cell in table.cells.values():
            ev_raw, _, _ = compute_ev(cell.hit_rate, cell)
            ev_values.append(ev_raw)

        _EV_BOUNDS = build_ev_normalizer(np.array(ev_values)) if ev_values else (0.0, 1.0)

        logger.info(
            "EV sizing loaded: %d CRT cells, bounds=(%.2f, %.2f)",
            len(table.cells),
            _EV_BOUNDS[0],
            _EV_BOUNDS[1],
        )
    except Exception:
        logger.exception("Failed to build conditional return table for EV sizing.")


class EVSizingPredictor:
    """
    EV-based sizing predictor built on the dual-model research layer.

    Computes per-signal EV from conditional return tables and uses it to
    adjust probability output.
    """

    @property
    def name(self) -> str:
        return "ev_sizing"

    def predict(self, market_ctx: dict[str, Any]) -> PredictionResult:
        from engine.trading_support.probability import _compute_probability_state_impl

        # 1. Standard engine pipeline
        raw = _compute_probability_state_impl(**market_ctx)

        rule_prob = raw.get("rule_move_probability")
        ml_prob = raw.get("ml_move_probability")
        model_features = raw.get("model_features")
        components = raw.get("components", {})
        hybrid_prob = raw.get("hybrid_move_probability")

        # 2. Research dual-model overlay
        rank_score = None
        confidence_score = None
        rank_bucket = None
        confidence_bucket = None
        research_prob = None
        try:
            from research.ml_models.ml_inference import infer_single

            result = infer_single(build_inference_row(market_ctx, raw))
            if result is not None:
                rank_score = result.ml_rank_score
                confidence_score = result.ml_confidence_score
                rank_bucket = result.ml_rank_bucket
                confidence_bucket = result.ml_confidence_bucket
                if confidence_score is not None:
                    research_prob = confidence_score
                elif rank_score is not None:
                    research_prob = rank_score
        except Exception:
            logger.debug(
                "Research ML inference unavailable; falling back to engine blend",
                exc_info=True,
            )

        # 3. EV computation
        ev_raw = None
        ev_normalized = None
        ev_bucket = None
        ev_size_multiplier = 1.0
        ev_reliability = None
        ev_expected_gain = None
        ev_expected_loss = None
        ev_backed_off = False

        if rank_bucket is not None and confidence_bucket is not None:
            try:
                self._ensure_crt_loaded()

                if _CRT_CACHE is not None and _EV_BOUNDS is not None:
                    from research.ml_evaluation.ev_and_regime_policy.conditional_return_tables import (
                        lookup,
                    )
                    from research.ml_evaluation.ev_and_regime_policy.ev_sizing_model import (
                        classify_ev_bucket,
                        compute_ev,
                        compute_ev_reliability,
                        ev_to_size_multiplier,
                        normalize_ev,
                    )

                    p_win = confidence_score if confidence_score is not None else 0.5
                    regime = str(components.get("gamma_regime", "UNKNOWN"))

                    cell = lookup(_CRT_CACHE, rank_bucket, confidence_bucket, regime)
                    ev_raw, ev_expected_gain, ev_expected_loss = compute_ev(p_win, cell)
                    ev_normalized = normalize_ev(ev_raw, _EV_BOUNDS[0], _EV_BOUNDS[1])
                    ev_bucket = classify_ev_bucket(ev_normalized)
                    ev_size_multiplier = ev_to_size_multiplier(ev_normalized)
                    ev_reliability = compute_ev_reliability(cell)
                    ev_backed_off = cell.backed_off
            except Exception:
                logger.debug("EV computation failed; using defaults", exc_info=True)

        # 4. Adjust probability based on EV sizing
        effective_prob = research_prob if research_prob is not None else hybrid_prob
        if ev_size_multiplier == 0.0 and effective_prob is not None:
            logger.debug(
                "EV sizing: negative EV (ev_norm=%.3f, bucket=%s); blocking signal "
                "(prob %.3f -> 0.0)",
                ev_normalized,
                ev_bucket,
                effective_prob,
            )
            effective_prob = 0.0
        elif ev_size_multiplier < 1.0 and effective_prob is not None:
            logger.debug(
                "EV sizing: sub-par EV (ev_norm=%.3f, bucket=%s, mult=%.2f); "
                "scaling prob %.3f -> %.3f",
                ev_normalized,
                ev_bucket,
                ev_size_multiplier,
                effective_prob,
                effective_prob * ev_size_multiplier,
            )
            effective_prob = effective_prob * ev_size_multiplier

        # 5. Enrich components with diagnostics
        components["research_rank_score"] = rank_score
        components["research_confidence_score"] = confidence_score
        components["engine_hybrid_probability"] = hybrid_prob
        components["ev_raw"] = ev_raw
        components["ev_normalized"] = ev_normalized
        components["ev_bucket"] = ev_bucket
        components["ev_size_multiplier"] = ev_size_multiplier
        components["ev_reliability"] = ev_reliability
        components["ev_expected_gain"] = ev_expected_gain
        components["ev_expected_loss"] = ev_expected_loss
        components["ev_backed_off"] = ev_backed_off

        return PredictionResult(
            rule_move_probability=rule_prob,
            ml_move_probability=ml_prob,
            hybrid_move_probability=effective_prob,
            model_features=model_features,
            components=components,
            predictor_name=self.name,
        )

    @staticmethod
    def _ensure_crt_loaded() -> None:
        """Trigger one-time lazy load of CRT and normalizer bounds."""
        if not _CRT_LOAD_ATTEMPTED:
            _load_crt_and_bounds()
