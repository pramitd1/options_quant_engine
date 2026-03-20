#!/usr/bin/env python3
"""
ML Inference Gap Diagnostic
============================
Investigates why 63% of live signals have no ML scores.

Checks:
1. sklearn version compatibility
2. Model loading status
3. Feature extraction failures
4. Inference pipeline execution on live signals
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

# Add repo to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# SECTION 1: Environment & Version Checks
# ──────────────────────────────────────────────────────────────────────

def check_versions():
    print("\n" + "="*80)
    print("SECTION 1: Version & Environment Checks")
    print("="*80)
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"scikit-learn version: {sklearn.__version__}")
    logger.info(f"numpy version: {np.__version__}")
    logger.info(f"pandas version: {pd.__version__}")
    
    # Check ML config
    from research.ml_models.ml_config import ML_RESEARCH_ENABLED, GBT_MODEL_PATH, LOGREG_MODEL_PATH
    logger.info(f"ML_RESEARCH_ENABLED: {ML_RESEARCH_ENABLED}")
    logger.info(f"GBT_MODEL_PATH exists: {GBT_MODEL_PATH.exists()}")
    logger.info(f"LOGREG_MODEL_PATH exists: {LOGREG_MODEL_PATH.exists()}")
    
    return {
        "sklearn_version": sklearn.__version__,
        "ml_enabled": ML_RESEARCH_ENABLED,
        "models_exist": GBT_MODEL_PATH.exists() and LOGREG_MODEL_PATH.exists(),
    }


# ──────────────────────────────────────────────────────────────────────
# SECTION 2: Model Loading Tests
# ──────────────────────────────────────────────────────────────────────

def test_model_loading():
    print("\n" + "="*80)
    print("SECTION 2: Model Loading & Deserialization")
    print("="*80)
    
    results = {}
    
    # Test GBT model
    try:
        from research.ml_models.gbt_model import _load_model as load_gbt
        model, meta = load_gbt()
        if model is None:
            logger.error("GBT model failed to load (None returned)")
            results["gbt_status"] = "FAILED"
        else:
            logger.info(f"GBT model loaded successfully: {type(model)}")
            logger.info(f"GBT has meta: {meta is not None}")
            if meta:
                logger.info(f"  - feature_names: {len(meta.get('feature_names', []))} features")
                logger.info(f"  - model_version: {meta.get('model_version', 'unknown')}")
            results["gbt_status"] = "OK"
    except Exception as e:
        logger.exception(f"GBT model loading exception: {e}")
        results["gbt_status"] = "EXCEPTION"
    
    # Test LogReg model
    try:
        from research.ml_models.logreg_model import _load_model as load_logreg
        model, meta = load_logreg()
        if model is None:
            logger.error("LogReg model failed to load (None returned)")
            results["logreg_status"] = "FAILED"
        else:
            logger.info(f"LogReg model loaded successfully: {type(model)}")
            logger.info(f"LogReg has meta: {meta is not None}")
            if meta:
                logger.info(f"  - feature_names: {len(meta.get('feature_names', []))} features")
                logger.info(f"  - model_version: {meta.get('model_version', 'unknown')}")
            results["logreg_status"] = "OK"
    except Exception as e:
        logger.exception(f"LogReg model loading exception: {e}")
        results["logreg_status"] = "EXCEPTION"
    
    return results


# ──────────────────────────────────────────────────────────────────────
# SECTION 3: Feature Extraction Tests
# ──────────────────────────────────────────────────────────────────────

def test_feature_extraction():
    print("\n" + "="*80)
    print("SECTION 3: Feature Extraction Tests")
    print("="*80)
    
    from models.expanded_feature_builder import extract_features, FEATURE_NAMES
    
    # Test 1: Minimal signal
    minimal_signal = {
        "gamma_regime": "NEUTRAL",
        "final_flow_signal": "NEUTRAL",
        "volatility_regime": "NORMAL",
    }
    
    try:
        features = extract_features(minimal_signal)
        logger.info(f"✓ Minimal signal extraction succeeded: {features.shape}")
        logger.info(f"  Feature values: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    except Exception as e:
        logger.exception(f"✗ Minimal signal extraction failed: {e}")
    
    # Test 2: Rich signal (simulating backtest)
    rich_signal = {
        "gamma_regime": "SHORT_GAMMA",
        "gamma_regime_numeric": 1.0,
        "final_flow_signal": "BULLISH",
        "flow_signal_numeric": 1.0,
        "volatility_regime": "VOL_EXPANSION",
        "vol_regime_numeric": 2.0,
        "dealer_hedging_bias": "UPSIDE_ACCEL",
        "spot_vs_flip": "ABOVE",
        "spot_vs_flip_numeric": 1.0,
        "liquidity_vacuum_state": "BREAKOUT_ZONE",
        "move_probability": 0.65,
        "gamma_flip_distance_pct": 2.5,
        "vacuum_strength": 0.8,
        "hedging_flow_ratio": 0.45,
        "smart_money_flow_score": 0.75,
        "atm_iv_percentile": 0.65,
        "intraday_range_pct": 1.2,
        "lookback_avg_range_pct": 1.5,
        "gap_pct": 0.3,
        "close_vs_prev_close_pct": 0.5,
        "spot_in_day_range": 0.6,
        "dealer_position": None,
        "vanna_regime": None,
        "charm_regime": None,
        "india_vix_level": 18.5,
        "india_vix_change_24h": 2.3,
        "oil_shock_score": 0.1,
        "commodity_risk_score": 0.15,
        "volatility_shock_score": 0.2,
        "macro_event_risk_score": 0.0,
        "days_to_expiry": 5.0,
        "weekday": 2.0,
    }
    
    try:
        features = extract_features(rich_signal)
        logger.info(f"✓ Rich signal extraction succeeded: {features.shape}")
        logger.info(f"  Feature values: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    except Exception as e:
        logger.exception(f"✗ Rich signal extraction failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# SECTION 4: Inference Pipeline Tests
# ──────────────────────────────────────────────────────────────────────

def test_inference_pipeline():
    print("\n" + "="*80)
    print("SECTION 4: Inference Pipeline Tests")
    print("="*80)
    
    from research.ml_models.ml_inference import infer_single
    
    # Test signal
    test_signal = {
        "gamma_regime": "SHORT_GAMMA",
        "final_flow_signal": "BULLISH",
        "volatility_regime": "VOL_EXPANSION",
        "dealer_hedging_bias": "UPSIDE_ACCEL",
        "move_probability": 0.65,
        "gamma_flip_distance_pct": 2.5,
        "vacuum_strength": 0.8,
        "hedging_flow_ratio": 0.45,
        "smart_money_flow_score": 0.75,
        "atm_iv_percentile": 0.65,
        "intraday_range_pct": 1.2,
        "lookback_avg_range_pct": 1.5,
        "gap_pct": 0.3,
        "close_vs_prev_close_pct": 0.5,
        "india_vix_level": 18.5,
        "india_vix_change_24h": 2.3,
        "days_to_expiry": 5.0,
        "trade_status": "TRADE",
        "direction": "CALL",
    }
    
    try:
        result = infer_single(test_signal)
        logger.info(f"✓ infer_single() succeeded")
        logger.info(f"  ML rank score: {result.ml_rank_score}")
        logger.info(f"  ML confidence score: {result.ml_confidence_score}")
        logger.info(f"  ML rank bucket: {result.ml_rank_bucket}")
        logger.info(f"  ML confidence bucket: {result.ml_confidence_bucket}")
        logger.info(f"  ML agreement: {result.ml_agreement_with_engine}")
        
        # Check if either score is populated
        if result.ml_rank_score is not None or result.ml_confidence_score is not None:
            logger.info("✓ At least one ML score was populated")
            return True
        else:
            logger.warning("✗ Both ML scores are None (inference silently failed)")
            return False
    except Exception as e:
        logger.exception(f"✗ infer_single() raised exception: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# SECTION 5: Live Signal Analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_live_signals():
    print("\n" + "="*80)
    print("SECTION 5: Live Signal Dataset Analysis")
    print("="*80)
    
    csv_path = ROOT / "data_store" / "signals_dataset_cumul.csv"
    
    if not csv_path.exists():
        logger.error(f"Live dataset not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} signals from {csv_path.name}")
    
    # Analysis
    has_ml_rank = df["ml_rank_score"].notna().sum()
    has_ml_conf = df["ml_confidence_score"].notna().sum()
    total = len(df)
    
    logger.info(f"\nML Score Population:")
    logger.info(f"  Total signals: {total}")
    logger.info(f"  With ml_rank_score: {has_ml_rank} ({100*has_ml_rank/total:.1f}%)")
    logger.info(f"  With ml_confidence_score: {has_ml_conf} ({100*has_ml_conf/total:.1f}%)")
    logger.info(f"  With both: {(df['ml_rank_score'].notna() & df['ml_confidence_score'].notna()).sum()}")
    logger.info(f"  With neither: {(df['ml_rank_score'].isna() & df['ml_confidence_score'].isna()).sum()}")
    
    # Check required fields
    logger.info(f"\nRequired Field Availability:")
    required_fields = [
        "gamma_regime", "final_flow_signal", "volatility_regime",
        "move_probability", "india_vix_level", "days_to_expiry",
        "trade_status", "direction"
    ]
    
    for field in required_fields:
        if field in df.columns:
            present = df[field].notna().sum()
            logger.info(f"  {field}: {present}/{total} ({100*present/total:.1f}%)")
        else:
            logger.warning(f"  {field}: NOT IN DATASET")
    
    # Check consistency: signals with outcomes but no ML scores
    missing_ml_mask = df["ml_rank_score"].isna() & df["ml_confidence_score"].isna()
    has_outcome_mask = df["spot_60m"].notna()
    
    missing_but_evaluated = (missing_ml_mask & has_outcome_mask).sum()
    logger.info(f"\nSignals with outcomes but NO ML scores: {missing_but_evaluated}")
    
    # Sample a signal with missing ML scores to understand why
    if missing_but_evaluated > 0:
        sample = df[missing_ml_mask & has_outcome_mask].iloc[0]
        logger.info(f"\nSample signal with missing ML scores:")
        for col in ["timestamp", "spot", "gamma_regime", "final_flow_signal", 
                    "volatility_regime", "move_probability", "trade_status", "direction",
                    "ml_rank_score", "ml_confidence_score"]:
            if col in sample.index:
                logger.info(f"  {col}: {sample[col]}")


# ──────────────────────────────────────────────────────────────────────
# SECTION 6: Root Cause Analysis
# ──────────────────────────────────────────────────────────────────────

def root_cause_analysis():
    print("\n" + "="*80)
    print("SECTION 6: Root Cause Analysis & Recommendations")
    print("="*80)
    
    logger.info("""
Potential root causes for 63% ML inference failure:

1. SKLEARN VERSION MISMATCH
   - Models pickled with sklearn 1.7.2, loaded under 1.6.1
   - joblib will silently fail to unpickle incompatible models
   - FIX: Re-save models with current sklearn version, or upgrade sklearn

2. MODELS NOT FOUND
   - Check models_store/registry/ has GBT_shallow_v1 and LogReg_ElasticNet_v1
   - FIX: Verify model files exist and are readable

3. FEATURE EXTRACTION FAILURES
   - Missing required fields (gamma_regime, move_probability, etc.)
   - Feature vector shape mismatch
   - NaN/inf values causing inference to fail
   - FIX: Add data validation and filtering

4. INFERENCE PIPELINE NOT TRIGGERED
   - ML_RESEARCH_ENABLED = 0 in config
   - Signals generated before ML inference was added
   - Concurrent batch jobs not running ML inference
   - FIX: Verify config setting, re-run signals through inference

5. SIGNAL SOURCE FILTERING
   - Live signals might be from different source/mode than backtest
   - Different field mapping or schema
   - FIX: Unify signal schema across all sources
    """)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  ML INFERENCE GAP DIAGNOSTIC".center(78) + "║")
    print("║" + "  Investigating why 63% of live signals have no ML scores".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        v = check_versions()
        m = test_model_loading()
        test_feature_extraction()
        inference_ok = test_inference_pipeline()
        analyze_live_signals()
        root_cause_analysis()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        print(f"sklearn version: {v['sklearn_version']}")
        print(f"ML infrastructure: {'✓ OK' if v['models_exist'] else '✗ PROBLEM: Models not found'}")
        print(f"Model loading: GBT={m.get('gbt_status')}, LogReg={m.get('logreg_status')}")
        print(f"Test inference: {'✓ WORKING' if inference_ok else '✗ FAILED'}")
        print("\nIf inference works in tests but fails in live signals,")
        print("the issue is likely SKLEARN VERSION MISMATCH on model (de)serialization.")
        print("\nNext steps:")
        print("1. Re-train and save models with current sklearn version")
        print("2. Or upgrade/downgrade sklearn to match model training version")
        print("3. Add instrumentation to log inference failures in production")
        
    except Exception as e:
        logger.exception(f"Diagnostic failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
