#!/usr/bin/env python3
"""
ML Inference Backfill — Apply ML Scores to Cumulative Signals
==============================================================

Retroactively applies ML inference to all cumulative signals that are missing
ml_rank_score and ml_confidence_score.

This fills the infrastructure gap where live signals are captured without
running through the research evaluation layer.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def apply_ml_inference_to_cumulative():
    """Load cumulative signals, apply ML inference, save back."""
    
    from research.signal_evaluation.dataset import (
        CUMULATIVE_DATASET_PATH,
        load_cumulative_dataset,
        write_signals_dataset,
    )
    from research.ml_models.ml_inference import infer_single, infer_batch
    from research.ml_models.ml_config import ML_RESEARCH_ENABLED
    
    if not ML_RESEARCH_ENABLED:
        logger.error("ML_RESEARCH_ENABLED is False. Set OQE_ML_RESEARCH_ENABLED=1 and retry.")
        sys.exit(1)
    
    logger.info(f"Loading cumulative signals from {CUMULATIVE_DATASET_PATH}")
    df = load_cumulative_dataset()
    
    if df.empty:
        logger.warning("Cumulative dataset is empty")
        return
    
    logger.info(f"Loaded {len(df)} total signals")
    
    # Check which signals need ML inference
    has_ml_rank = df["ml_rank_score"].notna()
    has_ml_conf = df["ml_confidence_score"].notna()
    has_both = has_ml_rank & has_ml_conf
    needs_inference = ~has_both
    
    n_needs = needs_inference.sum()
    n_have = has_both.sum()
    
    logger.info(f"\nCurrent ML Score Population:")
    logger.info(f"  Signals with BOTH scores: {n_have} ({100*n_have/len(df):.1f}%)")
    logger.info(f"  Signals needing inference: {n_needs} ({100*n_needs/len(df):.1f}%)")
    
    if n_needs == 0:
        logger.info("\n✓ All signals already have ML scores. Nothing to do.")
        return
    
    logger.info(f"\nApplying ML inference to {n_needs} signals...")
    
    # Extract signals that need inference
    to_infer = df[needs_inference].copy()
    
    # Run batch inference
    try:
        logger.info("Running batch ML inference...")
        inferred = infer_batch(to_infer)
        
        if inferred is not None:
            # Update the original dataframe with inferred columns
            df.loc[needs_inference, "ml_rank_score"] = inferred["ml_rank_score"]
            df.loc[needs_inference, "ml_confidence_score"] = inferred["ml_confidence_score"]
            df.loc[needs_inference, "ml_rank_bucket"] = inferred["ml_rank_bucket"]
            df.loc[needs_inference, "ml_confidence_bucket"] = inferred["ml_confidence_bucket"]
            df.loc[needs_inference, "ml_agreement_with_engine"] = inferred["ml_agreement_with_engine"]
            
            # Count how many got populated
            newly_populated = (
                df.loc[needs_inference, "ml_rank_score"].notna() |
                df.loc[needs_inference, "ml_confidence_score"].notna()
            ).sum()
            
            logger.info(f"✓ Batch inference completed")
            logger.info(f"  Newly scored signals: {newly_populated}/{n_needs}")
            
    except Exception as e:
        logger.exception(f"Batch inference failed: {e}")
        logger.info("Falling back to single-signal inference...")
        
        newly_populated = 0
        for idx, (i, row) in enumerate(to_infer.iterrows()):
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx+1}/{n_needs}")
            
            try:
                result = infer_single(row.to_dict())
                if result.ml_rank_score is not None or result.ml_confidence_score is not None:
                    df.loc[i, "ml_rank_score"] = result.ml_rank_score
                    df.loc[i, "ml_confidence_score"] = result.ml_confidence_score
                    df.loc[i, "ml_rank_bucket"] = result.ml_rank_bucket
                    df.loc[i, "ml_confidence_bucket"] = result.ml_confidence_bucket
                    df.loc[i, "ml_agreement_with_engine"] = result.ml_agreement_with_engine
                    newly_populated += 1
            except Exception as e2:
                logger.debug(f"Single-signal inference failed for signal_id={row.get('signal_id')}: {e2}")
        
        logger.info(f"✓ Single-signal inference completed")
        logger.info(f"  Newly scored signals: {newly_populated}/{n_needs}")
    
    # Final statistics
    has_rank_after = df["ml_rank_score"].notna().sum()
    has_conf_after = df["ml_confidence_score"].notna().sum()
    has_both_after = (df["ml_rank_score"].notna() & df["ml_confidence_score"].notna()).sum()
    
    logger.info(f"\nFinal ML Score Population:")
    logger.info(f"  Signals with ml_rank_score: {has_rank_after} ({100*has_rank_after/len(df):.1f}%)")
    logger.info(f"  Signals with ml_confidence_score: {has_conf_after} ({100*has_conf_after/len(df):.1f}%)")
    logger.info(f"  Signals with BOTH: {has_both_after} ({100*has_both_after/len(df):.1f}%)")
    
    # Save updated dataset
    logger.info(f"\nSaving updated cumulative dataset...")
    try:
        write_signals_dataset(df, CUMULATIVE_DATASET_PATH)
        logger.info(f"✓ Successfully saved to {CUMULATIVE_DATASET_PATH}")
    except Exception as e:
        logger.exception(f"Failed to save: {e}")
        sys.exit(1)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ML INFERENCE BACKFILL COMPLETE")
    logger.info(f"  Improvement: {n_have} → {has_both_after} signals with both ML scores")
    logger.info(f"  Coverage: {100*has_both_after/len(df):.1f}%")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    try:
        apply_ml_inference_to_cumulative()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
