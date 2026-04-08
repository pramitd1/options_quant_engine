"""
ML Signal Research Pipeline — Full Quantitative Evaluation
============================================================
RESEARCH ONLY — Does NOT modify any production code.

This script implements Sections 1–12 of the ML research framework:
  1.  Feature sanity & cleaning
  2.  Target redesign (all-signal targets from raw spots)
  3.  Dataset expansion (7,379 signals vs 2,691 trades)
  4.  Model benchmarking (4 models × multiple targets)
  5.  Time-series validation (expanding window)
  6.  Holdout testing (train 2016-2023 / test 2024-2025)
  7.  Ranking power analysis (quintile analysis)
  8.  Calibration analysis
  9.  Feature importance (permutation + coefficient-based)
  10. Shadow mode output schema
  11. Full research report generation
  12. Pass/fail evaluation against success criteria

Outputs go to: research/ml_research/

Author: Quantitative Research Pipeline
Date: 2026-03-18
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from models.expanded_feature_builder import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
    validate_no_post_signal_labels_in_features,
)

# ── Paths ───────────────────────────────────────────────────────────
DATASET_PATH = PROJECT_ROOT / "research" / "signal_evaluation" / "backtest_signals_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "research" / "ml_research"
RANDOM_STATE = 42

# ── Logging ─────────────────────────────────────────────────────────
_report_lines: list[str] = []


def log(msg: str = ""):
    print(msg)
    _report_lines.append(msg)


def section_header(num: int, title: str):
    log(f"\n{'='*90}")
    log(f"  SECTION {num}: {title}")
    log(f"{'='*90}")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — FEATURE SANITY & CLEANING                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

def feature_sanity_analysis(X: np.ndarray, names: list[str]) -> dict:
    """Analyse all features for variance, detect dead/constant features."""
    section_header(1, "FEATURE SANITY & CLEANING")

    results = {
        "original_count": len(names),
        "features": [],
        "dropped": [],
        "kept": [],
        "drop_reasons": {},
    }

    near_zero_threshold = 0.01  # features with std < 1% of mean(abs) are near-zero

    log(f"\n  Scanning {len(names)} features across {len(X)} samples...\n")
    log(f"  {'#':>3} {'Feature':<35} {'Mean':>10} {'Std':>10} {'NZ%':>8} {'Unique':>8} {'Verdict':<20}")
    log(f"  {'─'*3} {'─'*35} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*20}")

    keep_mask = np.ones(len(names), dtype=bool)

    for i, name in enumerate(names):
        col = X[:, i]
        mean_val = col.mean()
        std_val = col.std()
        nz_pct = np.count_nonzero(col) / len(col)
        n_unique = len(np.unique(col))

        verdict = "KEEP"
        reason = None

        if std_val == 0.0:
            verdict = "DROP: zero-variance"
            reason = "zero_variance"
            keep_mask[i] = False
        elif n_unique <= 2 and nz_pct < 0.05:
            verdict = "DROP: near-zero var"
            reason = "near_zero_variance"
            keep_mask[i] = False
        elif std_val < near_zero_threshold and abs(mean_val) > 0:
            # Very low relative variance
            rel_var = std_val / (abs(mean_val) + 1e-10)
            if rel_var < 0.001:
                verdict = "DROP: constant-like"
                reason = "constant_encoded"
                keep_mask[i] = False

        status = "✓" if keep_mask[i] else "✗"
        log(f"  {status} {i:>2} {name:<35} {mean_val:>10.4f} {std_val:>10.4f} {nz_pct:>7.1%} {n_unique:>8} {verdict}")

        feat_info = {
            "name": name,
            "index": i,
            "mean": round(float(mean_val), 6),
            "std": round(float(std_val), 6),
            "nz_pct": round(float(nz_pct), 4),
            "n_unique": int(n_unique),
            "verdict": verdict,
        }
        results["features"].append(feat_info)

        if not keep_mask[i]:
            results["dropped"].append(name)
            results["drop_reasons"][name] = reason
        else:
            results["kept"].append(name)

    X_clean = X[:, keep_mask]
    kept_names = [n for n, k in zip(names, keep_mask) if k]

    leaked = validate_no_post_signal_labels_in_features(kept_names)
    if leaked:
        raise RuntimeError(f"Feature leakage detected in active feature names: {leaked}")

    log(f"\n  Summary:")
    log(f"    Original features: {len(names)}")
    log(f"    Dropped: {len(results['dropped'])} — {results['dropped']}")
    log(f"    Kept: {len(kept_names)}")
    log(f"    Drop reasons: {dict(results['drop_reasons'])}")

    results["kept_count"] = len(kept_names)
    results["dropped_count"] = len(results["dropped"])

    return X_clean, kept_names, keep_mask, results


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 & 3 — TARGET REDESIGN & DATASET EXPANSION            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def build_expanded_dataset(df: pd.DataFrame) -> dict:
    """Build multi-target dataset using ALL signals, not just trades.

    Targets constructed from raw spot data:
      - large_move_60m:    |return_60m| > median(|return_60m|)
      - large_move_session: |return_session| > median
      - up_move_60m:       spot_60m > spot_at_signal
      - up_move_session:   session_close > spot_at_signal
      - up_move_next_close: next_close > spot_at_signal
    """
    section_header(2, "TARGET REDESIGN & DATASET EXPANSION")

    # Extract features for ALL rows
    log("\n  Extracting features for all 7,404 signals...")
    records = df.to_dict("records")
    X_all = np.vstack([extract_features(r) for r in records])
    timestamps = pd.to_datetime(df["signal_timestamp"])
    years = np.array([t.year if pd.notna(t) else 0 for t in timestamps])

    log(f"  Feature matrix shape: {X_all.shape}")

    # Build targets from raw spot data
    spot_signal = df["spot_at_signal"].values
    targets = {}

    # 1. Raw returns in bps (direction-free)
    for horizon, col in [("60m", "spot_60m"), ("120m", "spot_120m"),
                         ("session", "spot_session_close"), ("next_close", "spot_next_close"),
                         ("1d", "spot_1d")]:
        spot_h = df[col].values
        valid = np.isfinite(spot_signal) & np.isfinite(spot_h) & (spot_signal > 0)
        raw_ret = np.full(len(df), np.nan)
        raw_ret[valid] = (spot_h[valid] - spot_signal[valid]) / spot_signal[valid] * 10000
        targets[f"raw_return_{horizon}_bps"] = raw_ret

    # 2. Classification targets (binary)
    for horizon in ["60m", "120m", "session", "next_close", "1d"]:
        raw = targets[f"raw_return_{horizon}_bps"]
        valid = np.isfinite(raw)

        # Large move: |return| > median of |return| for valid samples
        abs_returns = np.abs(raw[valid])
        median_move = np.median(abs_returns)

        large_move = np.full(len(df), np.nan)
        large_move[valid] = (np.abs(raw[valid]) > median_move).astype(float)
        targets[f"large_move_{horizon}"] = large_move

        # Directional: up vs down
        up_move = np.full(len(df), np.nan)
        up_move[valid] = (raw[valid] > 0).astype(float)
        targets[f"up_move_{horizon}"] = up_move

    # 3. For signals WITH direction, signed correctness
    dir_numeric = df["direction_numeric"].values
    for horizon in ["60m", "120m", "session", "next_close", "1d"]:
        raw = targets[f"raw_return_{horizon}_bps"]
        has_dir = np.abs(dir_numeric) > 0
        valid = np.isfinite(raw) & has_dir

        correct = np.full(len(df), np.nan)
        # correct = 1 if sign(direction) == sign(return)
        correct[valid] = (np.sign(dir_numeric[valid]) == np.sign(raw[valid])).astype(float)
        targets[f"correct_{horizon}_all"] = correct

    # Report coverage
    log(f"\n  Target coverage (out of {len(df)} signals):")
    log(f"  {'Target':<30} {'Non-Null':>10} {'%':>8} {'Pos Rate':>10}")
    log(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*10}")
    target_meta = {}
    for tname, tvals in sorted(targets.items()):
        valid_mask = np.isfinite(tvals)
        n_valid = valid_mask.sum()
        pos_rate = tvals[valid_mask].mean() if n_valid > 0 else 0
        log(f"  {tname:<30} {n_valid:>10} {n_valid/len(df):>7.1%} {pos_rate:>10.3f}")
        target_meta[tname] = {"n_valid": int(n_valid), "pos_rate": round(float(pos_rate), 4)}

    # Also keep original targets for comparison
    for col in ["target_1d", "target_5d", "target_at_expiry"]:
        vals = df[col].values.astype(float) if col in df.columns else np.full(len(df), np.nan)
        targets[col] = vals
        valid_mask = np.isfinite(vals)
        n_valid = valid_mask.sum()
        pos_rate = vals[valid_mask].mean() if n_valid > 0 else 0
        log(f"  {col:<30} {n_valid:>10} {n_valid/len(df):>7.1%} {pos_rate:>10.3f}")
        target_meta[col] = {"n_valid": int(n_valid), "pos_rate": round(float(pos_rate), 4)}

    log(f"\n  Dataset expansion:")
    log(f"    Previous (trade-only target_1d): 2,691 samples")
    log(f"    New (all-signal large_move_60m): {int(np.isfinite(targets['large_move_60m']).sum())} samples")
    log(f"    Increase: {np.isfinite(targets['large_move_60m']).sum() / 2691:.1f}x")

    return {
        "X": X_all,
        "targets": targets,
        "target_meta": target_meta,
        "timestamps": timestamps,
        "years": years,
        "direction_numeric": dir_numeric,
    }


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — MODEL BENCHMARKING                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝

MODEL_CONFIGS = {
    "LogReg_L2": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, penalty="l2", solver="lbfgs",
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
    "LogReg_ElasticNet": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, penalty="elasticnet", solver="saga",
            l1_ratio=0.5, max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
    "GBT_shallow": lambda: HistGradientBoostingClassifier(
        max_iter=150, max_depth=3, learning_rate=0.03,
        min_samples_leaf=40, max_leaf_nodes=8,
        l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, scoring="neg_log_loss",
        random_state=RANDOM_STATE, class_weight="balanced",
    ),
    "RF_shallow": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=30,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ),
}


def train_and_evaluate(model, X_train, y_train, X_test, y_test, label=""):
    """Train a model and compute metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {}
    metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
    metrics["precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
    metrics["recall"] = round(recall_score(y_test, y_pred, zero_division=0), 4)
    metrics["f1"] = round(f1_score(y_test, y_pred, zero_division=0), 4)
    try:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
    except ValueError:
        metrics["roc_auc"] = None
    metrics["log_loss"] = round(log_loss(y_test, y_prob), 4)
    metrics["brier"] = round(brier_score_loss(y_test, y_prob), 4)
    metrics["n_samples"] = int(len(y_test))
    metrics["pos_rate"] = round(float(y_test.mean()), 4)

    return model, y_prob, metrics


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — TIME-SERIES VALIDATION                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

def expanding_window_cv(X, y, years, model_factory, min_train_years=2):
    """Expanding-window temporal CV. Returns per-fold metrics and predictions."""
    unique_years = sorted(set(years[np.isfinite(y)]))
    folds = []

    for val_idx in range(min_train_years, len(unique_years)):
        val_year = unique_years[val_idx]
        train_years_set = set(unique_years[:val_idx])

        train_mask = np.array([yr in train_years_set for yr in years]) & np.isfinite(y)
        val_mask = (years == val_year) & np.isfinite(y)

        if train_mask.sum() < 50 or val_mask.sum() < 30:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]

        model = model_factory()
        model, y_prob, metrics = train_and_evaluate(model, X_tr, y_tr, X_va, y_va)
        metrics["train_years"] = sorted(train_years_set)
        metrics["val_year"] = int(val_year)
        metrics["train_n"] = int(len(y_tr))

        folds.append({
            "metrics": metrics,
            "y_true": y_va,
            "y_prob": y_prob,
            "val_year": int(val_year),
        })

    return folds


def run_cv_for_all_models(X, y, years, target_name):
    """Run expanding-window CV for all model configs on one target."""
    results = {}
    for mname, mfactory in MODEL_CONFIGS.items():
        folds = expanding_window_cv(X, y, years, mfactory)
        if not folds:
            continue

        aucs = [f["metrics"]["roc_auc"] for f in folds if f["metrics"]["roc_auc"] is not None]
        briers = [f["metrics"]["brier"] for f in folds]
        accs = [f["metrics"]["accuracy"] for f in folds]

        # Stability = 1 - coefficient of variation of AUC across folds
        auc_std = np.std(aucs) if len(aucs) > 1 else 0
        auc_mean = np.mean(aucs) if aucs else 0
        stability = round(1.0 - (auc_std / (auc_mean + 1e-10)), 4)

        results[mname] = {
            "folds": folds,
            "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "std_auc": round(float(auc_std), 4),
            "mean_brier": round(float(np.mean(briers)), 4),
            "mean_acc": round(float(np.mean(accs)), 4),
            "stability": stability,
            "n_folds": len(folds),
        }

    return results


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — HOLDOUT TESTING                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

def holdout_test(X, y, years, model_factory, train_end=2023, test_start=2024):
    """Train on years <= train_end, test on years >= test_start."""
    train_mask = (years <= train_end) & np.isfinite(y)
    test_mask = (years >= test_start) & np.isfinite(y)

    if train_mask.sum() < 50 or test_mask.sum() < 30:
        return None

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    model = model_factory()
    model, y_prob, test_metrics = train_and_evaluate(model, X_tr, y_tr, X_te, y_te)

    # Train metrics for overfit check
    y_tr_pred = model.predict(X_tr)
    y_tr_prob = model.predict_proba(X_tr)[:, 1]
    train_auc = roc_auc_score(y_tr, y_tr_prob) if len(np.unique(y_tr)) > 1 else 0

    return {
        "model": model,
        "y_prob": y_prob,
        "y_true": y_te,
        "test_metrics": test_metrics,
        "train_auc": round(float(train_auc), 4),
        "overfit_gap": round(float(train_auc - (test_metrics["roc_auc"] or 0)), 4),
        "train_n": int(train_mask.sum()),
        "test_n": int(test_mask.sum()),
    }


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — RANKING POWER ANALYSIS                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

def quintile_analysis(y_true, y_prob, raw_returns=None, n_buckets=5):
    """Bucket predictions into quintiles, compute hit rate & avg return per bucket."""
    if len(y_true) < n_buckets * 5:
        return None

    # Use np.percentile to define bucket edges
    edges = np.percentile(y_prob, np.linspace(0, 100, n_buckets + 1))
    # Ensure monotonic edges
    edges = np.unique(edges)
    if len(edges) < 3:
        # Not enough unique values
        return None
    actual_buckets = len(edges) - 1

    buckets = []
    for i in range(actual_buckets):
        lo, hi = edges[i], edges[i + 1]
        if i == actual_buckets - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            continue

        bucket_info = {
            "bucket": i + 1,
            "range": f"[{lo:.3f}, {hi:.3f})",
            "n": int(mask.sum()),
            "hit_rate": round(float(y_true[mask].mean()), 4),
            "avg_prob": round(float(y_prob[mask].mean()), 4),
        }
        if raw_returns is not None:
            valid = mask & np.isfinite(raw_returns)
            if valid.sum() > 0:
                bucket_info["avg_return_bps"] = round(float(raw_returns[valid].mean()), 2)
                bucket_info["median_return_bps"] = round(float(np.median(raw_returns[valid])), 2)

        buckets.append(bucket_info)

    if len(buckets) < 2:
        return None

    # Check monotonicity: bottom quintile < top quintile
    bottom_hit = buckets[0]["hit_rate"]
    top_hit = buckets[-1]["hit_rate"]
    monotonic = top_hit > bottom_hit
    spread = round(top_hit - bottom_hit, 4)

    return {
        "buckets": buckets,
        "top_quintile_hit": top_hit,
        "bottom_quintile_hit": bottom_hit,
        "spread": spread,
        "monotonic": monotonic,
        "n_buckets": len(buckets),
    }


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — CALIBRATION ANALYSIS                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

def calibration_analysis(y_true, y_prob, n_bins=10):
    """Compute calibration curve, Brier score, detect over/under confidence."""
    if len(y_true) < n_bins * 5:
        return None

    try:
        fraction_positive, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform",
        )
    except ValueError:
        return None

    brier = brier_score_loss(y_true, y_prob)

    # Calibration error per bin
    bins = []
    overconfident_bins = 0
    underconfident_bins = 0
    for fp, mp in zip(fraction_positive, mean_predicted):
        error = mp - fp  # positive = overconfident
        bins.append({
            "predicted": round(float(mp), 4),
            "actual": round(float(fp), 4),
            "error": round(float(error), 4),
        })
        if error > 0.05:
            overconfident_bins += 1
        elif error < -0.05:
            underconfident_bins += 1

    # Expected calibration error
    ece = np.mean(np.abs(fraction_positive - mean_predicted))

    return {
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "bins": bins,
        "overconfident_bins": overconfident_bins,
        "underconfident_bins": underconfident_bins,
        "diagnosis": (
            "overconfident" if overconfident_bins > underconfident_bins
            else "underconfident" if underconfident_bins > overconfident_bins
            else "well_calibrated"
        ),
    }


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — FEATURE IMPORTANCE                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝

def compute_feature_importance(model, X_test, y_test, feature_names):
    """Compute permutation importance + model-native importance."""
    results = {"feature_names": feature_names}

    # Model-native importance
    inner = model[-1] if isinstance(model, Pipeline) else model
    native_imp = getattr(inner, "feature_importances_", None)
    if native_imp is None and hasattr(inner, "coef_"):
        native_imp = np.abs(inner.coef_.ravel())

    if native_imp is not None:
        results["native_importance"] = {
            n: round(float(v), 6) for n, v in zip(feature_names, native_imp)
        }

    # Permutation importance (5 repeats for stability)
    try:
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=5,
            scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
        )
        results["permutation_importance"] = {
            n: {
                "mean": round(float(m), 6),
                "std": round(float(s), 6),
            }
            for n, m, s in zip(feature_names, perm.importances_mean, perm.importances_std)
        }
    except Exception as e:
        results["permutation_importance_error"] = str(e)

    return results


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — SHADOW MODE OUTPUT                                ║
# ╚═══════════════════════════════════════════════════════════════════╝

def generate_shadow_predictions(model, X, feature_names, direction_numeric):
    """Generate shadow-mode predictions for ALL signals."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # ML confidence: distance from 0.5 (higher = more confident)
    ml_confidence = np.abs(y_prob - 0.5) * 2  # Scale to 0-1

    # ML signal strength: product of probability and confidence
    ml_strength = y_prob * ml_confidence

    # ML agreement with engine direction
    ml_direction = np.where(y_prob > 0.5, 1, -1)
    engine_dir = np.sign(direction_numeric)
    agreement = np.where(
        engine_dir == 0, "NO_ENGINE_SIGNAL",
        np.where(ml_direction == engine_dir, "AGREE", "DISAGREE"),
    )

    return {
        "ml_direction_probability": np.round(y_prob, 4),
        "ml_confidence_score": np.round(ml_confidence, 4),
        "ml_signal_strength": np.round(ml_strength, 4),
        "ml_agreement_with_engine": agreement,
    }


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MAIN RESEARCH PIPELINE                                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

def main():
    t0 = time.time()

    log("=" * 90)
    log("  ML SIGNAL RESEARCH PIPELINE — QUANTITATIVE EVALUATION")
    log(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Research mode only — NO production code is modified")
    log("=" * 90)

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    log(f"\n  Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    log(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")

    # ── Section 1: Feature sanity ──
    records = df.to_dict("records")
    X_raw = np.vstack([extract_features(r) for r in records])
    X_clean, active_names, keep_mask, feature_results = feature_sanity_analysis(X_raw, list(FEATURE_NAMES))

    # ── Section 2 & 3: Target redesign & dataset expansion ──
    dataset = build_expanded_dataset(df)
    X_all = dataset["X"][:, keep_mask]  # Apply same feature mask
    years = dataset["years"]
    direction_numeric = dataset["direction_numeric"]

    # ── Define research targets ──
    # Primary targets: use ALL signals (not just trades)
    research_targets = {
        # All-signal targets (7,379 samples)
        "large_move_60m":    dataset["targets"]["large_move_60m"],
        "up_move_next_close": dataset["targets"]["up_move_next_close"],
        "large_move_session": dataset["targets"]["large_move_session"],
        # Directional targets (2,708 signals with direction)
        "correct_60m_all":   dataset["targets"]["correct_60m_all"],
        "correct_1d_all":    dataset["targets"]["correct_1d_all"],
        # Original target for comparison
        "target_1d":         dataset["targets"]["target_1d"],
    }

    # ── Section 4 & 5: Model benchmarking with CV ──
    section_header(4, "MODEL BENCHMARKING (with expanding-window CV)")

    all_cv_results = {}
    for tname, tvals in research_targets.items():
        valid = np.isfinite(tvals)
        n_valid = valid.sum()
        if n_valid < 200:
            log(f"\n  Skipping {tname}: only {n_valid} valid samples")
            continue

        pos_rate = tvals[valid].mean()
        log(f"\n  ─── Target: {tname} ({n_valid} samples, {pos_rate:.1%} positive) ───")

        cv_results = run_cv_for_all_models(X_all, tvals, years, tname)
        all_cv_results[tname] = cv_results

        # Print compact summary
        log(f"\n  {'Model':<25} {'CV AUC':>10} {'±Std':>8} {'Brier':>8} {'Acc':>8} {'Stability':>10}")
        log(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
        for mname in sorted(cv_results, key=lambda k: cv_results[k]["mean_auc"] or 0, reverse=True):
            r = cv_results[mname]
            log(f"  {mname:<25} {r['mean_auc'] or 0:>10.4f} {r['std_auc']:>8.4f} "
                f"{r['mean_brier']:>8.4f} {r['mean_acc']:>8.4f} {r['stability']:>10.4f}")

    # ── Section 6: Holdout testing ──
    section_header(6, "HOLDOUT TESTING (Train 2016-2023 / Test 2024-2025)")

    holdout_results = {}
    best_overall = {"auc": 0, "model_name": None, "target": None, "result": None}

    for tname, tvals in research_targets.items():
        valid = np.isfinite(tvals)
        if valid.sum() < 200:
            continue

        log(f"\n  ─── Target: {tname} ───")
        log(f"  {'Model':<25} {'Train AUC':>10} {'Test AUC':>10} {'Gap':>8} {'Brier':>8} {'Acc':>8}")
        log(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

        target_holdout = {}
        for mname, mfactory in MODEL_CONFIGS.items():
            result = holdout_test(X_all, tvals, years, mfactory)
            if result is None:
                continue

            te = result["test_metrics"]
            log(f"  {mname:<25} {result['train_auc']:>10.4f} {te['roc_auc'] or 0:>10.4f} "
                f"{result['overfit_gap']:>8.4f} {te['brier']:>8.4f} {te['accuracy']:>8.4f}")

            target_holdout[mname] = result

            # Track best overall
            test_auc = te["roc_auc"] or 0
            if test_auc > best_overall["auc"]:
                best_overall = {
                    "auc": test_auc,
                    "model_name": mname,
                    "target": tname,
                    "result": result,
                }

        holdout_results[tname] = target_holdout

    log(f"\n  ★ Best overall: {best_overall['model_name']} on {best_overall['target']} "
        f"— Test AUC = {best_overall['auc']:.4f}")

    # ── Section 7: Ranking power analysis ──
    section_header(7, "RANKING POWER ANALYSIS (Quintile Breakdown)")

    ranking_results = {}
    for tname, target_holdout in holdout_results.items():
        for mname, result in target_holdout.items():
            y_true = result["y_true"]
            y_prob = result["y_prob"]

            # Get raw returns for the test set for this target
            raw_ret_col = None
            if "60m" in tname:
                raw_ret_col = "raw_return_60m_bps"
            elif "session" in tname:
                raw_ret_col = "raw_return_session_bps"
            elif "next_close" in tname or "1d" in tname:
                raw_ret_col = "raw_return_next_close_bps"

            raw_rets = None
            if raw_ret_col and raw_ret_col in dataset["targets"]:
                test_mask = (years >= 2024) & np.isfinite(dataset["targets"].get(tname, np.array([])))
                if len(test_mask) == len(dataset["targets"].get(raw_ret_col, [])):
                    raw_rets_all = dataset["targets"][raw_ret_col]
                    raw_rets = raw_rets_all[test_mask]

            qa = quintile_analysis(y_true, y_prob, raw_rets)
            key = f"{mname}__{tname}"
            ranking_results[key] = qa

            if qa:
                log(f"\n  {mname} on {tname}:")
                log(f"  {'Bucket':>8} {'Range':<20} {'N':>6} {'Hit Rate':>10} {'Avg Prob':>10} {'Avg Ret':>10}")
                log(f"  {'─'*8} {'─'*20} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
                for b in qa["buckets"]:
                    ret_str = f"{b.get('avg_return_bps', 'N/A'):>10}" if isinstance(b.get('avg_return_bps'), (int, float)) else f"{'N/A':>10}"
                    log(f"  {b['bucket']:>8} {b['range']:<20} {b['n']:>6} {b['hit_rate']:>10.4f} {b['avg_prob']:>10.4f} {ret_str}")
                log(f"  Spread (top-bottom): {qa['spread']:.4f} | Monotonic: {qa['monotonic']}")

    # ── Section 8: Calibration ──
    section_header(8, "CALIBRATION ANALYSIS")

    calibration_results = {}
    for tname, target_holdout in holdout_results.items():
        for mname, result in target_holdout.items():
            cal = calibration_analysis(result["y_true"], result["y_prob"])
            key = f"{mname}__{tname}"
            calibration_results[key] = cal

            if cal:
                log(f"\n  {mname} on {tname}:")
                log(f"    Brier: {cal['brier_score']:.4f} | ECE: {cal['ece']:.4f} | Diagnosis: {cal['diagnosis']}")
                log(f"    Overconfident bins: {cal['overconfident_bins']} | Underconfident: {cal['underconfident_bins']}")

    # ── Section 9: Feature importance ──
    section_header(9, "FEATURE IMPORTANCE")

    importance_results = {}
    if best_overall["result"]:
        best_model = best_overall["result"]["model"]
        best_target = best_overall["target"]
        best_tvals = research_targets[best_target]

        test_mask = (years >= 2024) & np.isfinite(best_tvals)
        X_te = X_all[test_mask]
        y_te = best_tvals[test_mask]

        log(f"\n  Computing importance for best model: {best_overall['model_name']} on {best_target}")
        importance_results = compute_feature_importance(best_model, X_te, y_te, active_names)

        # Print permutation importance
        if "permutation_importance" in importance_results:
            perm = importance_results["permutation_importance"]
            ranked = sorted(perm.items(), key=lambda x: x[1]["mean"], reverse=True)
            log(f"\n  {'Rank':>4} {'Feature':<35} {'Perm Imp':>10} {'±Std':>8}")
            log(f"  {'─'*4} {'─'*35} {'─'*10} {'─'*8}")
            for rank, (name, vals) in enumerate(ranked, 1):
                bar = "█" * max(1, int(vals["mean"] * 500))
                log(f"  {rank:>4} {name:<35} {vals['mean']:>10.6f} {vals['std']:>8.6f}  {bar}")

        # Check stability across CV folds (use best target's CV results)
        if best_target in all_cv_results and best_overall["model_name"] in all_cv_results[best_target]:
            log(f"\n  Feature importance stability across CV folds:")
            cv_folds = all_cv_results[best_target][best_overall["model_name"]]["folds"]
            fold_importances = []
            for fold in cv_folds:
                # Retrain to get fold-specific importance
                val_year = fold["val_year"]
                train_mask_f = np.array([yr < val_year for yr in years]) & np.isfinite(best_tvals)
                val_mask_f = (years == val_year) & np.isfinite(best_tvals)
                if train_mask_f.sum() < 50 or val_mask_f.sum() < 20:
                    continue
                m_fold = MODEL_CONFIGS[best_overall["model_name"]]()
                m_fold.fit(X_all[train_mask_f], best_tvals[train_mask_f])
                inner = m_fold[-1] if isinstance(m_fold, Pipeline) else m_fold
                fi = getattr(inner, "feature_importances_", None)
                if fi is None and hasattr(inner, "coef_"):
                    fi = np.abs(inner.coef_.ravel())
                if fi is not None:
                    fold_importances.append(fi)

            if len(fold_importances) >= 3:
                fi_matrix = np.vstack(fold_importances)
                fi_mean = fi_matrix.mean(axis=0)
                fi_std = fi_matrix.std(axis=0)
                ranked_stability = sorted(
                    zip(active_names, fi_mean, fi_std),
                    key=lambda x: x[1], reverse=True,
                )
                log(f"\n  {'Rank':>4} {'Feature':<35} {'Mean Imp':>10} {'Std':>8} {'Stable?':>8}")
                log(f"  {'─'*4} {'─'*35} {'─'*10} {'─'*8} {'─'*8}")
                for rank, (name, mean, std) in enumerate(ranked_stability[:15], 1):
                    stable = "YES" if std < mean * 0.5 else "NO"
                    log(f"  {rank:>4} {name:<35} {mean:>10.6f} {std:>8.6f} {stable:>8}")
                importance_results["fold_stability"] = {
                    n: {"mean": round(float(m), 6), "std": round(float(s), 6)}
                    for n, m, s in ranked_stability
                }

    # ── Section 10: Shadow mode schema ──
    section_header(10, "SHADOW MODE INTEGRATION")

    if best_overall["result"]:
        best_model = best_overall["result"]["model"]
        shadow = generate_shadow_predictions(best_model, X_all, active_names, direction_numeric)

        # Report shadow predictions summary
        probs = shadow["ml_direction_probability"]
        log(f"\n  Shadow predictions generated for all {len(probs)} signals:")
        log(f"    Probability: mean={probs.mean():.4f}, std={probs.std():.4f}, "
            f"min={probs.min():.4f}, max={probs.max():.4f}")
        log(f"    Confidence:  mean={shadow['ml_confidence_score'].mean():.4f}")

        agree_counts = pd.Series(shadow["ml_agreement_with_engine"]).value_counts()
        log(f"    Agreement with engine:")
        for k, v in agree_counts.items():
            log(f"      {k}: {v} ({v/len(probs):.1%})")

        # Save shadow predictions
        shadow_df = pd.DataFrame({
            "signal_id": df["signal_id"].values,
            "signal_timestamp": df["signal_timestamp"].values,
            "year": years,
            "ml_direction_probability": shadow["ml_direction_probability"],
            "ml_confidence_score": shadow["ml_confidence_score"],
            "ml_signal_strength": shadow["ml_signal_strength"],
            "ml_agreement_with_engine": shadow["ml_agreement_with_engine"],
        })
        shadow_path = OUTPUT_DIR / "shadow_predictions.csv"
        shadow_df.to_csv(shadow_path, index=False)
        log(f"    Saved to: {shadow_path}")

    # ── Section 11: Full research report ──
    section_header(11, "RESEARCH REPORT SUMMARY")

    # Model comparison table
    log(f"\n  ┌─{'─'*86}─┐")
    log(f"  │ {'Model':<20} {'Target':<22} {'CV AUC':>8} {'Test AUC':>9} {'Brier':>7} {'Stability':>10} {'Gap':>8} │")
    log(f"  ├─{'─'*86}─┤")

    report_rows = []
    for tname in research_targets:
        if tname not in all_cv_results or tname not in holdout_results:
            continue
        for mname in MODEL_CONFIGS:
            cv = all_cv_results.get(tname, {}).get(mname)
            ho = holdout_results.get(tname, {}).get(mname)
            if not cv or not ho:
                continue
            row = {
                "model": mname,
                "target": tname,
                "cv_auc": cv["mean_auc"],
                "test_auc": ho["test_metrics"]["roc_auc"],
                "brier": ho["test_metrics"]["brier"],
                "stability": cv["stability"],
                "overfit_gap": ho["overfit_gap"],
            }
            report_rows.append(row)
            log(f"  │ {mname:<20} {tname:<22} {row['cv_auc'] or 0:>8.4f} "
                f"{row['test_auc'] or 0:>9.4f} {row['brier']:>7.4f} "
                f"{row['stability']:>10.4f} {row['overfit_gap']:>8.4f} │")

    log(f"  └─{'─'*86}─┘")

    # Ranking power summary
    log(f"\n  Ranking Power (top vs bottom quintile):")
    log(f"  {'Model+Target':<50} {'Top Q':>8} {'Bot Q':>8} {'Spread':>8} {'Mono?':>6}")
    log(f"  {'─'*50} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    for key, qa in ranking_results.items():
        if qa:
            log(f"  {key:<50} {qa['top_quintile_hit']:>8.4f} {qa['bottom_quintile_hit']:>8.4f} "
                f"{qa['spread']:>8.4f} {'YES' if qa['monotonic'] else 'NO':>6}")

    # Calibration summary
    log(f"\n  Calibration Summary:")
    log(f"  {'Model+Target':<50} {'Brier':>8} {'ECE':>8} {'Diagnosis':<18}")
    log(f"  {'─'*50} {'─'*8} {'─'*8} {'─'*18}")
    for key, cal in calibration_results.items():
        if cal:
            log(f"  {key:<50} {cal['brier_score']:>8.4f} {cal['ece']:>8.4f} {cal['diagnosis']:<18}")

    # ── Section 12: Success criteria evaluation ──
    section_header(12, "SUCCESS CRITERIA EVALUATION")

    criteria = {
        "test_auc_above_055": False,
        "top_vs_bottom_significant": False,
        "calibration_stable": False,
        "performance_consistent_across_years": False,
    }

    # Check 1: Test AUC > 0.55
    if best_overall["auc"] > 0.55:
        criteria["test_auc_above_055"] = True
        log(f"  ✓ Test AUC: {best_overall['auc']:.4f} > 0.55")
    else:
        log(f"  ✗ Test AUC: {best_overall['auc']:.4f} ≤ 0.55 (FAIL)")

    # Check 2: Top quintile outperforms bottom
    best_key = f"{best_overall['model_name']}__{best_overall['target']}"
    best_qa = ranking_results.get(best_key)
    if best_qa and best_qa["spread"] > 0.05:
        criteria["top_vs_bottom_significant"] = True
        log(f"  ✓ Ranking spread: {best_qa['spread']:.4f} > 0.05")
    elif best_qa:
        log(f"  ✗ Ranking spread: {best_qa['spread']:.4f} ≤ 0.05 (FAIL)")
    else:
        log(f"  ✗ Ranking analysis unavailable (FAIL)")

    # Check 3: Calibration stable
    best_cal = calibration_results.get(best_key)
    if best_cal and best_cal["ece"] < 0.10:
        criteria["calibration_stable"] = True
        log(f"  ✓ Calibration ECE: {best_cal['ece']:.4f} < 0.10")
    elif best_cal:
        log(f"  ✗ Calibration ECE: {best_cal['ece']:.4f} ≥ 0.10 (FAIL)")
    else:
        log(f"  ✗ Calibration unavailable (FAIL)")

    # Check 4: Performance consistent (stability > 0.90)
    if best_overall["target"] in all_cv_results and best_overall["model_name"] in all_cv_results[best_overall["target"]]:
        stab = all_cv_results[best_overall["target"]][best_overall["model_name"]]["stability"]
        if stab > 0.90:
            criteria["performance_consistent_across_years"] = True
            log(f"  ✓ Stability: {stab:.4f} > 0.90")
        else:
            log(f"  ✗ Stability: {stab:.4f} ≤ 0.90 (FAIL)")

    passed = sum(criteria.values())
    total = len(criteria)
    all_pass = all(criteria.values())

    log(f"\n  {'='*60}")
    log(f"  FINAL VERDICT: {passed}/{total} criteria passed")
    if all_pass:
        log(f"  ★ RECOMMENDATION: READY for shadow production trial")
        log(f"    Deploy in shadow mode for 30 days before live activation.")
    elif passed >= 2:
        log(f"  ◆ RECOMMENDATION: PROMISING but NOT READY")
        log(f"    Continue research. Consider feature engineering or")
        log(f"    alternative target definitions.")
    else:
        log(f"  ✗ RECOMMENDATION: NOT READY for production")
        log(f"    ML adds insufficient value over heuristic baseline.")
        log(f"    Keep in research mode only.")
    log(f"  {'='*60}")

    # ── Save all artefacts ──
    elapsed = time.time() - t0

    # Save full report
    report_path = OUTPUT_DIR / "research_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(_report_lines))

    # Save structured results as JSON
    # Filter out numpy arrays and models for serialization
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    json_results = {
        "run_date": datetime.now().isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "dataset": {
            "total_signals": len(df),
            "feature_analysis": feature_results,
            "target_meta": dataset["target_meta"],
        },
        "cv_summary": {
            tname: {
                mname: {k: v for k, v in res.items() if k != "folds"}
                for mname, res in cv_results.items()
            }
            for tname, cv_results in all_cv_results.items()
        },
        "holdout_summary": {
            tname: {
                mname: {
                    "test_metrics": res["test_metrics"],
                    "train_auc": res["train_auc"],
                    "overfit_gap": res["overfit_gap"],
                }
                for mname, res in target_holdout.items()
            }
            for tname, target_holdout in holdout_results.items()
        },
        "ranking_power": {
            k: v for k, v in ranking_results.items() if v is not None
        },
        "calibration": {
            k: v for k, v in calibration_results.items() if v is not None
        },
        "feature_importance": importance_results,
        "best_model": {
            "model_name": best_overall["model_name"],
            "target": best_overall["target"],
            "test_auc": best_overall["auc"],
        },
        "success_criteria": criteria,
        "recommendation": "READY" if all_pass else ("PROMISING" if passed >= 2 else "NOT_READY"),
    }

    json_path = OUTPUT_DIR / "research_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=_serialize)

    log(f"\n  Outputs saved to {OUTPUT_DIR}/")
    log(f"    research_report.txt  — Full text report")
    log(f"    research_results.json — Structured results")
    log(f"    shadow_predictions.csv — Shadow mode predictions")
    log(f"  Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
