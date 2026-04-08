"""
Build Model Registry — Serialize All Research Models for Production Switching
=============================================================================
RESEARCH SCRIPT — creates serialized model artifacts in models_store/registry/.

This script trains every model configuration discovered during our ML research
and saves each one as a fully self-contained artifact that production code can
load instantly.  The script also writes a central manifest.json that maps model
names to their artifact paths, hyperparameters, feature lists, performance
metrics, and success-criteria verdicts.

After running this script, switching production to any research model is:
  1. Set ACTIVE_MODEL = "GBT_shallow_v1" in config/settings.py  (or env var)
  2. Restart the engine.

Models serialized (target: correct_60m_all):
  - LogReg_L2_v1           — L2 logistic regression, uncalibrated
  - LogReg_ElasticNet_v1   — ElasticNet logistic regression, uncalibrated
  - GBT_shallow_v1         — Gradient boosted trees (depth=3), uncalibrated
  - GBT_shallow_platt_v1   — Same GBT with Platt scaling calibration
  - RF_shallow_v1          — Random forest (depth=3), uncalibrated

Outputs:
  models_store/registry/manifest.json
  models_store/registry/<model_name>/model.joblib
  models_store/registry/<model_name>/meta.json

Author: Quantitative Research Pipeline
Date: 2026-03-18
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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

from models.expanded_feature_builder import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
    validate_no_post_signal_labels_in_features,
)
from models.trained_predictor import TrainedMovePredictor

# ── Paths ───────────────────────────────────────────────────────────
DATASET_PATH = PROJECT_ROOT / "research" / "signal_evaluation" / "backtest_signals_dataset.parquet"
REGISTRY_DIR = PROJECT_ROOT / "models_store" / "registry"
RANDOM_STATE = 42

# ── Success criteria thresholds (from research framework) ────────
SUCCESS_AUC = 0.55
SUCCESS_ECE = 0.10
SUCCESS_RANKING_SPREAD = 0.05
SUCCESS_STABILITY = 0.90


# ── Model configurations from research ──────────────────────────────
MODEL_CONFIGS = {
    "LogReg_L2_v1": {
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, penalty="l2", solver="lbfgs",
                max_iter=2000, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "hyperparameters": {
            "C": 0.1, "penalty": "l2", "solver": "lbfgs",
            "max_iter": 2000, "class_weight": "balanced",
            "scaler": "StandardScaler",
        },
        "model_class": "Pipeline(StandardScaler + LogisticRegression)",
        "calibration": "none",
    },
    "LogReg_ElasticNet_v1": {
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, penalty="elasticnet", solver="saga",
                l1_ratio=0.5, max_iter=2000, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "hyperparameters": {
            "C": 0.1, "penalty": "elasticnet", "solver": "saga",
            "l1_ratio": 0.5, "max_iter": 2000, "class_weight": "balanced",
            "scaler": "StandardScaler",
        },
        "model_class": "Pipeline(StandardScaler + LogisticRegression)",
        "calibration": "none",
    },
    "GBT_shallow_v1": {
        "factory": lambda: HistGradientBoostingClassifier(
            max_iter=150, max_depth=3, learning_rate=0.03,
            min_samples_leaf=40, max_leaf_nodes=8,
            l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, scoring="neg_log_loss",
            random_state=RANDOM_STATE, class_weight="balanced",
        ),
        "hyperparameters": {
            "max_iter": 150, "max_depth": 3, "learning_rate": 0.03,
            "min_samples_leaf": 40, "max_leaf_nodes": 8,
            "l2_regularization": 5.0, "early_stopping": True,
            "validation_fraction": 0.15, "n_iter_no_change": 25,
            "class_weight": "balanced",
        },
        "model_class": "HistGradientBoostingClassifier",
        "calibration": "none",
    },
    "GBT_shallow_platt_v1": {
        # This is handled specially — base model trained first, then Platt
        "factory": None,  # built from GBT_shallow_v1 + CalibratedClassifierCV
        "hyperparameters": {
            "base_model": "GBT_shallow_v1",
            "calibration_method": "sigmoid",
            "calibration_split": "2023 (temporal holdout)",
        },
        "model_class": "CalibratedClassifierCV(HistGradientBoostingClassifier, sigmoid)",
        "calibration": "platt",
    },
    "RF_shallow_v1": {
        "factory": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=30,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "hyperparameters": {
            "n_estimators": 200, "max_depth": 3, "min_samples_leaf": 30,
            "class_weight": "balanced",
        },
        "model_class": "RandomForestClassifier",
        "calibration": "none",
    },
}


# ── Metrics helpers ─────────────────────────────────────────────────

def compute_metrics(y_true, y_prob):
    """Full metrics including ECE."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {}
    try:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    except ValueError:
        metrics["roc_auc"] = None
    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    metrics["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
    metrics["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
    metrics["f1"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
    metrics["brier"] = round(brier_score_loss(y_true, y_prob), 4)
    metrics["log_loss"] = round(log_loss(y_true, y_prob), 4)

    # ECE
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        metrics["ece"] = round(float(np.mean(np.abs(frac_pos - mean_pred))), 4)
    except ValueError:
        metrics["ece"] = None

    metrics["n_samples"] = int(len(y_true))
    metrics["pos_rate"] = round(float(y_true.mean()), 4)
    return metrics


def quintile_analysis(y_true, y_prob, n_buckets=5):
    """Hit rate by quintile to verify ranking power."""
    edges = np.unique(np.percentile(y_prob, np.linspace(0, 100, n_buckets + 1)))
    if len(edges) < 3:
        return None
    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == len(edges) - 2 else (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        buckets.append({
            "bucket": i + 1,
            "n": int(mask.sum()),
            "hit_rate": round(float(y_true[mask].mean()), 4),
            "avg_prob": round(float(y_prob[mask].mean()), 4),
        })
    if len(buckets) < 2:
        return None
    return {
        "buckets": buckets,
        "top_hit": buckets[-1]["hit_rate"],
        "bottom_hit": buckets[0]["hit_rate"],
        "spread": round(buckets[-1]["hit_rate"] - buckets[0]["hit_rate"], 4),
        "monotonic": buckets[-1]["hit_rate"] > buckets[0]["hit_rate"],
    }


def evaluate_success_criteria(test_metrics, quintile_result, cv_stability):
    """Check pass/fail for each research success criterion."""
    criteria = {}

    # 1. AUC > 0.55
    auc = test_metrics.get("roc_auc")
    criteria["auc"] = {"threshold": SUCCESS_AUC, "value": auc, "pass": auc is not None and auc > SUCCESS_AUC}

    # 2. Calibration ECE < 0.10
    ece = test_metrics.get("ece")
    criteria["calibration"] = {"threshold": SUCCESS_ECE, "value": ece, "pass": ece is not None and ece < SUCCESS_ECE}

    # 3. Ranking spread > 0.05 and monotonic
    if quintile_result:
        spread = quintile_result["spread"]
        mono = quintile_result["monotonic"]
        criteria["ranking"] = {
            "threshold": SUCCESS_RANKING_SPREAD,
            "value": spread,
            "monotonic": mono,
            "pass": spread > SUCCESS_RANKING_SPREAD and mono,
        }
    else:
        criteria["ranking"] = {"pass": False, "reason": "insufficient data"}

    # 4. Stability > 0.90
    criteria["stability"] = {
        "threshold": SUCCESS_STABILITY,
        "value": cv_stability,
        "pass": cv_stability is not None and cv_stability > SUCCESS_STABILITY,
    }

    criteria["all_pass"] = all(c["pass"] for c in criteria.values())
    criteria["pass_count"] = sum(1 for c in criteria.values() if isinstance(c, dict) and c.get("pass"))
    criteria["total_count"] = sum(1 for c in criteria.values() if isinstance(c, dict) and "pass" in c)

    return criteria


# ── Expanding-window CV for stability measurement ──────────────────

def compute_cv_stability(X, y, years, model_factory, min_train_years=2):
    """Run expanding-window temporal CV and return mean AUC, std, stability."""
    unique_years = sorted(set(years[np.isfinite(y)]))
    aucs = []

    for val_idx in range(min_train_years, len(unique_years)):
        val_year = unique_years[val_idx]
        train_years_set = set(unique_years[:val_idx])

        train_mask = np.array([yr in train_years_set for yr in years]) & np.isfinite(y)
        val_mask = (years == val_year) & np.isfinite(y)

        if train_mask.sum() < 50 or val_mask.sum() < 20:
            continue

        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        y_prob = model.predict_proba(X[val_mask])[:, 1]

        try:
            aucs.append(roc_auc_score(y[val_mask], y_prob))
        except ValueError:
            pass

    if not aucs:
        return None, None, None

    mean_auc = round(np.mean(aucs), 4)
    std_auc = round(np.std(aucs), 4)
    stability = round(1.0 - std_auc, 4) if std_auc < 1.0 else 0.0
    return mean_auc, std_auc, stability


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

def main():
    t0 = time.time()
    now = datetime.now()

    print("=" * 90)
    print("  MODEL REGISTRY BUILDER")
    print(f"  Date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output: {REGISTRY_DIR}")
    print("=" * 90)

    # ── 1. Load & prepare ───────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")

    records = df.to_dict("records")
    X_raw = np.vstack([extract_features(r) for r in records])
    timestamps = pd.to_datetime(df["signal_timestamp"])
    years = np.array([t.year if pd.notna(t) else 0 for t in timestamps])

    # Drop zero-variance features (same 9 as research pipeline)
    stds = X_raw.std(axis=0)
    keep_mask = stds > 0
    for i, name in enumerate(FEATURE_NAMES):
        if name == "macro_event_risk_score":
            n_unique = len(np.unique(X_raw[:, i]))
            nz_pct = np.count_nonzero(X_raw[:, i]) / len(X_raw)
            if n_unique <= 2 and nz_pct < 0.05:
                keep_mask[i] = False

    X = X_raw[:, keep_mask]
    active_names = [n for n, k in zip(FEATURE_NAMES, keep_mask) if k]
    dropped_names = [n for n, k in zip(FEATURE_NAMES, keep_mask) if not k]
    leaked = validate_no_post_signal_labels_in_features(active_names)
    if leaked:
        raise RuntimeError(f"Feature leakage detected in active feature names: {leaked}")
    print(f"  Features: {X.shape[1]} active (dropped {sum(~keep_mask)}: {dropped_names})")

    # Build correct_60m_all target
    spot_signal = df["spot_at_signal"].values
    spot_60m = df["spot_60m"].values
    dir_numeric = df["direction_numeric"].values

    valid = np.isfinite(spot_signal) & np.isfinite(spot_60m) & (spot_signal > 0)
    raw_return = np.full(len(df), np.nan)
    raw_return[valid] = (spot_60m[valid] - spot_signal[valid]) / spot_signal[valid] * 10000

    has_dir = np.abs(dir_numeric) > 0
    valid_dir = np.isfinite(raw_return) & has_dir
    y_all = np.full(len(df), np.nan)
    y_all[valid_dir] = (np.sign(dir_numeric[valid_dir]) == np.sign(raw_return[valid_dir])).astype(float)

    target_name = "correct_60m_all"
    n_valid = np.isfinite(y_all).sum()
    pos_rate = y_all[np.isfinite(y_all)].mean()
    print(f"  Target: {target_name} — {n_valid} valid samples, pos_rate={pos_rate:.4f}")

    # ── 2. Temporal splits ──────────────────────────────────────────
    # Holdout: train 2016-2023 / test 2024-2025 (for uncalibrated models)
    # Platt:   train 2016-2022 / calibrate 2023 / test 2024-2025
    train_mask_full = (years >= 2016) & (years <= 2023) & np.isfinite(y_all)
    test_mask = (years >= 2024) & np.isfinite(y_all)

    train_mask_short = (years >= 2016) & (years <= 2022) & np.isfinite(y_all)
    cal_mask = (years == 2023) & np.isfinite(y_all)

    X_train_full, y_train_full = X[train_mask_full], y_all[train_mask_full]
    X_test, y_test = X[test_mask], y_all[test_mask]
    X_train_short, y_train_short = X[train_mask_short], y_all[train_mask_short]
    X_cal, y_cal = X[cal_mask], y_all[cal_mask]

    print(f"\n  Splits:")
    print(f"    Train (full):  2016-2023 → {len(y_train_full)} samples")
    print(f"    Train (short): 2016-2022 → {len(y_train_short)} samples (for Platt)")
    print(f"    Calibrate:     2023      → {len(y_cal)} samples")
    print(f"    Test:          2024-2025 → {len(y_test)} samples")

    # ── 3. Train & serialize each model ─────────────────────────────
    print(f"\n[2/5] Training and serializing {len(MODEL_CONFIGS)} models...")
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0",
        "created": now.isoformat(),
        "dataset": str(DATASET_PATH),
        "target": target_name,
        "n_features_original": N_FEATURES,
        "n_features_active": len(active_names),
        "feature_names": active_names,
        "all_feature_names": list(FEATURE_NAMES),
        "dropped_features": dropped_names,
        "feature_mask": keep_mask.tolist(),
        "train_samples": int(len(y_train_full)),
        "test_samples": int(len(y_test)),
        "test_years": [2024, 2025],
        "success_criteria": {
            "auc_threshold": SUCCESS_AUC,
            "ece_threshold": SUCCESS_ECE,
            "ranking_spread_threshold": SUCCESS_RANKING_SPREAD,
            "stability_threshold": SUCCESS_STABILITY,
        },
        "models": {},
    }

    trained_models = {}  # name -> fitted sklearn model (for Platt base)

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n  ── {model_name} {'─' * (60 - len(model_name))}")

        model_dir = REGISTRY_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if model_name == "GBT_shallow_platt_v1":
            # Special handling: Platt scaling on top of GBT_shallow
            print(f"    Training base GBT on 2016-2022...")
            base_factory = MODEL_CONFIGS["GBT_shallow_v1"]["factory"]
            base_model = base_factory()
            base_model.fit(X_train_short, y_train_short)

            print(f"    Fitting Platt calibrator on 2023 ({len(y_cal)} samples)...")
            try:
                # sklearn>=1.6: "prefit" is removed. Wrap a fitted model with
                # FrozenEstimator and set cv=None.
                from sklearn.frozen import FrozenEstimator

                calibrated = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base_model),
                    method="sigmoid",
                    cv=None,
                )
            except Exception:
                # Backward-compat fallback for older sklearn releases.
                calibrated = CalibratedClassifierCV(
                    base_model,
                    method="sigmoid",
                    cv="prefit",
                )
            calibrated.fit(X_cal, y_cal)
            sklearn_model = calibrated
            train_split_desc = "2016-2022 (base) + 2023 (Platt calibration)"
        else:
            print(f"    Training on 2016-2023 ({len(y_train_full)} samples)...")
            sklearn_model = config["factory"]()
            sklearn_model.fit(X_train_full, y_train_full)
            train_split_desc = "2016-2023"

        trained_models[model_name] = sklearn_model

        # Evaluate on test set
        y_prob_test = sklearn_model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_prob_test)
        quintile_result = quintile_analysis(y_test, y_prob_test)

        # Train AUC (for overfit gap measurement)
        if model_name == "GBT_shallow_platt_v1":
            y_prob_train = sklearn_model.predict_proba(X_train_short)[:, 1]
            train_auc = round(roc_auc_score(y_train_short, y_prob_train), 4)
        else:
            y_prob_train = sklearn_model.predict_proba(X_train_full)[:, 1]
            train_auc = round(roc_auc_score(y_train_full, y_prob_train), 4)

        # CV stability (only for non-calibrated models — use their factory)
        if config["factory"] is not None:
            cv_mean, cv_std, cv_stability = compute_cv_stability(
                X, y_all, years, config["factory"],
            )
        else:
            # For Platt, inherit stability from base GBT
            gbt_conf = MODEL_CONFIGS["GBT_shallow_v1"]
            cv_mean, cv_std, cv_stability = compute_cv_stability(
                X, y_all, years, gbt_conf["factory"],
            )

        # Success criteria
        criteria = evaluate_success_criteria(test_metrics, quintile_result, cv_stability)

        print(f"    Test AUC: {test_metrics['roc_auc']}  |  ECE: {test_metrics['ece']}  |  "
              f"Brier: {test_metrics['brier']}  |  Accuracy: {test_metrics['accuracy']}")
        print(f"    Train AUC: {train_auc}  |  Overfit gap: {round(train_auc - (test_metrics['roc_auc'] or 0), 4)}")
        print(f"    CV stability: {cv_stability}  |  Ranking spread: "
              f"{quintile_result['spread'] if quintile_result else 'N/A'}")
        print(f"    Criteria: {criteria['pass_count']}/{criteria['total_count']} pass"
              f"{'  ★ ALL PASS' if criteria['all_pass'] else ''}")

        # Serialize as TrainedMovePredictor
        wrapper = TrainedMovePredictor(
            model=sklearn_model,
            feature_names=active_names,
            feature_mask=keep_mask,
        )
        model_path = model_dir / "model.joblib"
        joblib.dump(wrapper, model_path)
        model_size = model_path.stat().st_size
        print(f"    Saved: {model_path} ({model_size / 1024:.1f} KB)")

        # Per-model metadata
        meta = {
            "model_name": model_name,
            "model_class": config["model_class"],
            "calibration": config["calibration"],
            "hyperparameters": config["hyperparameters"],
            "target": target_name,
            "train_split": train_split_desc,
            "test_years": [2024, 2025],
            "n_features": len(active_names),
            "feature_names": active_names,
            "feature_mask": keep_mask.tolist(),
            "dropped_features": dropped_names,
            "train_auc": train_auc,
            "overfit_gap": round(train_auc - (test_metrics["roc_auc"] or 0), 4),
            "test_metrics": test_metrics,
            "quintile_analysis": quintile_result,
            "cv_mean_auc": cv_mean,
            "cv_std_auc": cv_std,
            "cv_stability": cv_stability,
            "success_criteria": criteria,
            "artifact_path": f"registry/{model_name}/model.joblib",
            "artifact_size_kb": round(model_size / 1024, 1),
            "created": now.isoformat(),
            "random_state": RANDOM_STATE,
        }

        meta_path = model_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Add to manifest
        manifest["models"][model_name] = {
            "path": f"registry/{model_name}/model.joblib",
            "meta_path": f"registry/{model_name}/meta.json",
            "model_class": config["model_class"],
            "calibration": config["calibration"],
            "target": target_name,
            "test_auc": test_metrics["roc_auc"],
            "test_ece": test_metrics["ece"],
            "test_brier": test_metrics["brier"],
            "cv_stability": cv_stability,
            "ranking_spread": quintile_result["spread"] if quintile_result else None,
            "criteria_pass": criteria["pass_count"],
            "criteria_total": criteria["total_count"],
            "all_criteria_pass": criteria["all_pass"],
            "artifact_size_kb": round(model_size / 1024, 1),
        }

    # ── 4. Write manifest ───────────────────────────────────────────
    print(f"\n[3/5] Writing manifest...")
    manifest_path = REGISTRY_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Written: {manifest_path}")

    # ── 5. Summary report ───────────────────────────────────────────
    print(f"\n[4/5] Registry Summary")
    print(f"{'─' * 90}")
    print(f"  {'Model':<30} {'AUC':>7} {'ECE':>7} {'Brier':>7} {'Stab':>7} {'Spread':>8} {'Pass':>6}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*6}")

    for name, info in manifest["models"].items():
        star = " ★" if info["all_criteria_pass"] else ""
        print(f"  {name:<30} {info['test_auc'] or 'N/A':>7} {info['test_ece'] or 'N/A':>7} "
              f"{info['test_brier'] or 'N/A':>7} {info['cv_stability'] or 'N/A':>7} "
              f"{info['ranking_spread'] or 'N/A':>8} {info['criteria_pass']}/{info['criteria_total']}{star}")

    # ── 6. Write research documentation ─────────────────────────────
    print(f"\n[5/5] Writing documentation...")
    doc_lines = [
        "=" * 90,
        "  MODEL REGISTRY — Research Documentation",
        f"  Built: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 90,
        "",
        "PURPOSE",
        "─" * 40,
        "This registry contains all ML models evaluated during the signal research",
        "pipeline. Each model is serialized as a TrainedMovePredictor wrapper that",
        "production code can load directly.",
        "",
        "HOW TO SWITCH PRODUCTION MODEL",
        "─" * 40,
        "  Option 1: Set in config/settings.py:",
        '    ACTIVE_MODEL = "GBT_shallow_v1"',
        "",
        "  Option 2: Set environment variable:",
        '    export OQE_ACTIVE_MODEL="GBT_shallow_v1"',
        "",
        "  Option 3: Revert to legacy model:",
        '    ACTIVE_MODEL = ""  (or unset env var)',
        "",
        f"AVAILABLE MODELS ({len(manifest['models'])})",
        "─" * 40,
    ]

    for name, info in manifest["models"].items():
        star = " ★ ALL CRITERIA PASS" if info["all_criteria_pass"] else ""
        doc_lines.extend([
            f"",
            f"  {name}{star}",
            f"    Type:       {info['model_class']}",
            f"    Calibration:{info['calibration']}",
            f"    Test AUC:   {info['test_auc']}",
            f"    ECE:        {info['test_ece']}",
            f"    Brier:      {info['test_brier']}",
            f"    Stability:  {info['cv_stability']}",
            f"    Ranking:    spread={info['ranking_spread']}",
            f"    Criteria:   {info['criteria_pass']}/{info['criteria_total']}",
            f"    Artifact:   {info['path']} ({info['artifact_size_kb']} KB)",
        ])

    doc_lines.extend([
        "",
        "DATASET",
        "─" * 40,
        f"  Source:   {DATASET_PATH}",
        f"  Target:   {target_name}",
        f"  Features: {len(active_names)} active / {N_FEATURES} original",
        f"  Dropped:  {dropped_names}",
        f"  Train:    {len(y_train_full)} samples (2016-2023)",
        f"  Test:     {len(y_test)} samples (2024-2025)",
        "",
        "SUCCESS CRITERIA",
        "─" * 40,
        f"  AUC > {SUCCESS_AUC}",
        f"  ECE < {SUCCESS_ECE}",
        f"  Ranking spread > {SUCCESS_RANKING_SPREAD} (monotonic)",
        f"  CV stability > {SUCCESS_STABILITY}",
        "",
        f"  Runtime: {time.time() - t0:.1f}s",
        "",
    ])

    doc_path = REGISTRY_DIR / "registry_report.txt"
    with open(doc_path, "w") as f:
        f.write("\n".join(doc_lines))
    print(f"  Written: {doc_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"  REGISTRY BUILD COMPLETE — {len(manifest['models'])} models serialized in {elapsed:.1f}s")
    print(f"  Location: {REGISTRY_DIR}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
