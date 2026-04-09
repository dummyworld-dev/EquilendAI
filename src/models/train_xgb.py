"""
Tasks 06 & 10 — XGBoost Model Training

Two usage modes:
  1. train_advanced_model / evaluate_advanced_model
       Accepts already-preprocessed numpy arrays (used by the upstream app flow
       where preprocessing happens inline in app.py before calling this).

  2. train_and_save / load_artifacts
       Handles its own preprocessing via the sklearn pipeline, persists the
       model + preprocessor + test-set predictions to disk, and is used by the
       Threshold Optimizer page in the Streamlit app.
"""

import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb

# ── Path resolution ────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from evaluation.thresholds import get_business_recommended_threshold

# ── Defaults (check both data locations) ──────────────────────────────────────
_DATA_CANDIDATES = [
    os.path.join(_ROOT_DIR, "data",         "equilend_mock_data.csv"),
    os.path.join(_ROOT_DIR, "scripts", "data", "equilend_mock_data.csv"),
]
DEFAULT_DATA_PATH  = next((p for p in _DATA_CANDIDATES if os.path.exists(p)),
                          _DATA_CANDIDATES[0])
DEFAULT_MODELS_DIR = os.path.join(_ROOT_DIR, "models")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — takes pre-processed data (used by the inline preprocess() flow)
# ══════════════════════════════════════════════════════════════════════════════

def train_advanced_model(X_train, y_train):
    """
    Train an XGBoost model with GridSearchCV to maximise Accuracy.
    Accepts already-preprocessed feature arrays.
    """
    xgb_clf = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42,
    )

    param_grid = {
        "n_estimators":    [200, 400],
        "max_depth":       [4, 6, 8],
        "learning_rate":   [0.01, 0.05, 0.1],
        "min_child_weight":[1, 3],
        "subsample":       [0.8, 0.9],
        "colsample_bytree":[0.8, 0.9],
        "gamma":           [0, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_advanced_model(model, X_test, y_test):
    """Return a dict of evaluation metrics for the trained model."""
    y_pred  = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy":        accuracy_score(y_test, y_pred),
        "precision":       precision_score(y_test, y_pred, zero_division=0),
        "recall":          recall_score(y_test, y_pred, zero_division=0),
        "auc":             roc_auc_score(y_test, y_probs),
        "false_positives": fp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — self-contained: reads CSV, runs sklearn pipeline, saves artifacts
#           (used by the Threshold Optimizer page)
# ══════════════════════════════════════════════════════════════════════════════

def train_and_save(
    data_path:  str = DEFAULT_DATA_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
) -> tuple:
    """
    Train XGBoost end-to-end (with preprocessing), persist artifacts to disk.

    Returns:
        (model, preprocessor, y_test_array, y_prob_array, auc_score, threshold_info)
    """
    from preprocessing.pipeline import (
        build_preprocessing_pipeline,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )

    os.makedirs(models_dir, exist_ok=True)

    df     = pd.read_csv(data_path)
    feats  = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X      = df[feats]
    y      = df["default_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessing_pipeline()
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = spw,
        eval_metric      = "auc",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_tr, y_train)

    y_prob = model.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    threshold_info = get_business_recommended_threshold(y_test.values, y_prob)

    with open(os.path.join(models_dir, "xgb_model.pkl"),    "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    np.save(os.path.join(models_dir, "y_test.npy"), y_test.values)
    np.save(os.path.join(models_dir, "y_prob.npy"), y_prob)
    with open(os.path.join(models_dir, "threshold_info.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)

    return model, preprocessor, y_test.values, y_prob, auc, threshold_info


def load_artifacts(models_dir: str = DEFAULT_MODELS_DIR):
    """
    Load saved model artifacts produced by train_and_save().

    Returns:
        (model, preprocessor, y_test_array, y_prob_array, threshold_info)
        or None if any artifact is missing.
    """
    paths = {
        "model":        os.path.join(models_dir, "xgb_model.pkl"),
        "preprocessor": os.path.join(models_dir, "preprocessor.pkl"),
        "y_test":       os.path.join(models_dir, "y_test.npy"),
        "y_prob":       os.path.join(models_dir, "y_prob.npy"),
        "threshold_info": os.path.join(models_dir, "threshold_info.json"),
    }

    if not all(os.path.exists(p) for p in paths.values()):
        return None

    with open(paths["model"],        "rb") as f:
        model = pickle.load(f)
    with open(paths["preprocessor"], "rb") as f:
        preprocessor = pickle.load(f)

    with open(paths["threshold_info"], "r", encoding="utf-8") as f:
        threshold_info = json.load(f)

    return model, preprocessor, np.load(paths["y_test"]), np.load(paths["y_prob"]), threshold_info


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training XGBoost (self-contained mode) …")
    _, _, y_test, y_prob, auc, threshold_info = train_and_save()
    print(f"Done — ROC-AUC = {auc:.4f}")
    print(f"Business threshold = {threshold_info['threshold']:.2f}")
    print(f"Artifacts saved to: {DEFAULT_MODELS_DIR}/")
