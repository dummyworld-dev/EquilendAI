import os

import sys

import pickle

import json

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    roc_auc_score,

    confusion_matrix,

)

import xgboost as xgb

from imblearn.over_sampling import SMOTE



# ── Path resolution ────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_SRC_DIR  = os.path.dirname(_THIS_DIR)

_ROOT_DIR = os.path.dirname(_SRC_DIR)

if _SRC_DIR not in sys.path:

    sys.path.insert(0, _SRC_DIR)



# Import business logic from evaluation module

try:

    from evaluation.thresholds import get_business_recommended_threshold

except ImportError:

    # Fallback if evaluation module is not in path

    def get_business_recommended_threshold(y_true, y_prob):

        return {"threshold": 0.5, "metric": "default"}



# ── Defaults ──────────────────────────────────────────────────────────────────

_DATA_CANDIDATES = [

    os.path.join(_ROOT_DIR, "data", "equilend_mock_data.csv"),

    os.path.join(_ROOT_DIR, "scripts", "data", "equilend_mock_data.csv"),

    r"D:\EquilendAI\scripts\data\equilend_mock_data.csv"

]

DEFAULT_DATA_PATH  = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])

DEFAULT_MODELS_DIR = os.path.join(_ROOT_DIR, "models")



# ══════════════════════════════════════════════════════════════════════════════

# MODE 1 — High-Accuracy Tuned Model (Feature_10 Logic)

# ══════════════════════════════════════════════════════════════════════════════



def train_xgb_tuned(X_train, y_train, X_val, y_val):

    """

    Tuned version from feature_10 that achieved 0.878 accuracy.

    Uses SMOTE and specific regularization.

    """

    logger_print("Applying SMOTE for class balance...")

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)



    model = xgb.XGBClassifier(

        n_estimators=1000,

        learning_rate=0.02,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        gamma=1,

        random_state=42,

        eval_metric="logloss",

        early_stopping_rounds=50

    )



    model.fit(

        X_resampled, y_resampled,

        eval_set=[(X_val, y_val)],

        verbose=False

    )

    return model



# ══════════════════════════════════════════════════════════════════════════════

# MODE 2 — End-to-End Production (Main Branch Structure)

# ══════════════════════════════════════════════════════════════════════════════



def train_and_save(data_path=DEFAULT_DATA_PATH, models_dir=DEFAULT_MODELS_DIR):

    """

    End-to-end flow: Preprocessing -> Tuned Training -> Saving Artifacts.

    """

    os.makedirs(models_dir, exist_ok=True)

   

    # 1. Load and Engineer Features (Feature_10 Logic)

    df = pd.read_csv(data_path)

    df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]

    df["risk_proxy"] = (df["utility_bill_average"] / df["monthly_income"]) - (df["repayment_history_pct"] / 100)

    df["income_to_bill_ratio"] = df["monthly_income"] / df["utility_bill_average"]

    df["repayment_bill_interaction"] = df["repayment_history_pct"] * df["utility_bill_average"]

   

    df.replace([float("inf"), -float("inf")], np.nan, inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    df.columns = df.columns.str.replace(r"[<>\[\]]", "", regex=True)



    X = df.drop("default_status", axis=1)

    y = df["default_status"]



    # 2. Split (Train, Val for Early Stopping, Test for Final Report)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)



    # 3. Impute

    medians = X_train.median(numeric_only=True)

    X_train, X_val, X_test = X_train.fillna(medians), X_val.fillna(medians), X_test.fillna(medians)



    # 4. Train using the Tuned Logic

    model = train_xgb_tuned(X_train, y_train, X_val, y_val)



    # 5. Evaluate and Save Artifacts (Main structure)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    threshold_info = get_business_recommended_threshold(y_test.values, y_prob)



    # Saving as per main branch requirement

    with open(os.path.join(models_dir, "xgb_model.pkl"), "wb") as f:

        pickle.dump(model, f)

    with open(os.path.join(models_dir, "medians.pkl"), "wb") as f:

        pickle.dump(medians, f)

   

    np.save(os.path.join(models_dir, "y_test.npy"), y_test.values)

    np.save(os.path.join(models_dir, "y_prob.npy"), y_prob)

   

    with open(os.path.join(models_dir, "threshold_info.json"), "w") as f:

        json.dump(threshold_info, f, indent=2)



    return model, auc, threshold_info



def logger_print(msg):

    print(f"INFO: {msg}")



# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Running Resolved XGBoost Training Pipeline...")

    model, auc, t_info = train_and_save()

    print(f"Done — Accuracy Target: 0.878+ | Current ROC-AUC: {auc:.4f}")

    print(f"Artifacts saved to: {DEFAULT_MODELS_DIR}")

