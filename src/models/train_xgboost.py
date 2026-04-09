import os
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.preprocessing.data_cleaning import load_and_clean
from src.preprocessing.feature_encoding import encode_categorical_features
from src.preprocessing.scaling import scale_numeric_features


def load_prepared_data(csv_path: str) -> pd.DataFrame:
    """
    Load, clean, encode, and scale the EquiLend dataset.

    This mirrors the preprocessing used in the Random Forest pipeline so that
    both models see comparable feature representations.
    """
    df = load_and_clean(csv_path)
    df = encode_categorical_features(df)
    df, _ = scale_numeric_features(df)
    return df


def train_xgboost_classifier(
    csv_path: str,
    model_output_path: str = "models/xgboost_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier on the cleaned dataset.

    Includes:
    - Train/test split.
    - Optional class balancing using SMOTE on the training data.
    - XGBoost model with reasonably tuned hyperparameters.
    - Evaluation metrics on the test set.
    - Saving of the trained model to disk.

    Returns:
        Dict[str, Any]: Dictionary with key performance metrics.
    """
    # Prepare the data
    df = load_prepared_data(csv_path)

    if "default_status" not in df.columns:
        raise ValueError("Expected 'default_status' column as the target.")

    X = df.drop(columns=["default_status"])
    y = df["default_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Optionally apply SMOTE for class balancing on the training set only.
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Base XGBoost classifier; hyperparameters will be tuned via RandomizedSearchCV.
    base_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        # scale_pos_weight is left at 1.0 because SMOTE balances classes.
        scale_pos_weight=1.0,
        use_label_encoder=False,
    )

    # Hyperparameter search space for XGBoost.
    param_distributions = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
    }

    # Stratified K-fold cross-validation to respect class balance.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=random_state,
    )

    # Run hyperparameter search on the training set.
    search.fit(X_train, y_train)
    clf = search.best_estimator_

    print("Best XGBoost hyperparameters found via RandomizedSearchCV:")
    print(search.best_params_)

    # Predictions and evaluation on the untouched test set.
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        # In edge cases where only one class is present in y_test
        auc = np.nan

    print("XGBoost Evaluation Metrics (Best Model)")
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    if not np.isnan(auc):
        print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Ensure output directory exists and save model.
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"\nXGBoost model saved to: {model_output_path}")

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


if __name__ == "__main__":
    default_csv_path = "data/equilend_mock_data.csv"
    train_xgboost_classifier(default_csv_path)

