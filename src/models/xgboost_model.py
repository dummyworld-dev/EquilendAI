import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Support both:
# - `python -m src.models.xgboost_model` (package execution)
# - `python src/models/xgboost_model.py` (direct script execution)
try:
    from src.preprocessing.data_cleaning import load_and_clean
    from src.preprocessing.feature_encoding import encode_categorical_features
except ModuleNotFoundError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.preprocessing.data_cleaning import load_and_clean
    from src.preprocessing.feature_encoding import encode_categorical_features


@dataclass(frozen=True)
class XGBoostModelArtifact:
    """
    A single, joblib-serializable bundle containing everything needed at inference time.

    Keeping model + scaler + feature columns together prevents train/serve skew.
    """

    model: Any
    scaler: StandardScaler
    feature_cols: List[str]
    best_params: Dict[str, Any]
    test_auc: float


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make feature names compatible with XGBoost restrictions.

    XGBoost disallows feature names containing characters like:
    - '['
    - ']'
    - '<'
    """
    sanitized = df.copy()
    sanitized.columns = (
        sanitized.columns.astype(str)
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "lt_", regex=False)
        .str.replace(">", "gt_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return sanitized


def _prepare_training_frame(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV and apply the existing preprocessing steps.

    Requirements note:
    - We keep the project's current preprocessing intact by reusing:
      - `load_and_clean` (missing values / outliers)
      - `encode_categorical_features` (one-hot encoding)
    """
    df = load_and_clean(csv_path)
    df = encode_categorical_features(df)
    df = sanitize_feature_names(df)
    return df


def _fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on training features for reproducible scaling.

    The fitted scaler is saved inside the artifact and reused at inference time.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def train_tuned_xgboost_with_smote(
    csv_path: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
    n_iter: int = 20,
) -> Tuple[XGBoostModelArtifact, Dict[str, Any]]:
    """
    Train an XGBoost classifier with SMOTE + hyperparameter tuning and evaluate via AUC.

    Hyperparameters tuned (per requirement):
    - max_depth
    - learning_rate
    - n_estimators
    - subsample

    Returns:
        (artifact, metrics)
    """
    df = _prepare_training_frame(csv_path)

    if "default_status" not in df.columns:
        raise ValueError("Expected 'default_status' column as the target.")

    feature_cols = [c for c in df.columns if c != "default_status"]
    X = df[feature_cols]
    y = df["default_status"].astype(int)

    # Split first, so SMOTE is applied ONLY on the training set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features after split to avoid leaking test distribution.
    scaler = _fit_scaler(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    # Handle class imbalance using SMOTE (training set only).
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    # Base model; tuned via RandomizedSearchCV.
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        # With SMOTE enabled, class weights are less necessary.
        scale_pos_weight=1.0,
    )

    # Hyperparameter tuning space (focused on requested parameters).
    param_distributions = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 300, 400, 600],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        random_state=random_state,
        verbose=0,
    )

    # Fit the search on (optionally SMOTE-balanced) training data.
    search.fit(X_train_scaled, y_train)
    best_model = search.best_estimator_

    # Evaluate on the untouched test set using AUC.
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = float(roc_auc_score(y_test, y_proba))

    artifact = XGBoostModelArtifact(
        model=best_model,
        scaler=scaler,
        feature_cols=feature_cols,
        best_params=dict(search.best_params_),
        test_auc=test_auc,
    )

    metrics = {
        "test_auc": test_auc,
        "best_params": dict(search.best_params_),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "use_smote": bool(use_smote),
    }
    return artifact, metrics


def save_artifact(artifact: XGBoostModelArtifact, output_path: str) -> None:
    """
    Save the trained model artifact to disk using joblib.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(artifact, output_path)


def load_artifact(model_path: str) -> XGBoostModelArtifact:
    """
    Load a previously saved XGBoostModelArtifact.
    """
    # Backward compatibility:
    # If the artifact was created while running this file directly
    # (`python src/models/xgboost_model.py`), pickle may record the class path
    # as `main.XGBoostModelArtifact`. In Streamlit/runtime contexts, that module
    # path is different, causing:
    # "module 'main' has no attribute 'XGBoostModelArtifact'".
    #
    # We alias the class onto any loaded `main`/`__main__` modules before loading.
    main_mod = sys.modules.get("main")
    if main_mod is not None and not hasattr(main_mod, "XGBoostModelArtifact"):
        setattr(main_mod, "XGBoostModelArtifact", XGBoostModelArtifact)
    dunder_main_mod = sys.modules.get("__main__")
    if dunder_main_mod is not None and not hasattr(dunder_main_mod, "XGBoostModelArtifact"):
        setattr(dunder_main_mod, "XGBoostModelArtifact", XGBoostModelArtifact)

    artifact = joblib.load(model_path)
    if not isinstance(artifact, XGBoostModelArtifact):
        raise TypeError(
            "Loaded object is not an XGBoostModelArtifact. "
            "Re-train and re-save the model using xgboost_model.py."
        )
    return artifact


def predict_default_probability(
    artifact: XGBoostModelArtifact,
    raw_features: Dict[str, Any],
) -> float:
    """
    Generate a default probability from raw (unencoded, unscaled) features.

    This function:
    - wraps raw_features into a single-row DataFrame
    - one-hot encodes categorical variables (consistent with training)
    - aligns columns to training feature set
    - scales using the saved scaler
    - returns P(default=1)
    """
    user_df = pd.DataFrame([raw_features])
    user_encoded = encode_categorical_features(user_df)
    user_encoded = sanitize_feature_names(user_encoded)

    # Add any missing training-time columns (unseen categories => 0 dummy columns).
    for col in artifact.feature_cols:
        if col not in user_encoded.columns:
            user_encoded[col] = 0.0
    user_encoded = user_encoded[artifact.feature_cols]

    user_scaled = artifact.scaler.transform(user_encoded)
    return float(artifact.model.predict_proba(user_scaled)[0, 1])


def train_and_save(
    csv_path: str = "data/equilend_mock_data.csv",
    output_path: str = "models/xgboost_model.joblib",
) -> Dict[str, Any]:
    """
    Convenience entrypoint:
    - trains tuned XGBoost with SMOTE
    - prints and returns metrics
    - saves a single artifact to disk
    """
    artifact, metrics = train_tuned_xgboost_with_smote(csv_path)
    save_artifact(artifact, output_path)
    return metrics


def train_xgb_model(
    csv_path: str = "data/equilend_mock_data.csv",
    output_path: str = "models/xgboost_model.joblib",
) -> XGBoostModelArtifact:
    """
    Required public API:
    Train tuned XGBoost model and save artifact to disk.
    """
    artifact, metrics = train_tuned_xgboost_with_smote(csv_path)
    save_artifact(artifact, output_path)
    print(f"Saved model artifact to: {output_path}")
    print(f"Test AUC: {metrics['test_auc']:.4f}")
    print(f"Best params: {metrics['best_params']}")
    return artifact


def predict_xgb(model: XGBoostModelArtifact, X_new: pd.DataFrame) -> np.ndarray:
    """
    Required public API:
    Predict default probabilities for new samples.

    Args:
        model: Loaded XGBoostModelArtifact (via `load_artifact` or `train_xgb_model`).
        X_new: Raw feature dataframe (unencoded/unscaled) with columns from training schema.
    """
    # Apply the same one-hot encoding as training.
    X_encoded = encode_categorical_features(X_new.copy())
    X_encoded = sanitize_feature_names(X_encoded)

    # Align with training feature columns.
    for col in model.feature_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0.0
    X_encoded = X_encoded[model.feature_cols]

    # Reuse saved scaler and model.
    X_scaled = model.scaler.transform(X_encoded)
    return model.model.predict_proba(X_scaled)[:, 1]


if __name__ == "__main__":
    train_xgb_model()

