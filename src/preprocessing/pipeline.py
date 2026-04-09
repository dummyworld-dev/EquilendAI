"""
Task 04 — Scaling & Encoding Pipeline
Builds a scikit-learn ColumnTransformer that handles both numeric
(iterative imputation + scaling) and categorical (imputation + one-hot) features.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["monthly_income", "utility_bill_average", "repayment_history_pct"]
CATEGORICAL_FEATURES = ["employment_length", "gender"]


def infer_feature_types(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Infer or validate which columns are numeric vs. categorical.
    """
    if numeric_cols is None:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != "default_status"
        ]
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c != "default_status"]
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    df_sample: Optional[pd.DataFrame] = None,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a robust preprocessing ColumnTransformer.

    Numeric path:
    - IterativeImputer (for richer missing-value estimation)
    - StandardScaler

    Categorical path:
    - SimpleImputer(most_frequent)
    - OneHotEncoder(handle_unknown='ignore')
    """
    if df_sample is not None:
        numeric_cols, categorical_cols = infer_feature_types(
            df_sample, numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )

    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", IterativeImputer(max_iter=10, random_state=42)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor
