
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Infer or validate which columns are numeric vs. categorical.

    This helper makes the preprocessing pipeline more reusable by allowing
    callers to either pass explicit column lists or rely on sensible defaults.
    """
    if numeric_cols is None:
        numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c != "default_status"
        ]
    if categorical_cols is None:
        categorical_cols = [
            c
            for c in df.columns
            if c not in numeric_cols and c != "default_status"
        ]
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    df_sample: Optional[pd.DataFrame] = None,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a robust preprocessing pipeline for the EquiLend dataset.

    The pipeline:
    - Imputes missing numeric values with the median.
    - Scales numeric features using StandardScaler.
    - Imputes missing categorical values with the most frequent category.
    - One-hot encodes categorical features (dropping one level to avoid redundancy).

    Passing a small sample dataframe (df_sample) allows automatic inference of
    numeric vs. categorical features; otherwise, you can supply the column
    lists directly.

    Args:
        df_sample (Optional[pd.DataFrame]): Sample dataframe used only to infer
            feature types. It is not transformed by this function.
        numeric_cols (Optional[List[str]]): Explicit list of numeric feature
            names. If None, inferred from df_sample.
        categorical_cols (Optional[List[str]]): Explicit list of categorical
            feature names. If None, inferred from df_sample.

    Returns:
        sklearn.compose.ColumnTransformer: A fitted-ready preprocessing transformer
        that can be used inside a modeling Pipeline.
    """
    if df_sample is not None:
        numeric_cols, categorical_cols = infer_feature_types(
            df_sample, numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )

    # Fallbacks in case df_sample is not provided.
    if numeric_cols is None:
        numeric_cols = ["monthly_income", "utility_bill_average", "repayment_history_pct"]
    if categorical_cols is None:
        categorical_cols = ["gender", "employment_length"]

    # Numeric pipeline: median imputation + scaling.
    numeric_transformer = Pipeline(
        steps=[
            # Median is robust to outliers and works well with skewed data.
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: most-frequent imputation + one-hot encoding.
    categorical_transformer = Pipeline(
        steps=[
            # Most frequent preserves the existing distribution reasonably well.
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )

    # ColumnTransformer applies the appropriate pipeline to each feature subset.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor

