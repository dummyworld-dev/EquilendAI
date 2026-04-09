import pandas as pd
from typing import List, Optional


def encode_categorical_features(
    df: pd.DataFrame,
    *,
    categorical_cols: Optional[List[str]] = None,
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Encode categorical features for machine learning.

    This helper is designed to work on the cleaned EquiLend dataset and
    returns a DataFrame that is ready to be passed into a model training
    script (e.g., XGBoost, RandomForest, etc.).

    Strategy:
    - Uses one-hot encoding (via pandas.get_dummies) for all specified
      categorical columns.
    - Keeps numeric columns unchanged.
    - Does not modify the target column (e.g., "default_status").

    Args:
        df (pd.DataFrame): Input dataframe, typically already cleaned.
        categorical_cols (Optional[List[str]]): Columns to one-hot encode.
            If None, a sensible default based on the synthetic generator
            is used.
        drop_first (bool): Whether to drop the first level from each
            encoded categorical variable to avoid perfect multicollinearity.

    Returns:
        pd.DataFrame: DataFrame with categorical features encoded and ready
        for modeling.
    """
    df = df.copy()

    # Default categorical columns based on scripts/generate_data.py
    if categorical_cols is None:
        categorical_cols = [
            "gender",
            "employment_length",
        ]

    # Ensure we only try to encode columns that exist in the frame.
    existing_cats = [c for c in categorical_cols if c in df.columns]

    # pandas.get_dummies will:
    # - One-hot encode the specified categorical columns.
    # - Leave numeric and other non-encoded columns (including the target)
    #   untouched.
    encoded_df = pd.get_dummies(
        df,
        columns=existing_cats,
        drop_first=drop_first,
        dtype=float,  # ensure encoded columns are numeric
    )

    return encoded_df

