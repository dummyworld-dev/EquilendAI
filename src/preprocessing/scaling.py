import pandas as pd
from typing import List, Optional, Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_numeric_features(
    df: pd.DataFrame,
    *,
    numeric_cols: Optional[List[str]] = None,
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features for modeling.

    This utility applies either StandardScaler or MinMaxScaler to the selected
    numeric columns and returns both the transformed DataFrame and the fitted
    scaler object so the same transformation can be reused (e.g., on validation
    or production data).

    Args:
        df (pd.DataFrame): Input dataframe, typically already cleaned/encoded.
        numeric_cols (Optional[List[str]]): Columns to scale. If None, all
            numeric (float/int) columns except the target ("default_status")
            are scaled.
        scaler_type (str): Which scaler to use: "standard" (StandardScaler)
            or "minmax" (MinMaxScaler).

    Returns:
        Tuple[pd.DataFrame, object]:
            - Transformed DataFrame with scaled numeric columns.
            - Fitted scaler instance (StandardScaler or MinMaxScaler) that
              can be reused for reproducible transformations.
    """
    df = df.copy()

    # Decide which columns to scale.
    if numeric_cols is None:
        # Use all numeric columns except the target, if present.
        numeric_cols = [
            c
            for c in df.select_dtypes(include=["number"]).columns
            if c != "default_status"
        ]

    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
    elif scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError('scaler_type must be either "standard" or "minmax".')

    # Fit the scaler on the specified columns and transform them in place.
    scaled_values = scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = scaled_values

    return df, scaler

