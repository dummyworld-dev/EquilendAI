import pandas as pd

from src.preprocessing.feature_encoding import encode_categorical_features
from src.preprocessing.scaling import scale_numeric_features


def test_encode_categorical_features_creates_dummies():
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "employment_length": ["1-3 years", "8+ years"],
            "monthly_income": [50000, 60000],
            "utility_bill_average": [2000, 2500],
            "repayment_history_pct": [80, 90],
            "default_status": [0, 1],
        }
    )

    encoded = encode_categorical_features(df)

    # Original categorical columns should be expanded into dummy columns.
    assert "gender_Male" in encoded.columns or "gender_Female" in encoded.columns
    assert any(col.startswith("employment_length_") for col in encoded.columns)


def test_scale_numeric_features_transforms_values():
    df = pd.DataFrame(
        {
            "monthly_income": [50000, 60000],
            "utility_bill_average": [2000, 2500],
            "repayment_history_pct": [80, 90],
            "default_status": [0, 1],
        }
    )

    scaled_df, scaler = scale_numeric_features(df)

    # default_status should remain unchanged.
    assert (scaled_df["default_status"] == df["default_status"]).all()

    # Scaled numeric columns should have zero mean (approximately) under StandardScaler.
    numeric_cols = ["monthly_income", "utility_bill_average", "repayment_history_pct"]
    means = scaled_df[numeric_cols].mean().abs()
    assert (means < 1e-6).all()

    # Scaler should retain the same number of features as input numeric columns.
    assert scaler.n_features_in_ == len(numeric_cols)

