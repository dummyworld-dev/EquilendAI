import pandas as pd

from src.preprocessing.data_cleaning import clean_equilend_data


def test_clean_equilend_data_handles_missing_and_types():
    # Create a small dataframe mimicking the generated schema with some issues.
    df = pd.DataFrame(
        {
            "gender": ["Male", None],
            "monthly_income": [50000, "60000"],
            "utility_bill_average": [2000, None],
            "repayment_history_pct": [80, None],
            "employment_length": ["1-3 years", None],
            "default_status": [1, 0],
        }
    )

    cleaned = clean_equilend_data(df)

    # No missing values should remain in the core numeric/categorical features.
    assert not cleaned["utility_bill_average"].isna().any()
    assert not cleaned["repayment_history_pct"].isna().any()
    assert not cleaned["employment_length"].isna().any()

    # Numeric columns should be numeric dtype.
    assert pd.api.types.is_numeric_dtype(cleaned["monthly_income"])
    assert pd.api.types.is_numeric_dtype(cleaned["utility_bill_average"])
    assert pd.api.types.is_numeric_dtype(cleaned["repayment_history_pct"])

