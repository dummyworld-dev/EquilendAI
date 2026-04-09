import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 02: Data Cleaning
    - Drops duplicate rows
    - Fills missing utility_bill_average with median
    - Fills missing repayment_history_pct with median
    """
    df = df.drop_duplicates()
    df['utility_bill_average'] = df['utility_bill_average'].fillna(df['utility_bill_average'].median())
    df['repayment_history_pct'] = df['repayment_history_pct'].fillna(df['repayment_history_pct'].median())
    return df
