import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 02: Feature Engineering
    - Income to utility bill ratio
    - Cash flow consistency proxy
    - Expense brackets
    """
    # 1. Income to utility ratio
    df['income_to_utility_ratio'] = df['monthly_income'] / df['utility_bill_average'].replace(0, 1)

    # 2. Cash flow consistency
    df['cash_flow_consistency'] = df['repayment_history_pct'] / 100

    # 3. Expense brackets
    df['expense_bracket'] = pd.cut(
        df['utility_bill_average'],
        bins=[0, 1500, 3000, 5000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    return df