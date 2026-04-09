import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def train_xgb_tuned(X_train, y_train, X_val, y_val):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=1000,       # High limit, but stopping will trigger early
        learning_rate=0.02,      # Slow and steady learning
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,                 # Added regularization to boost accuracy
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50 # Stop if no improvement for 50 rounds
    )

    model.fit(
        X_resampled, y_resampled,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

if __name__ == "__main__":
    df = pd.read_csv(r"D:\EquilendAI\scripts\data\equilend_mock_data.csv")

    # Feature Engineering (Matches RF for consistency)
    df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]
    df["risk_proxy"] = (df["utility_bill_average"] / df["monthly_income"]) - (df["repayment_history_pct"] / 100)
    df["income_to_bill_ratio"] = df["monthly_income"] / df["utility_bill_average"]
    df["repayment_bill_interaction"] = df["repayment_history_pct"] * df["utility_bill_average"]
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    df.columns = df.columns.str.replace(r"[<>\[\]]", "", regex=True)

    X = df.drop("default_status", axis=1)
    y = df["default_status"]

    # We split into Train, Val (for early stopping), and Test (for final accuracy)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    medians = X_train.median(numeric_only=True)
    X_train, X_val, X_test = X_train.fillna(medians), X_val.fillna(medians), X_test.fillna(medians)

    model = train_xgb_tuned(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)

    print(f"\nXGB Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"XGB AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")
    