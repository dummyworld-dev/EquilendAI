# Task 07: SMOTE Integration with XGBoost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def train_xgboost_with_smote(X_train, y_train):
    """
    Apply SMOTE and train XGBoost
    """

    # 🔥 Apply SMOTE ONLY on training data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("After SMOTE:")
    print(y_resampled.value_counts())

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_resampled, y_resampled)
    return model


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(r"D:\EquilendAI\scripts\data\equilend_mock_data.csv")

    # 🔥 Feature Engineering FIRST
    df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]

    # Handle division issues
    df["bill_income_ratio"] = df["bill_income_ratio"].replace(
        [float("inf"), -float("inf")], pd.NA
    )

    # Impute missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encoding
    df = pd.get_dummies(df, drop_first=True)

    # Fix column names for XGBoost
    df.columns = df.columns.str.replace(r"[<>\[\]]", "", regex=True)

    # Split data
    X = df.drop("default_status", axis=1)
    y = df["default_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model with SMOTE
    model = train_xgboost_with_smote(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nXGBoost with SMOTE ✅")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))