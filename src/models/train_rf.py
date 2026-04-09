import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_rf_final(X_train, y_train):
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Parameters tuned for 0.88 accuracy without nested parallelism hangs
    model = RandomForestClassifier(
        n_estimators=600,         
        max_depth=22,              
        min_samples_leaf=1,        
        min_samples_split=5,       
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1                  
    )

    print("Training Random Forest...")
    model.fit(X_resampled, y_resampled)
    return model

if __name__ == "__main__":
    df = pd.read_csv(r"D:\EquilendAI\scripts\data\equilend_mock_data.csv")
    
    # Feature Engineering
    df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]
    df["risk_proxy"] = (df["utility_bill_average"] / df["monthly_income"]) - (df["repayment_history_pct"] / 100)
    df["income_to_bill_ratio"] = df["monthly_income"] / df["utility_bill_average"]
    df["repayment_bill_interaction"] = df["repayment_history_pct"] * df["utility_bill_average"]
    df["log_income"] = np.log1p(df["monthly_income"])
    
    df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("default_status", axis=1)
    y = df["default_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    medians = X_train.median(numeric_only=True)
    X_train, X_test = X_train.fillna(medians), X_test.fillna(medians)

    model = train_rf_final(X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")