# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import os

from models.train_rf import train_baseline_model, evaluate_model
from models.train_xgb import train_advanced_model, evaluate_advanced_model
from sklearn.model_selection import train_test_split

# Theme Colors
PRIMARY_COLOR = "#2E7D32"
ACCENT_COLOR = "#5D4037"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "scripts", "data", "equilend_mock_data.csv")

def load_data():
    return pd.read_csv(DATA_PATH)

def preprocess(data):
    # 1. Feature Engineering (Adding logic to help the model learn faster)
    # Ratio of utility bills to income - high ratio often indicates risk
    data['bill_to_income_ratio'] = data['utility_bill_average'] / (data['monthly_income'] + 1)
    
    X = data.drop("default_status", axis=1)
    y = data["default_status"]

    # 2. Categorical Encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # 3. Sanitize column names for XGBoost
    X.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X.columns]

    # 4. Handle missing values
    X = X.fillna(X.mean())

    return X, y

def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")
    
    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### Assessing creditworthiness through advanced gradient boosting.")

    menu = ["New Application", "Dashboard", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    data = load_data()
    X, y = preprocess(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    @st.cache_resource
    def get_trained_models(_X_train, _y_train):
        rf = train_baseline_model(_X_train, _y_train)
        xgb = train_advanced_model(_X_train, _y_train)
        return rf, xgb

    rf_model, xgb_model = get_trained_models(X_train, y_train)

    if choice == "New Application":
        st.subheader("Manual Loan Application")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value="Jane Doe")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            income = st.number_input("Monthly Income (₹)", min_value=0, value=50000)
        
        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0, value=2000)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 90)

        if st.button("Analyze Risk"):
            with st.spinner('XGBoost Deep Scanning...'):
                time.sleep(0.5)

                # Construct input with same feature engineering as training
                input_data = pd.DataFrame({
                    "monthly_income": [income],
                    "utility_bill_average": [utility_bill],
                    "repayment_history_pct": [repayment_history],
                    "bill_to_income_ratio": [utility_bill / (income + 1)],
                    "gender_Male": [1],
                    "gender_Non-Binary": [0],
                    "employment_length_1-3 years": [0],
                    "employment_length_4-7 years": [0],
                    "employment_length_8+ years": [0],
                })

                input_data = input_data.reindex(columns=X.columns, fill_value=0)

                prediction = xgb_model.predict(input_data)[0]
                prob = xgb_model.predict_proba(input_data)[0][1]
                risk_level = "High" if prediction == 1 else "Low"

                st.success(f"Analysis Complete for {name}")
                m1, m2 = st.columns(2)
                m1.metric(label="Predicted Risk", value=risk_level)
                m2.metric(label="Default Probability", value=f"{prob*100:.1f}%")

                st.write("---")
                st.write("### 📊 Accuracy Benchmarking")
                rf_res = evaluate_model(rf_model, X_test, y_test)
                xgb_res = evaluate_advanced_model(xgb_model, X_test, y_test)

                c1, c2 = st.columns(2)
                with c1:
                    st.info("**RF Baseline**")
                    st.write(f"Accuracy: {rf_res[0]:.2f}")
                with c2:
                    st.success("**XGBoost Hyper-Tuned**")
                    st.write(f"Accuracy: {xgb_res['accuracy']:.2f}")
                    st.write(f"Precision: {xgb_res['precision']:.2f}")

    elif choice == "Dashboard":
        st.subheader("Model Performance Dashboard")
        rf_res = evaluate_model(rf_model, X_test, y_test)
        xgb_res = evaluate_advanced_model(xgb_model, X_test, y_test)

        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall"],
            "RF Baseline": [rf_res[0], rf_res[1], rf_res[2]],
            "XGBoost Advanced": [xgb_res['accuracy'], xgb_res['precision'], xgb_res['recall']]
        })
        
        st.table(comparison_df)
        
        col1, col2 = st.columns(2)
        col1.metric("Area Under Curve (AUC)", f"{xgb_res['auc']:.4f}")
        col2.metric("False Positives", xgb_res['false_positives'], delta_color="inverse")
        st.bar_chart(comparison_df.set_index("Metric"))

    elif choice == "Audit Logs":
        st.subheader("Audit Logs")
        st.info("No logs available yet.")

if __name__ == '__main__':
    main()