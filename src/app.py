import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# Model and Evaluation imports
from models.train_rf import train_baseline_model, evaluate_model
from sklearn.model_selection import train_test_split
from evaluation.explainer import generate_shap_explanation  # Integrated SHAP logic

# --- GLOBAL CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Verified working paths
DATA_PATH = r"C:\Users\Purva\OneDrive\Desktop\innovation_test\EquilendAI\EquilendAI\src\scripts\data\equilend_mock_data.csv"
ALT_DATA_PATH = r"C:\Users\Purva\OneDrive\Desktop\innovation_test\EquilendAI\EquilendAI\scripts\data\equilend_mock_data.csv"

def load_data():
    """Tries your verified absolute paths to load the dataset."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    elif os.path.exists(ALT_DATA_PATH):
        return pd.read_csv(ALT_DATA_PATH)
    else:
        st.error("🚨 **Data file not found!**")
        return pd.DataFrame()

def preprocess(data):
    """Cleans data and adds feature engineering for the AI Engine."""
    if data.empty:
        return data, None
    
    # Handle negative values & Fill missing data
    data['monthly_income'] = data['monthly_income'].clip(lower=0)
    data = data.fillna(data.mean(numeric_only=True))
    
    # Feature Engineering (Ratios) - Using +1 to prevent DivisionByZero
    data['bill_to_income_ratio'] = data['utility_bill_average'] / (data['monthly_income'] + 1)
    
    X = data.drop("default_status", axis=1)
    y = data["default_status"]

    # Categorical Encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Sanitization for column names
    X.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X.columns]

    return X, y

def main():
    st.set_page_config(page_title="EquiLend AI - Decision Explainer", layout="wide")
    
    st.title("⚖️ EquiLend AI: Credit Scoring")
    st.markdown("### Transparent AI for Alternative Data Lending")

    st.sidebar.title("Control Panel")
    menu = ["New Application", "Dashboard"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Load and process data
    data = load_data()
    if data.empty:
        return

    X, y = preprocess(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    @st.cache_resource
    def get_trained_assets(_X_train, _y_train):
        # Using Random Forest for near-instant execution
        rf = train_baseline_model(_X_train, _y_train)
        return rf

    rf_model = get_trained_assets(X_train, y_train)

    if choice == "New Application":
        st.subheader("Manual Loan Application")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value="Jane Doe")
            age = st.number_input("Age", min_value=18, max_value=120, value=30)
            income = st.number_input("Monthly Income (₹)", min_value=0, value=50000)
        
        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0, value=2000)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 90)

        if st.button("Analyze Risk"):
            with st.spinner('AI Decision Engine Processing...'):
                time.sleep(0.3)

                # Prepare input
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

                # Predict
                prediction = rf_model.predict(input_data)[0]
                prob = rf_model.predict_proba(input_data)[0][1]
                risk_level = "High" if prediction == 1 else "Low"

                st.success(f"Analysis Complete for {name}")
                m1, m2 = st.columns(2)
                m1.metric("Predicted Risk", risk_level)
                m2.metric("Default Probability", f"{prob*100:.1f}%")

                # --- SHAP EXPLAINER INTEGRATION ---
                with st.expander("🔍 Why this decision? (XAI Breakdown)"):
                    st.write("Generating factor analysis for loan officer...")
                    plot_path = generate_shap_explanation(rf_model, input_data)
                    st.image(plot_path, caption="SHAP Waterfall Plot: Contribution of each feature to the final decision.")
                    st.info("**Red** values increase risk, **Blue** values decrease risk.")

                if risk_level == "High":
                    st.error("🚨 Flagged for review based on risk profile.")
                else:
                    st.balloons()
                    st.success("✅ Application approved.")

    elif choice == "Dashboard":
        st.subheader("Model Performance & Metrics")
        rf_res = evaluate_model(rf_model, X_test, y_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{rf_res[0]:.2f}")
        c2.metric("Precision", f"{rf_res[1]:.2f}")
        c3.metric("Recall", f"{rf_res[2]:.2f}")
        
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall"],
            "Score": [rf_res[0], rf_res[1], rf_res[2]]
        })
        st.bar_chart(metrics_df.set_index("Metric"))

if __name__ == '__main__':
    main()