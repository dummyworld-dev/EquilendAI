import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.preprocessing import StandardScaler

from src.preprocessing.data_cleaning import load_and_clean
from src.preprocessing.feature_encoding import encode_categorical_features
from src.evaluation.shap_analysis import (
    compute_shap_values,
    shap_feature_importance_bar_streamlit,
    shap_single_prediction_force_plot_streamlit,
)

# Earthy / Professional Theme Colors
PRIMARY_COLOR = "#2E7D32"  # Forest Green
ACCENT_COLOR = "#5D4037"  # Soil Brown

MODEL_PATH = "models/xgboost_model.joblib"
DATA_PATH = "data/equilend_mock_data.csv"


@st.cache_resource
def load_model_and_preprocessor():
    """
    Load the trained ML model and build a preprocessing pipeline that
    mirrors the training-time transformations.

    The scaler is fit on the same cleaned + encoded training data so that
    incoming user inputs are transformed consistently.
    """
    try:
        # Load and preprocess the historical dataset.
        df = load_and_clean(DATA_PATH)
        df = encode_categorical_features(df)

        # Features used for training exclude the target.
        feature_cols = [c for c in df.columns if c != "default_status"]
        X = df[feature_cols]

        scaler = StandardScaler()
        scaler.fit(X)

        model = joblib.load(MODEL_PATH)
    except Exception as e:
        # If anything fails (e.g., model file missing), surface the error in the UI.
        st.error(f"Failed to load model or preprocessing pipeline: {e}")
        return None, None, None

    return model, scaler, feature_cols

def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")
    
    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### Assessing creditworthiness through alternative data.")

    # Sidebar for Navigation
    menu = ["New Application", "Dashboard", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "New Application":
        st.subheader("Manual Loan Application")

        model, scaler, feature_cols = load_model_and_preprocessor()
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            income = st.number_input("Monthly Income (₹)", min_value=0)
        
        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)

        if model is None or scaler is None or feature_cols is None:
            st.warning("Model not available. Please train the model before running predictions.")
            return

        if st.button("Analyze Risk"):
            # LOGICAL ERRORS 1, 2, and 3 are hidden in this block
            with st.spinner('AI Model Calculating...'):
                time.sleep(1) # Simulate processing
                
                # Build a single-row dataframe from user input using the same
                # schema as the training data (minus the target column).
                input_data = {
                    "gender": "Male",  # Default or extend UI later if needed
                    "monthly_income": income,
                    "utility_bill_average": utility_bill,
                    "repayment_history_pct": repayment_history,
                    "employment_length": "1-3 years",  # Default category
                }
                user_df = pd.DataFrame([input_data])

                # Apply the same encoding as during training.
                user_encoded = encode_categorical_features(user_df)

                # Ensure all training-time feature columns are present for the model.
                for col in feature_cols:
                    if col not in user_encoded.columns:
                        user_encoded[col] = 0
                user_encoded = user_encoded[feature_cols]

                # Scale numeric features using the pre-fitted scaler.
                user_scaled = scaler.transform(user_encoded)

                # Predict default probability using the trained model.
                default_proba = model.predict_proba(user_scaled)[0, 1]

                # Higher probability of default implies higher risk.
                risk_level = "High" if default_proba >= 0.5 else "Low"

                st.success(f"Analysis Complete for {name}")
                st.metric(
                    label="Predicted Default Probability",
                    value=f"{default_proba:.2%}",
                )
                st.write(f"Recommended Decision: **{risk_level} Risk**")

                # SHAP-based explanation for this specific prediction.
                with st.expander("Explain this prediction (SHAP)"):
                    # Build a DataFrame matching the model's feature space for SHAP.
                    user_scaled_df = pd.DataFrame(user_scaled, columns=feature_cols)

                    explainer, shap_values = compute_shap_values(model, user_scaled_df)

                    # Global-style bar plot for this single prediction (mean |SHAP|).
                    shap_feature_importance_bar_streamlit(
                        shap_values,
                        user_scaled_df,
                        title="Feature Importance (SHAP values)",
                    )

                    # Local force plot showing how each feature influenced this decision.
                    shap_single_prediction_force_plot_streamlit(
                        explainer,
                        shap_values,
                        user_scaled_df,
                        instance_index=0,
                        title="Contribution of Each Feature",
                    )

                # LOGICAL ERROR 4: Data is not saved anywhere yet

    elif choice == "Dashboard":
        st.subheader("Lender Rules Engine Overview")
        # Placeholder for visual charts
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Approved', 'Rejected', 'Pending'])
        st.line_chart(chart_data)

if __name__ == '__main__':
    main()
