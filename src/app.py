import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json

from src.preprocessing.data_cleaning import load_and_clean
from src.preprocessing.feature_encoding import encode_categorical_features
from src.evaluation.shap_analysis import (
    compute_shap_values,
    shap_feature_importance_bar_streamlit,
    shap_single_prediction_force_plot_streamlit,
    squash_shap_values_near_zero,
)
from src.models.xgboost_model import (
    load_artifact,
    predict_default_probability,
    sanitize_feature_names,
)
from src.data_ingestion.mongodb import save_decision, fetch_recent_decisions

# Earthy / Professional Theme Colors
PRIMARY_COLOR = "#2E7D32"  # Forest Green
ACCENT_COLOR = "#5D4037"  # Soil Brown

MODEL_PATH = "models/xgboost_model.joblib"
DATA_PATH = "data/equilend_mock_data.csv"


def evaluate_zero_trust_guard(
    monthly_income: float,
    utility_bill_average: float,
    repayment_history_pct: int,
) -> tuple[bool, str]:
    """
    Black-swan safety gate ("zero-trust" mode).

    If signals look extreme or unstable, we force a conservative decision
    regardless of model score to reduce catastrophic approval risk.
    """
    if monthly_income <= 0:
        return True, "Income is zero or invalid."
    if repayment_history_pct <= 5:
        return True, "Repayment consistency is critically low."
    if monthly_income > 0 and (utility_bill_average / monthly_income) >= 0.5:
        return True, "Utility-bill-to-income ratio indicates severe stress."
    return False, ""


@st.cache_resource
def load_model_and_preprocessor():
    """
    Load the trained XGBoost artifact (model + scaler + feature columns).

    The artifact is produced by `src/models/xgboost_model.py` and ensures
    inference uses the exact same preprocessing metadata as training.
    """
    try:
        artifact = load_artifact(MODEL_PATH)
    except Exception as e:
        # If anything fails (e.g., model file missing), surface the error in the UI.
        st.error(f"Failed to load model artifact: {e}")
        return None

    return artifact


def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")

    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### Assessing creditworthiness through alternative data.")

    # Sidebar for Navigation
    menu = ["New Application", "Dashboard", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "New Application":
        st.subheader("Manual Loan Application")

        artifact = load_model_and_preprocessor()

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            income = st.number_input("Monthly Income (₹)", min_value=0)

        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
            employment_length = st.selectbox(
                "Employment Length",
                ["< 1 year", "1-3 years", "4-7 years", "8+ years"],
            )

        if artifact is None:
            st.warning("Model not available. Please train the model before running predictions.")
            return

        if st.button("Analyze Risk"):
            # Age guard: applicants under 18 should not be scored.
            if age < 18:
                st.error("Applicants must be at least 18 years old for credit scoring.")
                return

            with st.spinner("AI Model Calculating..."):
                time.sleep(1)  # Simulate processing

                input_data = {
                    "gender": gender,
                    "monthly_income": income,
                    "utility_bill_average": utility_bill,
                    "repayment_history_pct": repayment_history,
                    "employment_length": employment_length,
                }
                default_proba = predict_default_probability(artifact, input_data)

                # Higher probability of default implies higher risk.
                risk_level = "High" if default_proba >= 0.5 else "Low"

                # Apply black-swan zero-trust safety override.
                zero_trust_block, block_reason = evaluate_zero_trust_guard(
                    monthly_income=float(income),
                    utility_bill_average=float(utility_bill),
                    repayment_history_pct=int(repayment_history),
                )
                if zero_trust_block:
                    risk_level = "High"
                    # Raise displayed probability floor so decision aligns with override.
                    default_proba = max(default_proba, 0.95)

                st.success(f"Analysis Complete for {name}")
                st.metric(label="Predicted Default Probability", value=f"{default_proba:.2%}")
                st.metric(label="Model Test AUC", value=f"{artifact.test_auc:.4f}")
                st.write(f"Recommended Decision: **{risk_level} Risk**")
                if zero_trust_block:
                    st.error(f"Zero-Trust Safety Triggered: {block_reason}")

                # Persist decision for auditability/state persistence across refreshes.
                saved = save_decision(
                    {
                        "name": name,
                        "age": int(age),
                        "gender": gender,
                        "monthly_income": float(income),
                        "utility_bill_average": float(utility_bill),
                        "repayment_history_pct": int(repayment_history),
                        "employment_length": employment_length,
                        "pred_default_proba": float(default_proba),
                        "risk_level": risk_level,
                        "zero_trust_triggered": bool(zero_trust_block),
                        "zero_trust_reason": block_reason,
                    }
                )
                if saved:
                    st.caption("Decision saved to MongoDB.")
                else:
                    st.caption(
                        "MongoDB not configured (set MONGODB_URI/MONGO_URI). "
                        "Decision was not persisted."
                    )

                # SHAP-based explanation for this specific prediction.
                with st.expander("Explain this prediction (SHAP)"):
                    user_df = pd.DataFrame([input_data])
                    user_encoded = encode_categorical_features(user_df)
                    user_encoded = sanitize_feature_names(user_encoded)
                    for col in artifact.feature_cols:
                        if col not in user_encoded.columns:
                            user_encoded[col] = 0.0
                    user_encoded = user_encoded[artifact.feature_cols]
                    user_scaled = artifact.scaler.transform(user_encoded)
                    user_scaled_df = pd.DataFrame(user_scaled, columns=artifact.feature_cols)

                    explainer, shap_values = compute_shap_values(artifact.model, user_scaled_df)

                    # Black-swan UI rule: keep SHAP contributions close to zero when
                    # zero-trust safety is triggered.
                    if zero_trust_block:
                        shap_values = squash_shap_values_near_zero(shap_values, factor=0.01)
                        st.caption("Black-swan mode active: SHAP values squashed near zero.")

                    shap_feature_importance_bar_streamlit(
                        shap_values,
                        user_scaled_df,
                        title="Feature Importance (SHAP values)",
                        income_feature="monthly_income",
                    )

                    shap_single_prediction_force_plot_streamlit(
                        explainer,
                        shap_values,
                        user_scaled_df,
                        instance_index=0,
                        title="Contribution of Each Feature",
                    )

                with st.expander("Model Details (for submission evidence)"):
                    st.write("Best hyperparameters")
                    st.json(artifact.best_params)
                    st.write("Top model metadata")
                    st.json(
                        {
                            "model_path": MODEL_PATH,
                            "test_auc": artifact.test_auc,
                            "feature_count": len(artifact.feature_cols),
                            "zero_trust_triggered": bool(zero_trust_block),
                        }
                    )

    elif choice == "Dashboard":
        st.subheader("Lender Rules Engine Overview")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3), columns=["Approved", "Rejected", "Pending"]
        )
        st.line_chart(chart_data)

        st.markdown("### Dummy Predictions (XGBoost)")
        artifact = load_model_and_preprocessor()
        if artifact is None:
            st.warning("Model not available. Train the XGBoost model to see predictions.")
        else:
            try:
                df = load_and_clean(DATA_PATH)
                sample = df.head(5).copy()
                X_raw = sample.drop(columns=["default_status"])

                X_enc = encode_categorical_features(X_raw)
                X_enc = sanitize_feature_names(X_enc)
                for col in artifact.feature_cols:
                    if col not in X_enc.columns:
                        X_enc[col] = 0.0
                X_enc = X_enc[artifact.feature_cols]
                X_scaled = artifact.scaler.transform(X_enc)

                sample["pred_default_proba"] = artifact.model.predict_proba(X_scaled)[:, 1]
                st.dataframe(
                    sample[
                        [
                            "monthly_income",
                            "utility_bill_average",
                            "repayment_history_pct",
                            "gender",
                            "employment_length",
                            "default_status",
                            "pred_default_proba",
                        ]
                    ]
                )
                st.caption(f"Model test AUC (from training artifact): {artifact.test_auc:.4f}")
                with st.expander("Model Card: Everything Visible"):
                    st.metric("AUC", f"{artifact.test_auc:.4f}")
                    st.write("Best hyperparameters")
                    st.json(artifact.best_params)
                    st.write("Training feature columns")
                    st.dataframe(pd.DataFrame({"feature_name": artifact.feature_cols}))
            except Exception as e:
                st.error(f"Failed to generate dummy predictions: {e}")

        st.markdown("### Recent Saved Decisions")
        recent = fetch_recent_decisions(limit=10)
        if recent:
            st.dataframe(pd.DataFrame(recent))
        else:
            st.info(
                "No saved decisions found (or MongoDB not configured). "
                "Set MONGODB_URI to persist decisions."
            )

        st.markdown("### Fairness Report Snapshot")
        fairness_path = "reports/fairness_report.json"
        if os.path.exists(fairness_path):
            try:
                with open(fairness_path, "r", encoding="utf-8") as f:
                    fairness_json = json.load(f)
                st.json(fairness_json)
            except Exception as e:
                st.error(f"Failed to load fairness report: {e}")
        else:
            st.info(
                "Fairness report not found yet. Run: "
                "`python -m src.evaluation.run_bias_detection`"
            )

        st.markdown("### Published Fairness Report (Markdown)")
        fairness_md_path = "Fairness_Report.md"
        if os.path.exists(fairness_md_path):
            try:
                with open(fairness_md_path, "r", encoding="utf-8") as f:
                    fairness_md = f.read()
                st.markdown(fairness_md)
                st.download_button(
                    label="Download Fairness_Report.md",
                    data=fairness_md,
                    file_name="Fairness_Report.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Failed to load Fairness_Report.md: {e}")
        else:
            st.info("Fairness_Report.md not found in project root.")


if __name__ == "__main__":
    main()
