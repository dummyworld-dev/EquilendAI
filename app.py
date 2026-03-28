import streamlit as st
import pandas as pd
import numpy as np
import time

# Earthy / Professional Theme Colors
PRIMARY_COLOR = "#2E7D32" # Forest Green
ACCENT_COLOR = "#5D4037"  # Soil Brown

def main():
    st.set_page_config(page_title="EquiLend AI - Credit Scoring", layout="wide")
    
    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("### Assessing creditworthiness through alternative data.")

    # Sidebar for Navigation
    menu = ["New Application", "Dashboard", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "New Application":
        st.subheader("Manual Loan Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            income = st.number_input("Monthly Income (₹)", min_value=0)
        
        with col2:
            utility_bill = st.number_input("Average Utility Bill (₹)", min_value=0)
            repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)

        if st.button("Analyze Risk"):
            # LOGICAL ERRORS 1, 2, and 3 are hidden in this block
            with st.spinner('AI Model Calculating...'):
                time.sleep(1) # Simulate processing
                
                # Placeholder Score Logic
                base_score = (income / (utility_bill + 1)) * (repayment_history / 100)
                risk_level = "High" if base_score < 5 else "Low"
                
                st.success(f"Analysis Complete for {name}")
                st.metric(label="Calculated Risk Score", value=round(base_score, 2))
                st.write(f"Recommended Decision: **{risk_level} Risk**")
                
                # LOGICAL ERROR 4: Data is not saved anywhere yet

    elif choice == "Dashboard":
        st.subheader("Lender Rules Engine Overview")
        # Placeholder for visual charts
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Approved', 'Rejected', 'Pending'])
        st.line_chart(chart_data)

if __name__ == '__main__':
    main()
