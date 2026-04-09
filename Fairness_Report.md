## EquiLend AI – Fairness & Explainability Report

### 1) Final Model Summary

- **Production model**: XGBoost with SMOTE balancing and hyperparameter tuning.
- **Training module**: `src/models/xgboost_model.py`
- **Saved artifact**: `models/xgboost_model.joblib`
- **Dataset**: `data/equilend_mock_data.csv`
- **Final test AUC**: **0.8445**
- **Best parameters**:
  - `subsample: 0.8`
  - `n_estimators: 200`
  - `max_depth: 6`
  - `learning_rate: 0.1`

### 2) Preprocessing Pipeline Used

The model uses existing preprocessing modules without breaking prior steps:

- `src/preprocessing/data_cleaning.py`
  - numeric coercion
  - missing value handling
  - outlier clipping
- `src/preprocessing/feature_encoding.py`
  - one-hot encoding for categorical features
- `src/models/xgboost_model.py`
  - feature-name sanitization for XGBoost compatibility
  - scaler fit on train split only

### 3) Explainability (SHAP)

Implemented in `src/evaluation/shap_analysis.py` and integrated in `src/app.py`.

- **Global view**: SHAP feature importance bar.
- **Local view**: SHAP force plot for each user prediction.
- **UI behavior added**:
  - “Explain this prediction (SHAP)” in New Application page.
  - Income prioritized at top ordering in SHAP summary/bar displays.
  - For black-swan-triggered cases, SHAP contributions are intentionally squashed near zero for safety-mode presentation.

### 4) Fairness / Bias Detection

- Core fairness logic: `src/evaluation/fairness.py`
- Runner script: `src/evaluation/run_bias_detection.py`
- Output report: `reports/fairness_report.json`

Checks performed:

- Group-level predicted-positive and observed-positive rates.
- Age-band and state-group fairness comparisons.
- Bias flag if max inter-group difference exceeds threshold (`max_diff`, default 0.10).
- Overall flag: `overall_bias_detected`.

### 5) Black-Swan Safety Mitigation

Added a **Zero-Trust Black-Swan Guard** in `src/app.py`:

- Force conservative outcome (`High Risk`) for extreme-risk signals.
- Trigger examples:
  - `monthly_income <= 0`
  - `repayment_history_pct <= 5`
  - `utility_bill_average / monthly_income >= 0.5`
- Records guard state in decision payload:
  - `zero_trust_triggered`
  - `zero_trust_reason`

This reduces catastrophic false approvals during out-of-distribution or stress scenarios.

### 6) Streamlit Evidence Available

`src/app.py` now shows:

- Predicted default probability
- Model test AUC
- Best hyperparameters
- Feature list used by model
- SHAP explanation (local + importance)
- Fairness report snapshot (if generated)
- Recent persisted decisions (if MongoDB configured)

### 7) Repro Steps

```bash
python scripts/generate_data.py
python src/models/xgboost_model.py
python -m streamlit run src/app.py
python -m src.evaluation.run_bias_detection
```

### 8) Conclusion

EquiLend AI now uses a tuned, explainable XGBoost model with fairness checks and black-swan safeguards.  
The pipeline is reproducible, the model is exported with joblib, and all critical outputs (AUC, SHAP, fairness) are visible in both scripts and dashboard.


