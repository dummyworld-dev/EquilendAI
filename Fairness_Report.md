## EquiLend AI – Fairness & Explainability Report

### 1. Overview

This document summarizes the current state of model performance, fairness analysis, and explainability tooling for the EquiLend AI credit risk models. It focuses primarily on the Random Forest and XGBoost classifiers trained on the synthetic `equilend_mock_data.csv` dataset.

### 2. Model Performance (Summary)

Models are trained using:

- **Data cleaning**: `src/preprocessing/data_cleaning.py` (type coercion, missing value imputation, outlier clipping).
- **Feature encoding**: `src/preprocessing/feature_encoding.py` (one‑hot encoding of categorical variables).
- **Scaling**: `src/preprocessing/scaling.py` (numeric scaling for tabular models).
- **Models**:
  - Random Forest: `src/models/train_random_forest.py`
  - XGBoost (with hyperparameter tuning): `src/models/train_xgboost.py`

Key evaluation metrics (computed in the training scripts and `src/evaluation/model_evaluation.py`):

- **Accuracy**: overall fraction of correctly classified cases.
- **ROC AUC**: ability to discriminate between default and non‑default across thresholds.
- **F1 score**: balance between precision and recall for the default class.

These metrics are printed to the console during training and can be persisted or logged as needed. XGBoost additionally uses cross‑validated ROC AUC during hyperparameter search (`RandomizedSearchCV` with `StratifiedKFold`) to select the best model.

### 3. Fairness Analysis

Fairness checks are implemented in `src/evaluation/fairness.py`.

- **Per‑attribute analysis**: `check_fairness_for_attribute` computes, for each group of a protected attribute (e.g., age band, state):
  - Predicted positive rate (fraction predicted default = 1).
  - True positive rate (observed default rate).
  - Maximum inter‑group differences in these rates.
- **Multi‑attribute analysis**: `check_model_fairness` runs these checks over:
  - **Age** (bucketed into `<25`, `25–34`, `35–44`, `45–54`, `55+`).
  - **State** (categorical field, when present).
- **Bias flagging**:
  - If the maximum difference in group‑level predicted or true default rates exceeds a configurable threshold (default **10 percentage points**), the attribute’s `bias_flag` is set to `true`.
  - An overall `overall_bias_detected` flag is raised if any monitored attribute is flagged.

These outputs can be surfaced in dashboards or logs and used to trigger alerts when fairness thresholds are breached.

### 4. SHAP Explainability

SHAP‑based explainability for tree‑based models is implemented in `src/evaluation/shap_analysis.py` and integrated into the Streamlit app (`src/app.py`):

- **Global explanations**:
  - `shap_feature_importance_bar_streamlit` renders a bar chart of mean absolute SHAP values, highlighting globally important features.
  - `shap_summary_plot_streamlit` (available for use) shows distributional impacts of each feature across the dataset.
- **Local (per‑prediction) explanations**:
  - `shap_single_prediction_force_plot_streamlit` displays a SHAP force plot for an individual prediction, showing how each feature pushed the default risk up or down relative to the baseline.
- **Streamlit integration**:
  - In the “New Application” flow, once a prediction is made, an **“Explain this prediction (SHAP)”** expander shows:
    - A SHAP feature‑importance bar plot for that specific input.
    - A force plot explaining the individual decision.

This combination of global and local views helps auditors and users understand *why* the model produced a given risk score.

### 5. Bias Mitigation Strategies

Current steps taken to reduce bias and support fair decisions include:

- **Data preprocessing**:
  - Robust handling of missing values (median imputation for numeric features, most‑frequent for categorical), preventing spurious correlations from NaNs or encoding artifacts.
  - Outlier clipping on key numeric features to reduce the influence of extreme synthetic values.
- **Model design and training**:
  - Use of **SMOTE** (optional) in `train_xgboost.py` to address class imbalance, reducing bias toward the majority (non‑default) class.
  - Cross‑validated hyperparameter tuning (StratifiedKFold + RandomizedSearchCV) for XGBoost to avoid overfitting specific subpopulations.
- **Post‑hoc fairness checks**:
  - Automated fairness reports via `check_model_fairness`, which monitor differences in predicted and true default rates across age bands and states and raise a bias flag when thresholds are exceeded.
- **Explainability‑driven review**:
  - SHAP plots are used to inspect whether sensitive or proxy attributes systematically drive predictions, allowing manual review and potential feature engineering (e.g., dropping or transforming problematic features) in future iterations.

### 6. Next Steps

- Persist fairness and performance metrics (including SHAP summaries) to disk or a monitoring system after each training run.
- Extend fairness checks to additional protected attributes (e.g., gender) where appropriate for the use case and regulations.
- Implement automated alerts in CI/CD or a monitoring dashboard whenever `overall_bias_detected` is `true` or metric thresholds regress.


