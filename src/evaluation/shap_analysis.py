from typing import Optional

import numpy as np
import pandas as pd
import shap
import streamlit as st


def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for a fitted tree-based model.

    This wrapper uses shap.TreeExplainer, which works well with tree models
    such as RandomForest, XGBoost, LightGBM, etc.

    Args:
        model: Trained model (e.g., RandomForestClassifier, XGBClassifier).
        X (pd.DataFrame): Feature matrix used for explanation.

    Returns:
        explainer: Fitted SHAP explainer.
        shap_values: SHAP values for the dataset X.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def shap_summary_plot_streamlit(
    explainer,
    shap_values,
    X: pd.DataFrame,
    class_index: Optional[int] = None,
    title: str = "SHAP Summary Plot",
):
    """
    Render a SHAP summary plot inside Streamlit.

    For binary classifiers, shap_values is often a list where index 1
    corresponds to the positive class; this can be controlled via class_index.
    """
    st.subheader(title)

    # Handle binary vs multi-class SHAP value structures.
    values_to_plot = shap_values
    if isinstance(shap_values, list):
        # Default to positive class if not specified.
        if class_index is None:
            class_index = 1 if len(shap_values) > 1 else 0
        values_to_plot = shap_values[class_index]

    # Streamlit-compatible SHAP plot.
    shap.summary_plot(values_to_plot, X, show=False)
    st.pyplot(bbox_inches="tight", clear_figure=True)


def shap_feature_importance_bar_streamlit(
    shap_values,
    X: pd.DataFrame,
    class_index: Optional[int] = None,
    title: str = "SHAP Feature Importance",
):
    """
    Display a bar plot of mean absolute SHAP values (global importance)
    inside Streamlit.
    """
    st.subheader(title)

    values_to_plot = shap_values
    if isinstance(shap_values, list):
        if class_index is None:
            class_index = 1 if len(shap_values) > 1 else 0
        values_to_plot = shap_values[class_index]

    shap.summary_plot(
        values_to_plot,
        X,
        plot_type="bar",
        show=False,
    )
    st.pyplot(bbox_inches="tight", clear_figure=True)


def shap_single_prediction_force_plot_streamlit(
    explainer,
    shap_values,
    X: pd.DataFrame,
    instance_index: int,
    class_index: Optional[int] = None,
    title: str = "SHAP Force Plot (Single Prediction)",
):
    """
    Show a force plot explaining a single prediction, embedded in Streamlit.

    Args:
        explainer: SHAP explainer returned by compute_shap_values.
        shap_values: SHAP values as returned by compute_shap_values.
        X (pd.DataFrame): Feature matrix.
        instance_index (int): Row index of the instance to explain.
        class_index (Optional[int]): For multi-class models, which class to view.
        title (str): Section title displayed in Streamlit.
    """
    st.subheader(title)

    # Safeguard the index
    instance_index = int(np.clip(instance_index, 0, len(X) - 1))

    x_instance = X.iloc[[instance_index]]

    values_to_use = shap_values
    if isinstance(shap_values, list):
        if class_index is None:
            class_index = 1 if len(shap_values) > 1 else 0
        values_to_use = shap_values[class_index]

    # Use the JS-based force plot and render it in Streamlit via st.components.
    # Note: Streamlit requires the HTML to be passed as a string.
    force_plot_html = shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[class_index],
        values_to_use[instance_index],
        x_instance,
        matplotlib=False,
        show=False,
    )

    st.components.v1.html(
        shap.getjs() + force_plot_html.html(),
        height=300,
    )

