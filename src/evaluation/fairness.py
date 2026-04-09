from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def _group_positive_rate(y_true: pd.Series, y_pred: pd.Series, group: pd.Series) -> pd.DataFrame:
    """
    Compute positive prediction rates and outcome rates per group.

    Positive label is assumed to be 1 (e.g., default = 1).
    """
    df = pd.DataFrame(
        {
            "y_true": y_true.values,
            "y_pred": y_pred.values,
            "group": group.values,
        }
    )

    grouped = df.groupby("group")
    metrics = grouped.agg(
        positive_rate_pred=("y_pred", lambda x: (x == 1).mean()),
        positive_rate_true=("y_true", lambda x: (x == 1).mean()),
    )
    return metrics


def check_fairness_for_attribute(
    y_true: pd.Series,
    y_pred: pd.Series,
    protected_attr: pd.Series,
    attr_name: str,
    max_diff: float = 0.10,
) -> Dict[str, Any]:
    """
    Run simple fairness checks for a single protected attribute.

    This function checks for differences in positive prediction rates and
    outcome rates across groups of the protected attribute. It raises a flag
    if the maximum absolute difference exceeds a configurable threshold.

    Args:
        y_true (pd.Series): Ground-truth labels (0/1).
        y_pred (pd.Series): Predicted labels (0/1).
        protected_attr (pd.Series): Series representing the protected attribute
            (e.g., age bucket, state, gender).
        attr_name (str): Name of the protected attribute for reporting.
        max_diff (float): Maximum allowed absolute difference in group-level
            rates before flagging potential bias.

    Returns:
        Dict[str, Any]: Summary including per-group metrics and bias flags.
    """
    metrics = _group_positive_rate(y_true, y_pred, protected_attr)

    # Differences across groups
    pred_rates = metrics["positive_rate_pred"]
    true_rates = metrics["positive_rate_true"]

    pred_diff = pred_rates.max() - pred_rates.min()
    true_diff = true_rates.max() - true_rates.min()

    bias_flag = (pred_diff > max_diff) or (true_diff > max_diff)

    result = {
        "attribute": attr_name,
        "group_metrics": metrics.to_dict(orient="index"),
        "max_pred_rate_diff": float(pred_diff),
        "max_outcome_rate_diff": float(true_diff),
        "bias_flag": bool(bias_flag),
        "threshold": float(max_diff),
    }

    return result


def check_model_fairness(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    age_col: Optional[str] = None,
    state_col: Optional[str] = None,
    max_diff: float = 0.10,
) -> Dict[str, Any]:
    """
    High-level fairness checks for common protected attributes such as age and state.

    This helper assumes that the evaluation dataframe contains the protected
    attributes you wish to analyze. It is robust to missing columns: if a
    column is not present, that check is skipped.

    Typical usage:
        fairness_results = check_model_fairness(df_eval, y_true, y_pred,
                                                age_col=\"age\", state_col=\"state\")

    Args:
        df (pd.DataFrame): Evaluation dataframe with protected attributes.
        y_true (pd.Series): Ground-truth labels.
        y_pred (pd.Series): Predicted labels.
        age_col (Optional[str]): Column name for age. If provided, will be
            bucketed into age bands before analysis.
        state_col (Optional[str]): Column name for state (categorical).
        max_diff (float): Threshold for allowable inter-group rate differences.

    Returns:
        Dict[str, Any]: Combined fairness report with per-attribute summaries
        and an overall bias flag.
    """
    reports = {}

    # Age-based fairness: bucket into coarse age bands for stability.
    if age_col and age_col in df.columns:
        age = pd.to_numeric(df[age_col], errors="coerce")
        age_bins = [0, 24, 34, 44, 54, 200]
        age_labels = ["<25", "25-34", "35-44", "45-54", "55+"]
        age_groups = pd.cut(age, bins=age_bins, labels=age_labels, right=True, include_lowest=True)

        reports["age"] = check_fairness_for_attribute(
            y_true=y_true,
            y_pred=y_pred,
            protected_attr=age_groups,
            attr_name="age",
            max_diff=max_diff,
        )

    # State-based fairness: directly use the categorical state codes.
    if state_col and state_col in df.columns:
        state_series = df[state_col].astype("category")
        reports["state"] = check_fairness_for_attribute(
            y_true=y_true,
            y_pred=y_pred,
            protected_attr=state_series,
            attr_name="state",
            max_diff=max_diff,
        )

    # Aggregate overall bias flag across all checked attributes.
    any_bias = any(report["bias_flag"] for report in reports.values())

    return {
        "attributes": reports,
        "overall_bias_detected": any_bias,
        "threshold": max_diff,
    }


