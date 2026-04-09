"""
Task 13 - Threshold Optimizer

Tools for:
- sweeping thresholds for audit/visualization
- inspecting metrics at a specific threshold
- selecting an empirical operating threshold from the precision-recall curve
  while respecting business approval and risk constraints
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve


@dataclass
class ThresholdCandidate:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    approval_rate: float
    tp: int
    tn: int
    fp: int
    fn: int


def _validate_binary_inputs(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate threshold-optimizer inputs and coerce them into numpy arrays."""
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)

    if y_true_arr.size == 0 or y_prob_arr.size == 0:
        raise ValueError("y_true and y_prob must not be empty.")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same number of rows.")
    if np.isnan(y_prob_arr).any():
        raise ValueError("y_prob contains NaN values.")
    if np.any((y_prob_arr < 0.0) | (y_prob_arr > 1.0)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    unique_labels = set(np.unique(y_true_arr).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError("y_true must be a binary label array containing only 0 and 1.")

    return y_true_arr, y_prob_arr


def _compute_candidate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> ThresholdCandidate:
    """Compute a full metric bundle for one threshold value."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(y_true)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    approval_rate = float((y_pred == 0).mean())

    return ThresholdCandidate(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        specificity=float(specificity),
        approval_rate=approval_rate,
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
    )


def sweep_thresholds(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Evaluate the model across thresholds for dashboards and audits.

    Returns a DataFrame with one row per threshold and lender-facing metrics.
    """
    if step <= 0 or step >= 1:
        raise ValueError("step must be greater than 0 and less than 1.")

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    thresholds = np.round(np.arange(step, 1.0, step), 4)
    rows = [
        asdict(_compute_candidate(y_true_arr, y_prob_arr, float(threshold)))
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def get_metrics_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float,
) -> dict[str, Any]:
    """Return a complete metric dictionary for one explicit threshold."""
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be greater than 0 and less than 1.")

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    candidate = _compute_candidate(y_true_arr, y_prob_arr, float(threshold))
    payload = asdict(candidate)
    payload["confusion_matrix"] = np.array(
        [[candidate.tn, candidate.fp], [candidate.fn, candidate.tp]]
    )
    return payload


def optimize_threshold_from_pr_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    min_precision: float = 0.60,
    min_recall: float = 0.20,
    min_approval_rate: float = 0.20,
) -> dict[str, Any]:
    """
    Empirically select a threshold from the precision-recall curve.

    The optimizer keeps only thresholds that satisfy minimum risk and approval
    constraints, then chooses the candidate with the best F1 score. If no point
    satisfies the constraints, it falls back to the globally best F1 threshold.
    """
    for label, value in {
        "min_precision": min_precision,
        "min_recall": min_recall,
        "min_approval_rate": min_approval_rate,
    }.items():
        if value < 0 or value > 1:
            raise ValueError(f"{label} must be between 0 and 1.")

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)

    if thresholds.size == 0:
        return {
            "objective": "fallback_default_threshold",
            **get_metrics_at_threshold(y_true_arr, y_prob_arr, 0.5),
        }

    candidates = []
    for idx, threshold in enumerate(thresholds):
        candidate = _compute_candidate(y_true_arr, y_prob_arr, float(threshold))
        # precision_recall_curve returns one extra point, so align to threshold index.
        candidate.precision = float(precision[idx])
        candidate.recall = float(recall[idx])
        candidate.f1 = (
            2 * candidate.precision * candidate.recall / (candidate.precision + candidate.recall)
            if (candidate.precision + candidate.recall) > 0
            else 0.0
        )
        candidates.append(candidate)

    feasible = [
        candidate
        for candidate in candidates
        if candidate.precision >= min_precision
        and candidate.recall >= min_recall
        and candidate.approval_rate >= min_approval_rate
    ]

    objective = "max_f1_subject_to_precision_recall_approval_constraints"
    pool = feasible
    if not pool:
        pool = candidates
        objective = "max_f1_without_feasible_constraints"

    best = max(
        pool,
        key=lambda candidate: (
            candidate.f1,
            candidate.precision,
            candidate.recall,
            candidate.approval_rate,
        ),
    )

    result = asdict(best)
    result["objective"] = objective
    result["min_precision"] = float(min_precision)
    result["min_recall"] = float(min_recall)
    result["min_approval_rate"] = float(min_approval_rate)
    result["evaluated_points"] = [asdict(candidate) for candidate in candidates]
    return result


OBJECTIVE_DESCRIPTIONS = {
    "f1": "Maximize F1 Score - balanced precision/recall trade-off",
    "precision": "Maximize Precision - minimize false approvals for stricter risk control",
    "recall": "Maximize Recall - catch as many defaulters as possible",
    "balanced": "Balanced (G-Mean) - geometric mean of sensitivity and specificity",
    "profit": "Maximize Profit - weighted business-value objective",
    "business": "Business-recommended threshold balancing risk capture and approval goals",
    "pr_curve": "Precision-Recall optimizer with business approval constraints",
}


def get_business_recommended_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """
    Return the default operating threshold for lending decisions.

    Business logic:
    - positive class = default risk = deny
    - keep recall reasonably high to catch defaulters
    - keep precision healthy to avoid over-denying good borrowers
    - preserve a meaningful approval rate so the lender still books loans
    """
    return optimize_threshold_from_pr_curve(
        y_true,
        y_prob,
        min_precision=0.55,
        min_recall=0.70,
        min_approval_rate=0.35,
    )


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    objective: str = "f1",
) -> dict[str, Any]:
    """Find the threshold that best matches the requested business objective."""
    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)

    if objective == "business":
        return get_business_recommended_threshold(y_true_arr, y_prob_arr)
    if objective == "pr_curve":
        return optimize_threshold_from_pr_curve(y_true_arr, y_prob_arr)

    df = sweep_thresholds(y_true_arr, y_prob_arr).copy()

    if objective == "f1":
        best_idx = df["f1"].idxmax()
    elif objective == "precision":
        best_idx = df["precision"].idxmax()
    elif objective == "recall":
        best_idx = df["recall"].idxmax()
    elif objective == "balanced":
        df["gmean"] = np.sqrt(df["recall"] * df["specificity"])
        best_idx = df["gmean"].idxmax()
    elif objective == "profit":
        # Business mapping for this model:
        # tn = good borrower approved correctly           -> +1.0
        # fp = good borrower denied incorrectly          -> -0.5
        # fn = defaulter approved incorrectly            -> -5.0
        # tp = defaulter denied correctly                -> +0.2
        df["profit_score"] = (
            df["tn"] * 1.0
            + df["fp"] * (-0.5)
            + df["tp"] * 0.2
            + df["fn"] * (-5.0)
        )
        best_idx = df["profit_score"].idxmax()
    else:
        raise ValueError(
            f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVE_DESCRIPTIONS)}"
        )

    row = df.loc[best_idx]
    return {
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "accuracy": float(row["accuracy"]),
        "specificity": float(row["specificity"]),
        "approval_rate": float(row["approval_rate"]),
        "tp": int(row["tp"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "objective": objective,
    }
