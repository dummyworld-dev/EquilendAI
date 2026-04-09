import json
import os
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.evaluation.fairness import check_model_fairness
from src.models.xgboost_model import load_artifact, predict_xgb


def _build_bias_alerts(fairness_result: Dict[str, Any], threshold: float) -> list[Dict[str, Any]]:
    """
    Convert fairness metrics into explicit alert objects for reporting/monitoring.
    """
    alerts: list[Dict[str, Any]] = []
    for attr_name, attr_report in fairness_result.get("attributes", {}).items():
        pred_diff = float(attr_report.get("max_pred_rate_diff", 0.0))
        outcome_diff = float(attr_report.get("max_outcome_rate_diff", 0.0))
        flagged = bool(attr_report.get("bias_flag", False))
        severity = "high" if max(pred_diff, outcome_diff) >= (threshold * 1.5) else "medium"
        if flagged:
            alerts.append(
                {
                    "attribute": attr_name,
                    "severity": severity,
                    "message": (
                        f"Potential bias detected for '{attr_name}'. "
                        f"pred_rate_diff={pred_diff:.4f}, outcome_rate_diff={outcome_diff:.4f}, "
                        f"threshold={threshold:.4f}"
                    ),
                }
            )
    return alerts


def _write_markdown_report(report: Dict[str, Any], output_md_path: str) -> None:
    """
    Persist a human-readable markdown report in addition to JSON.
    """
    lines = [
        "## Bias Detection Script Report",
        "",
        f"- Overall bias detected: **{report['overall_bias_detected']}**",
        f"- Threshold: **{report['threshold']:.4f}**",
        f"- Rows evaluated: **{report['n_rows_evaluated']}**",
        f"- Positive prediction rate: **{report['pred_positive_rate']:.4f}**",
        f"- Positive outcome rate: **{report['true_positive_rate']:.4f}**",
        "",
        "### Alerts",
    ]
    if report["alerts"]:
        for alert in report["alerts"]:
            lines.append(f"- [{alert['severity'].upper()}] {alert['message']}")
    else:
        lines.append("- No bias alerts triggered.")

    lines.append("")
    lines.append("### Attribute Details")
    for attr_name, attr_report in report.get("attributes", {}).items():
        lines.append(f"- **{attr_name}**")
        lines.append(f"  - bias_flag: `{attr_report.get('bias_flag', False)}`")
        lines.append(f"  - max_pred_rate_diff: `{attr_report.get('max_pred_rate_diff', 0.0):.4f}`")
        lines.append(f"  - max_outcome_rate_diff: `{attr_report.get('max_outcome_rate_diff', 0.0):.4f}`")

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_bias_detection(
    model_path: str = "models/xgboost_model.joblib",
    data_path: str = "data/equilend_mock_data.csv",
    threshold: float = 0.10,
    output_path: str = "reports/fairness_report.json",
    output_md_path: str = "reports/fairness_report.md",
) -> Dict[str, Any]:
    """
    Load model + data, score predictions, and run fairness checks.

    Since the synthetic dataset currently lacks true `age` and `state` columns,
    this script creates deterministic placeholder groups so fairness plumbing
    can run end-to-end on dummy data.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Train model first using: python src/models/xgboost_model.py"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Generate data first using: python scripts/generate_data.py"
        )

    artifact = load_artifact(model_path)
    df = pd.read_csv(data_path)

    # If the dataset does not contain these attributes yet, create placeholders
    # to keep the bias detection path executable for challenge workflows.
    if "age" not in df.columns:
        rng = np.random.default_rng(42)
        df["age"] = rng.integers(18, 70, size=len(df))
    if "state" not in df.columns:
        rng = np.random.default_rng(42)
        df["state"] = rng.choice(["CA", "TX", "NY", "FL", "WA"], size=len(df))

    # Model features are expected to be the raw columns used during training.
    feature_candidates = [
        "gender",
        "monthly_income",
        "utility_bill_average",
        "repayment_history_pct",
        "employment_length",
    ]
    X_new = df[feature_candidates].copy()
    y_true = df["default_status"].astype(int)

    y_proba = predict_xgb(artifact, X_new)
    y_pred = (y_proba >= 0.5).astype(int)

    fairness = check_model_fairness(
        df=df,
        y_true=y_true,
        y_pred=pd.Series(y_pred),
        age_col="age",
        state_col="state",
        max_diff=threshold,
    )

    alerts = _build_bias_alerts(fairness, threshold=threshold)
    report: Dict[str, Any] = {
        **fairness,
        "alerts": alerts,
        "n_rows_evaluated": int(len(df)),
        "pred_positive_rate": float((y_pred == 1).mean()),
        "true_positive_rate": float((y_true == 1).mean()),
        "model_path": model_path,
        "data_path": data_path,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_markdown_report(report, output_md_path=output_md_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fairness and bias detection checks.")
    parser.add_argument("--model-path", default="models/xgboost_model.joblib")
    parser.add_argument("--data-path", default="data/equilend_mock_data.csv")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output-json", default="reports/fairness_report.json")
    parser.add_argument("--output-md", default="reports/fairness_report.md")
    args = parser.parse_args()

    result = run_bias_detection(
        model_path=args.model_path,
        data_path=args.data_path,
        threshold=args.threshold,
        output_path=args.output_json,
        output_md_path=args.output_md,
    )
    print("Bias detection complete.")
    print(f"Overall bias detected: {result['overall_bias_detected']}")
    print(f"Alerts raised: {len(result['alerts'])}")
    print(f"Saved JSON report to: {args.output_json}")
    print(f"Saved Markdown report to: {args.output_md}")
    print(json.dumps(result, indent=2))

