"""
EquiLend AI — Streamlit Dashboard
Fixes applied (Task 00):
  Bug 1 — Division-by-zero: utility_bill clamped to min 1 before any division
  Bug 2 — Age guard: hard block on applicants < 18
  Bug 3 — Linear formula replaced with trained XGBoost model
  Bug 4 — State persistence: decisions saved to MongoDB (session log fallback)

New feature: Threshold Optimizer (Lender Rules Engine)
"""

import os
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# ── Constants ─────────────────────────────────────────────────────────────────
PRIMARY  = "#2E7D32"   # Forest Green
ACCENT   = "#5D4037"   # Soil Brown
ORANGE   = "#F57C00"

MODELS_DIR = os.path.join(_ROOT_DIR, "models")

# Support both data locations (generate_data.py vs upstream mock_data_generator.py)
_DATA_CANDIDATES = [
    os.path.join(_ROOT_DIR, "data",          "equilend_mock_data.csv"),
    os.path.join(_ROOT_DIR, "scripts", "data", "equilend_mock_data.csv"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_artifacts():
    """Load saved model + preprocessor + test predictions. Returns None if absent."""
    try:
        from models.train_xgb import load_artifacts
        return load_artifacts(MODELS_DIR)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _sweep_thresholds(y_test_tup, y_prob_tup):
    """Threshold sweep — result cached so plots reuse it without recomputing."""
    from evaluation.thresholds import sweep_thresholds
    return sweep_thresholds(np.array(y_test_tup), np.array(y_prob_tup))


# ── Shared training helper ────────────────────────────────────────────────────

def _do_train():
    """Train XGBoost, clear cache, rerun. Runs inside a Streamlit spinner."""
    from models.train_xgb import train_and_save
    model, pre, y_test, y_prob, auc, threshold_info = train_and_save(DATA_PATH, MODELS_DIR)
    _load_artifacts.clear()
    return auc, threshold_info


# ── Page: New Application ─────────────────────────────────────────────────────

def page_new_application():
    st.subheader("Manual Loan Application")

    col1, col2 = st.columns(2)

    with col1:
        name              = st.text_input("Full Name")
        # Bug 2 fix: min_value kept at 0 so the field accepts any input,
        # but we validate >= 18 before scoring.
        age               = st.number_input("Age", min_value=0, max_value=120, step=1)
        income            = st.number_input("Monthly Income (₹)", min_value=0, step=500)
        gender            = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])

    with col2:
        # Bug 1 fix: min_value=0 is fine for display; we clamp to 1 before use.
        utility_bill      = st.number_input("Average Utility Bill (₹)", min_value=0, step=100)
        repayment_history = st.slider("Past Repayment Consistency (%)", 0, 100, 50)
        employment_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1-3 years", "4-7 years", "8+ years"],
        )

    if st.button("Analyze Risk", type="primary"):

        # ── Bug 2: Age guard ──────────────────────────────────────────────────
        if age < 18:
            st.error("❌ Applicant must be at least 18 years old to apply.")
            return

        if not name.strip():
            st.warning("Please enter the applicant's full name.")
            return

        with st.spinner("AI Model Calculating…"):
            time.sleep(0.4)

            artifacts = _load_artifacts()

            if artifacts is None:
                # ── Bug 1 & 3 fallback (no model trained yet) ────────────────
                safe_bill  = max(utility_bill, 1)          # Bug 1 fix
                base_score = (income / safe_bill) * (repayment_history / 100)
                risk_level = "High" if base_score < 5 else "Low"
                st.warning("⚠️ ML model not trained yet. Using formula fallback.")
                st.metric("Formula Score", round(base_score, 2))
                st.write(f"Recommended Decision: **{risk_level} Risk**")
                return

            model, preprocessor, _, _, threshold_info = artifacts

            # Bug 3 fix: use the trained ML model
            input_df = pd.DataFrame([{
                "monthly_income":       income,
                "utility_bill_average": max(utility_bill, 1),   # Bug 1 fix
                "repayment_history_pct": repayment_history,
                "employment_length":    employment_length,
                "gender":               gender,
            }])
            X_proc       = preprocessor.transform(input_df)
            prob_default = float(model.predict_proba(X_proc)[0, 1])

            # Read the lender's active threshold (set in Threshold Optimizer)
            threshold = st.session_state.get(
                "thresh_slider",
                float(threshold_info.get("threshold", 0.50)),
            )
            decision  = "Deny — Default Risk" if prob_default >= threshold else "Approve"
            is_deny   = prob_default >= threshold

        st.success(f"Analysis complete for **{name}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Default Probability", f"{prob_default * 100:.1f}%")
        m2.metric("Active Threshold",    f"{threshold:.2f}")
        m3.metric("Decision",            decision)

        if is_deny:
            st.error(f"🚫 {decision}")
        else:
            st.success(f"✅ {decision}")

        # ── Bug 4 fix: persist decision to session log + MongoDB ──────────────
        record = {
            "name":                  name,
            "age":                   int(age),
            "gender":                gender,
            "monthly_income":        income,
            "utility_bill":          utility_bill,
            "repayment_history_pct": repayment_history,
            "employment_length":     employment_length,
            "prob_default":          round(prob_default, 4),
            "threshold":             threshold,
            "decision":              decision,
            "timestamp":             pd.Timestamp.now().isoformat(),
        }

        if "audit_log" not in st.session_state:
            st.session_state.audit_log = []
        st.session_state.audit_log.append(record)

        try:
            from data_ingestion.mongo_loader import save_decision
            save_decision(record)
            st.caption("✅ Decision saved to MongoDB.")
        except Exception:
            st.caption("ℹ️ Decision saved to session log (MongoDB not configured).")


# ── Page: Dashboard ───────────────────────────────────────────────────────────

def page_dashboard():
    st.subheader("📊 Model Performance Overview")

    artifacts = _load_artifacts()
    if artifacts is None:
        st.info(
            "No trained model found. Open **Threshold Optimizer** in the sidebar "
            "to train the model first."
        )
        return

    _, _, y_test, y_prob, _ = artifacts
    auc = roc_auc_score(y_test, y_prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC",          f"{auc:.4f}")
    c2.metric("Test Samples",     f"{len(y_test):,}")
    c3.metric("Default Rate (test)", f"{y_test.mean():.1%}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_frame = pd.DataFrame(
        {
            "false_positive_rate": np.concatenate([fpr, np.array([0.0, 1.0])]),
            "true_positive_rate": np.concatenate([tpr, np.array([0.0, 1.0])]),
            "series": ["Model"] * len(fpr) + ["Baseline", "Baseline"],
        }
    )
    chart = (
        alt.Chart(roc_frame)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("false_positive_rate", title="False Positive Rate"),
            y=alt.Y("true_positive_rate", title="True Positive Rate"),
            color=alt.Color("series", title="Curve"),
            strokeDash=alt.condition(
                alt.datum.series == "Baseline",
                alt.value([6, 4]),
                alt.value([1, 0]),
            ),
        )
        .properties(title=f"ROC Curve (AUC = {auc:.4f})", height=320)
    )
    st.altair_chart(chart, use_container_width=True)


# ── Page: Threshold Optimizer ─────────────────────────────────────────────────

def page_threshold_optimizer():
    st.subheader("🎯 Threshold Optimizer — Lender Rules Engine")

    st.markdown("""
The model outputs a **probability of default** for every applicant (0 = safe payer → 1 = likely default).
A **decision threshold** converts that probability into a binary verdict — *Approve* or *Deny*.

| Threshold direction | Effect |
|---|---|
| **Lower** (e.g. 0.30) | Approve more applicants → higher revenue, more defaults slip through |
| **Higher** (e.g. 0.70) | Stricter approval → fewer defaults, but more good borrowers rejected |

Use this tool to explore the trade-off and lock in the threshold that matches your lending policy.
    """)

    # ── Step 1: model status ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 1 — Model Status")

    artifacts = _load_artifacts()

    if artifacts is None:
        data_ok = os.path.exists(DATA_PATH)
        if not data_ok:
            st.error(
                "❌ Dataset not found. Run `python scripts/generate_data.py` "
                "(or `python scripts/mock_data_generator.py`) from the project root first."
            )
            return

        col_msg, col_btn = st.columns([3, 1])
        col_msg.warning("⚠️ No trained model found. Click **Train Model** to begin.")
        if col_btn.button("🚀 Train Model"):
            try:
                with st.spinner("Training XGBoost on the synthetic dataset (this takes ~10 s)…"):
                    auc, threshold_info = _do_train()
                st.session_state["thresh_slider"] = float(threshold_info["threshold"])
                st.success(
                    f"✅ Training complete!  ROC-AUC = **{auc:.4f}** "
                    f"| Business threshold = **{threshold_info['threshold']:.2f}**"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        return

    _, _, y_test, y_prob, threshold_info = artifacts
    auc = roc_auc_score(y_test, y_prob)

    col_a, col_b, col_c, col_retrain = st.columns([2, 1, 1, 1])
    col_a.success(f"✅ Model loaded — ROC-AUC = **{auc:.4f}**")
    col_b.metric("Test samples", f"{len(y_test):,}")
    col_c.metric("Default rate", f"{y_test.mean():.1%}")
    if col_retrain.button("🔄 Retrain"):
        try:
            with st.spinner("Retraining…"):
                auc, threshold_info = _do_train()
            st.session_state["thresh_slider"] = float(threshold_info["threshold"])
            st.success(
                f"Retrained — AUC = {auc:.4f} | Business threshold = {threshold_info['threshold']:.2f}"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Retrain failed: {exc}")

    try:
        from evaluation.thresholds import (
            find_optimal_threshold,
            get_metrics_at_threshold,
            optimize_threshold_from_pr_curve,
        )
    except Exception as exc:
        st.error(f"Threshold optimizer import failed: {exc}")
        return

    try:
        sweep_df = _sweep_thresholds(tuple(y_test.tolist()), tuple(y_prob.tolist()))
    except Exception as exc:
        st.error(f"Threshold sweep failed: {exc}")
        return

    # ── Step 2: threshold slider ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 2 — Set Decision Threshold")

    st.info(
        "Recommended business threshold: "
        f"**{threshold_info['threshold']:.2f}** "
        f"({threshold_info.get('objective', 'business')})."
    )

    threshold = st.slider(
        "Decision Threshold  ←  lower = approve more  |  higher = approve fewer  →",
        min_value=0.01,
        max_value=0.99,
        value=float(
            st.session_state.get(
                "thresh_slider",
                float(threshold_info.get("threshold", 0.50)),
            )
        ),
        step=0.01,
        format="%.2f",
        key="thresh_slider",
    )

    # ── Step 3: live metrics ──────────────────────────────────────────────────
    try:
        m = get_metrics_at_threshold(y_test, y_prob, threshold)
    except Exception as exc:
        st.error(f"Metric calculation failed at threshold {threshold:.2f}: {exc}")
        return

    st.markdown("---")
    st.markdown(f"#### Step 3 — Live Metrics at Threshold = **{threshold:.2f}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision",     f"{m['precision']:.3f}",
              help="Of all *denied* applicants, what fraction were true defaulters?")
    c2.metric("Recall",        f"{m['recall']:.3f}",
              help="Of all true defaulters, what fraction did we correctly identify?")
    c3.metric("F1 Score",      f"{m['f1']:.3f}",
              help="Harmonic mean of Precision and Recall.")
    c4.metric("Accuracy",      f"{m['accuracy']:.3f}",
              help="Overall fraction of correctly classified applicants.")
    c5.metric("Approval Rate", f"{m['approval_rate']:.1%}",
              help="Fraction of applicants that would be *approved* at this threshold.")

    # ── Step 4: confusion matrix ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Confusion Matrix")

    cm_col, explain_col = st.columns([1, 1])

    with cm_col:
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        cm_frame = pd.DataFrame(
            [
                {"actual": "Paid", "predicted": "Approved", "count": int(cm[0, 0])},
                {"actual": "Paid", "predicted": "Denied", "count": int(cm[0, 1])},
                {"actual": "Default", "predicted": "Approved", "count": int(cm[1, 0])},
                {"actual": "Default", "predicted": "Denied", "count": int(cm[1, 1])},
            ]
        )
        heatmap = (
            alt.Chart(cm_frame)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="Predicted"),
                y=alt.Y("actual:N", title="Actual"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="greens")),
                tooltip=["actual", "predicted", "count"],
            )
            .properties(title=f"Confusion Matrix at Threshold {threshold:.2f}", height=220)
        )
        labels = (
            alt.Chart(cm_frame)
            .mark_text(fontSize=14, fontWeight="bold")
            .encode(
                x="predicted:N",
                y="actual:N",
                text="count:Q",
                color=alt.condition(alt.datum.count > cm.max() * 0.55, alt.value("white"), alt.value("black")),
            )
        )
        st.altair_chart(heatmap + labels, use_container_width=True)

    with explain_col:
        st.markdown(f"""
| Quadrant | Count | Meaning |
|---|---|---|
| **True Negative (TN)** | {m['tn']:,} | Correctly **approved** good borrowers ✅ |
| **False Positive (FP)** | {m['fp']:,} | Incorrectly **denied** good borrowers ❌ |
| **False Negative (FN)** | {m['fn']:,} | Defaulters **approved** (risk leaked) ⚠️ |
| **True Positive (TP)** | {m['tp']:,} | Correctly **denied** defaulters ✅ |

---
**Approval Rate:** {m['approval_rate']:.1%} of test applicants approved.
**Default leakage:** {m['fn']:,} defaulters would slip through as approved.
**Good borrowers blocked:** {m['fp']:,} creditworthy applicants incorrectly denied.
        """)

    # ── Step 5: metrics vs threshold chart ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### How Metrics Shift with Threshold")

    metrics_frame = sweep_df.melt(
        id_vars=["threshold"],
        value_vars=["precision", "recall", "f1", "approval_rate"],
        var_name="metric",
        value_name="value",
    )
    line_chart = (
        alt.Chart(metrics_frame)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("threshold:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["threshold", "metric", "value"],
        )
        .properties(title="Metric Trade-offs Across Thresholds", height=320)
    )
    threshold_rule = alt.Chart(
        pd.DataFrame({"threshold": [threshold], "label": [f"Current ({threshold:.2f})"]})
    ).mark_rule(color=ORANGE, strokeDash=[6, 4], strokeWidth=2).encode(x="threshold:Q")
    st.altair_chart(line_chart + threshold_rule, use_container_width=True)

    # ── Step 6: ROC and PR curves ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ROC Curve & Precision-Recall Curve")

    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_test, y_prob)
    pr_prec, pr_rec, pr_thresh   = precision_recall_curve(y_test, y_prob)

    roc_col, pr_col = st.columns(2)

    with roc_col:
        # Mark the point on the ROC curve closest to the chosen threshold
        if len(roc_thresh) > 1:
            roc_idx = int(np.argmin(np.abs(roc_thresh - threshold)))
            pt_fpr, pt_tpr = fpr_arr[roc_idx], tpr_arr[roc_idx]
        else:
            pt_fpr, pt_tpr = None, None

        roc_frame = pd.DataFrame({"fpr": fpr_arr, "tpr": tpr_arr})
        baseline_frame = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
        chart = (
            alt.Chart(roc_frame)
            .mark_line(color=PRIMARY, strokeWidth=3)
            .encode(x=alt.X("fpr:Q", title="False Positive Rate"), y=alt.Y("tpr:Q", title="True Positive Rate"))
            .properties(title="ROC Curve", height=280)
        )
        baseline = (
            alt.Chart(baseline_frame)
            .mark_line(color="gray", strokeDash=[6, 4])
            .encode(x="fpr:Q", y="tpr:Q")
        )
        layers = chart + baseline
        if pt_fpr is not None:
            point_frame = pd.DataFrame({"fpr": [pt_fpr], "tpr": [pt_tpr]})
            point = alt.Chart(point_frame).mark_point(color=ORANGE, size=100).encode(x="fpr:Q", y="tpr:Q")
            layers = layers + point
        st.altair_chart(layers, use_container_width=True)

    with pr_col:
        if len(pr_thresh) > 1:
            pr_idx = int(np.argmin(np.abs(pr_thresh - threshold)))
            pt_rec_v, pt_prec_v = pr_rec[pr_idx], pr_prec[pr_idx]
        else:
            pt_rec_v, pt_prec_v = None, None

        pr_frame = pd.DataFrame({"recall": pr_rec, "precision": pr_prec})
        baseline_frame = pd.DataFrame({"recall": [0.0, 1.0], "precision": [float(y_test.mean()), float(y_test.mean())]})
        chart = (
            alt.Chart(pr_frame)
            .mark_line(color=ACCENT, strokeWidth=3)
            .encode(x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision"))
            .properties(title="Precision-Recall Curve", height=280)
        )
        baseline = (
            alt.Chart(baseline_frame)
            .mark_line(color="gray", strokeDash=[6, 4])
            .encode(x="recall:Q", y="precision:Q")
        )
        layers = chart + baseline
        if pt_rec_v is not None:
            point_frame = pd.DataFrame({"recall": [pt_rec_v], "precision": [pt_prec_v]})
            point = alt.Chart(point_frame).mark_point(color=ORANGE, size=100).encode(x="recall:Q", y="precision:Q")
            layers = layers + point
        st.altair_chart(layers, use_container_width=True)

    # ── Step 7: optimal threshold recommendations ─────────────────────────────
    st.markdown("---")
    st.markdown("#### Optimal Threshold Recommendations")
    st.markdown(
        "The table below shows the best threshold for each business objective. "
        "Click a button to instantly apply that threshold."
    )

    st.markdown("#### Precision-Recall Business Optimizer")
    pr_c1, pr_c2, pr_c3 = st.columns(3)
    min_precision = pr_c1.slider("Minimum Precision", 0.0, 1.0, 0.60, 0.01)
    min_recall = pr_c2.slider("Minimum Recall", 0.0, 1.0, 0.20, 0.01)
    min_approval_rate = pr_c3.slider("Minimum Approval Rate", 0.0, 1.0, 0.20, 0.01)

    try:
        pr_best = optimize_threshold_from_pr_curve(
            y_test,
            y_prob,
            min_precision=min_precision,
            min_recall=min_recall,
            min_approval_rate=min_approval_rate,
        )
    except Exception as exc:
        st.error(f"Precision-Recall threshold optimization failed: {exc}")
        return

    pr_metric_cols = st.columns(5)
    pr_metric_cols[0].metric("PR Threshold", f"{pr_best['threshold']:.2f}")
    pr_metric_cols[1].metric("Precision", f"{pr_best['precision']:.3f}")
    pr_metric_cols[2].metric("Recall", f"{pr_best['recall']:.3f}")
    pr_metric_cols[3].metric("F1", f"{pr_best['f1']:.3f}")
    pr_metric_cols[4].metric("Approval Rate", f"{pr_best['approval_rate']:.1%}")
    st.caption(f"Objective used: {pr_best['objective']}")

    if st.button("Apply PR-Optimized Threshold", key="apply_pr_curve_threshold"):
        st.session_state["thresh_slider"] = float(pr_best["threshold"])
        st.rerun()

    OBJECTIVES = {
        "PR Curve Optimizer": "pr_curve",
        "Maximize F1":        "f1",
        "Maximize Precision": "precision",
        "Maximize Recall":    "recall",
        "Balanced (G-Mean)":  "balanced",
        "Maximize Profit":    "profit",
    }
    OBJECTIVE_TIPS = {
        "pr_curve":  "Uses the precision-recall curve plus minimum precision, recall, and approval-rate constraints.",
        "f1":        "Best balanced trade-off between catching defaulters and approving good borrowers.",
        "precision": "Risk-averse lender: minimise bad loans approved (may reject good borrowers).",
        "recall":    "Catch-all policy: identify every defaulter (accepts more false rejections).",
        "balanced":  "Equal weight to sensitivity and specificity — suitable for imbalanced datasets.",
        "profit":    "Weighted by business value: each approved good borrower = +1, bad loan = −5.",
    }

    rows = []
    opt_results = {}
    for label, obj in OBJECTIVES.items():
        if obj == "pr_curve":
            best = pr_best
        else:
            best = find_optimal_threshold(y_test, y_prob, objective=obj)
        opt_results[obj] = best
        rows.append({
            "Objective":          label,
            "Optimal Threshold":  f"{best['threshold']:.2f}",
            "Precision":          f"{best['precision']:.3f}",
            "Recall":             f"{best['recall']:.3f}",
            "F1":                 f"{best['f1']:.3f}",
            "Approval Rate":      f"{best['approval_rate']:.1%}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Quick-apply buttons
    st.markdown("**Quick-apply an objective:**")
    btn_cols = st.columns(len(OBJECTIVES))
    for idx, (label, obj) in enumerate(OBJECTIVES.items()):
        with btn_cols[idx]:
            short = (
                label.replace("Maximize ", "")
                .replace("Balanced (G-Mean)", "Balanced")
                .replace("PR Curve Optimizer", "PR Optimizer")
            )
            if st.button(short, key=f"apply_{obj}", help=OBJECTIVE_TIPS[obj]):
                st.session_state["thresh_slider"] = float(opt_results[obj]["threshold"])
                st.rerun()

    st.caption(
        "💡 Tip: start with **Maximize F1** for balanced performance, "
        "switch to **Maximize Recall** if reducing default leakage is the priority, "
        "or **Maximize Profit** for a dollar-weighted optimum."
    )

    if st.button("Apply Recommended Business Threshold", key="apply_business_threshold"):
        st.session_state["thresh_slider"] = float(threshold_info["threshold"])
        st.rerun()


# ── Page: Audit Logs ──────────────────────────────────────────────────────────

def page_audit_logs():
    st.subheader("📋 Audit Logs")

    logs = list(st.session_state.get("audit_log", []))

    # Also try to pull from MongoDB if available
    try:
        from data_ingestion.mongo_loader import load_decisions
        mongo_records = load_decisions()
        if mongo_records:
            st.info(f"Showing {len(mongo_records)} record(s) from MongoDB.")
            logs = mongo_records
    except Exception:
        pass

    if not logs:
        st.info(
            "No decisions recorded yet. "
            "Use **New Application** in the sidebar to score applicants."
        )
        return

    df = pd.DataFrame(logs)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="equilend_audit_log.csv",
        mime="text/csv",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="EquiLend AI — Credit Scoring",
        layout="wide",
        page_icon="⚖️",
    )

    st.title("⚖️ EquiLend AI: Transparent Credit Scoring")
    st.markdown("*Bridging the credit gap with fair, explainable, alternative-data ML.*")

    menu   = ["New Application", "Dashboard", "Threshold Optimizer", "Audit Logs"]
    choice = st.sidebar.selectbox("Navigation", menu)

    st.sidebar.markdown("---")
    artifacts = _load_artifacts()
    recommended_thresh = 0.50
    if artifacts is not None:
        recommended_thresh = float(artifacts[4].get("threshold", 0.50))
    active_thresh = st.session_state.get("thresh_slider", recommended_thresh)
    st.sidebar.metric("Active Threshold", f"{active_thresh:.2f}")
    st.sidebar.caption("Set in **Threshold Optimizer**")

    if choice == "New Application":
        page_new_application()
    elif choice == "Dashboard":
        page_dashboard()
    elif choice == "Threshold Optimizer":
        page_threshold_optimizer()
    elif choice == "Audit Logs":
        page_audit_logs()


if __name__ == "__main__":
    main()
