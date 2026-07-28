"""
app.py
------
Streamlit product UI for the churn model. Two modes:

  1. Single Customer  - a sales/success rep fills in one customer's
     profile and gets an instant churn-risk score + top risk drivers.
  2. Batch Scoring    - upload a CSV of many customers and get a scored,
     downloadable CSV back, ranked by churn risk (this is what a
     retention team would actually run weekly).

Run locally:
    streamlit run app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from feature_engineering import (  # noqa: E402
    ALL_FEATURES,
    TARGET_COL,
    clean_data,
    engineer_features,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return joblib.load(MODELS_DIR / "churn_pipeline.pkl")


def risk_bucket(prob: float) -> tuple[str, str]:
    if prob >= 0.60:
        return "High risk", "#dc2626"
    elif prob >= 0.30:
        return "Medium risk", "#d97706"
    else:
        return "Low risk", "#16a34a"


def score_dataframe(pipeline, raw_df: pd.DataFrame) -> np.ndarray:
    """
    Take a raw dataframe shaped like the source Telco CSV (minus Churn),
    run it through the same cleaning + feature engineering used in
    training, and return churn probabilities.
    """
    df = raw_df.copy()

    if TARGET_COL not in df.columns:
        df[TARGET_COL] = "No"

    df = clean_data(df)
    df = engineer_features(df)

    X = df[ALL_FEATURES]

    return pipeline.predict_proba(X)[:, 1].astype(float)


def main():
    st.title("📉 Customer Churn Predictor")
    st.caption(
        "Predicts the probability that a telecom customer cancels their "
        "subscription, trained on the IBM Telco Customer Churn dataset "
        "(7,043 customers) with an XGBoost pipeline (ROC-AUC ≈ 0.85)."
    )

    try:
        pipeline = load_pipeline()
    except FileNotFoundError:
        st.error(
            "No trained model found at `models/churn_pipeline.pkl`. "
            "Run `python src/train.py` first."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🧍 Single Customer", "📁 Batch Scoring (CSV)", "📊 Model Report"])

    # ------------------------------------------------------------------
    # TAB 1 — single customer, manual entry form
    # ------------------------------------------------------------------
    with tab1:
        st.subheader("Score one customer")
        with st.form("single_customer_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Demographics**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", ["No", "Yes"])
                partner = st.selectbox("Has partner", ["No", "Yes"])
                dependents = st.selectbox("Has dependents", ["No", "Yes"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)

            with c2:
                st.markdown("**Account**")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
                payment = st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
                total_charges = st.number_input(
                    "Total charges to date ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)), step=10.0
                )

            with c3:
                st.markdown("**Services**")
                phone = st.selectbox("Phone service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

            submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

        if submitted:
            row = pd.DataFrame([{
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner, "Dependents": dependents, "tenure": tenure,
                "PhoneService": phone, "MultipleLines": multiple_lines,
                "InternetService": internet, "OnlineSecurity": online_security,
                "OnlineBackup": online_backup, "DeviceProtection": device_protection,
                "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            }])
            proba = float(score_dataframe(pipeline, row)[0])
            label, color = risk_bucket(proba)

            st.divider()
            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Churn probability", f"{proba:.1%}")
                st.markdown(
                    f"<span style='background-color:{color};color:white;"
                    f"padding:4px 12px;border-radius:12px;font-weight:600'>{label}</span>",
                    unsafe_allow_html=True,
                )
            with r2:
                st.progress(min(float(proba), 1.0))
                if contract == "Month-to-month":
                    st.write("⚠️ Month-to-month contract is the single biggest churn driver in this model.")
                if tenure < 6:
                    st.write("⚠️ New customer (under 6 months) — churn risk is highest in the first months.")
                num_services = sum(
                    1 for v in [online_security, online_backup, device_protection,
                                tech_support, streaming_tv, streaming_movies] if v == "Yes"
                )
                if num_services == 0 and internet != "No":
                    st.write("⚠️ No add-on services — customers with more bundled services tend to stay longer.")
                if not any([contract == "Month-to-month", tenure < 6, num_services == 0 and internet != "No"]):
                    st.write("✅ No major red flags detected for this customer.")

    # ------------------------------------------------------------------
    # TAB 2 — batch scoring
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Score a batch of customers")
        st.write(
            "Upload a CSV with the same columns as the Telco Customer Churn "
            "dataset (a `Churn` column is not required). Every row gets a "
            "churn probability and risk tier back."
        )
        sample_path = Path(__file__).resolve().parent / "data" / "telco_churn.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    "Download a sample CSV to try",
                    f,
                    file_name="sample_customers.csv",
                    mime="text/csv",
                )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            id_col = "customerID" if "customerID" in raw_df.columns else None
            try:
                proba = score_dataframe(pipeline, raw_df)
            except KeyError as e:
                st.error(f"Missing expected column in the uploaded file: {e}")
                st.stop()

            out = raw_df.copy()
            out["churn_probability"] = proba
            out["risk_tier"] = [risk_bucket(p)[0] for p in proba]
            out = out.sort_values("churn_probability", ascending=False)

            st.success(f"Scored {len(out)} customers.")
            c1, c2, c3 = st.columns(3)
            c1.metric("High risk", int((out["risk_tier"] == "High risk").sum()))
            c2.metric("Medium risk", int((out["risk_tier"] == "Medium risk").sum()))
            c3.metric("Low risk", int((out["risk_tier"] == "Low risk").sum()))

            show_cols = ([id_col] if id_col else []) + ["churn_probability", "risk_tier"]
            st.dataframe(out[show_cols + [c for c in out.columns if c not in show_cols]], use_container_width=True)

            st.download_button(
                "Download scored CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="scored_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # TAB 3 — static model report generated by src/evaluate.py
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Held-out test set performance")
        reports_dir = Path(__file__).resolve().parent / "reports"
        metrics_path = reports_dir / "metrics_summary.json"

        if metrics_path.exists():
            import json
            metrics = json.loads(metrics_path.read_text())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROC-AUC", metrics["roc_auc"])
            m2.metric("Avg. Precision", metrics["average_precision"])
            m3.metric("Top-decile lift", f"{metrics['top_decile_lift']}x")
            m4.metric("F1 (tuned threshold)", metrics["tuned_threshold"]["f1"])

            g1, g2 = st.columns(2)
            with g1:
                if (reports_dir / "roc_curve.png").exists():
                    st.image(str(reports_dir / "roc_curve.png"))
                if (reports_dir / "confusion_matrix.png").exists():
                    st.image(str(reports_dir / "confusion_matrix.png"))
            with g2:
                if (reports_dir / "precision_recall_curve.png").exists():
                    st.image(str(reports_dir / "precision_recall_curve.png"))
                if (reports_dir / "feature_importance.png").exists():
                    st.image(str(reports_dir / "feature_importance.png"))

            if (reports_dir / "lift_chart.png").exists():
                st.image(str(reports_dir / "lift_chart.png"))
                st.caption(
                    f"Targeting the top 2 risk deciles (20% of customers) catches "
                    f"~{metrics['top_2_deciles_capture_pct']}% of all customers who actually churn — "
                    f"a {metrics['top_decile_lift']}x lift over contacting customers at random."
                )
        else:
            st.warning("No report found. Run `python src/evaluate.py` after training to generate it.")


if __name__ == "__main__":
    main()
