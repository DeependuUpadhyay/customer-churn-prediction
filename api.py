"""
api.py
------
Optional lightweight REST API (Flask) for the same model, for when you
want to call churn scoring from another service instead of a human using
the Streamlit UI (e.g. a nightly batch job, or a CRM webhook).

Run:
    python api.py
Then:
    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
           "tenure":1,"PhoneService":"No","MultipleLines":"No phone service",
           "InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes",
           "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
           "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
           "PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from feature_engineering import ALL_FEATURES, TARGET_COL, clean_data, engineer_features  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models"

app = Flask(__name__)
pipeline = joblib.load(MODELS_DIR / "churn_pipeline.pkl")


def risk_tier(prob: float) -> str:
    if prob >= 0.60:
        return "High risk"
    elif prob >= 0.30:
        return "Medium risk"
    return "Low risk"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts a single customer JSON object OR a list of customer objects."""
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]

    df = pd.DataFrame(records)
    df[TARGET_COL] = "No"  # placeholder, dropped by feature selection below
    try:
        df = clean_data(df)
        df = engineer_features(df)
        X = df[ALL_FEATURES]
    except KeyError as e:
        return jsonify({"error": f"missing field: {e}"}), 400

    probs = pipeline.predict_proba(X)[:, 1]
    results = [
        {"churn_probability": round(float(p), 4), "risk_tier": risk_tier(p)}
        for p in probs
    ]
    return jsonify(results if isinstance(payload, list) else results[0])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
