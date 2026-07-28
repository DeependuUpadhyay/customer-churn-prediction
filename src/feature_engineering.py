"""
feature_engineering.py
-----------------------
Cleans the raw Telco Customer Churn data and engineers new features that
give the model more predictive signal than the raw columns alone.

Design notes
------------
- All engineered features are derived ONLY from information that would be
  available at prediction time (no leakage from the target).
- Categorical encoding + scaling live in a scikit-learn ColumnTransformer
  (see train.py) so that the exact same transformation is applied at
  training and inference time -> no train/serve skew.
"""

import numpy as np
import pandas as pd

# Columns that describe an add-on service. Each can be "Yes" / "No" /
# "No internet service" / "No phone service".
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in the raw IBM Telco dataset."""
    df = df.copy()

    # TotalCharges is read as an object because 11 brand-new customers
    # (tenure == 0) have a blank string instead of a number.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Normalize the target to 0/1 early so every downstream step is numeric.
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of the cleaned raw columns."""
    df = df.copy()

    # 1. Tenure bucket - churn risk is highly non-linear in tenure
    #    (huge drop-off in the first year), so binning helps linear models
    #    and adds an easy-to-explain feature for tree models too.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 60, np.inf],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-5yr", "5yr+"],
    ).astype(str)

    # 2. Average monthly spend actually realized so far (differs from
    #    MonthlyCharges when a customer's plan changed over time).
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # 3. Number of add-on services subscribed to (0-6). Customers with more
    #    services embedded in their bill are stickier -> lower churn.
    df["num_add_on_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    # 4. Whether the customer has any internet service at all.
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # 5. Month-to-month contract flag - the single strongest churn driver
    #    in this dataset (no cancellation penalty).
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # 6. Paperless billing + electronic check is a well-known high-churn
    #    combination for this dataset (younger / more price-sensitive
    #    segment with lower payment friction to leave).
    df["risky_payment_profile"] = (
        (df["PaperlessBilling"] == "Yes")
        & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    # 7. Senior citizen living alone (no partner, no dependents) - a
    #    higher-risk demographic combination worth flagging explicitly.
    df["senior_living_alone"] = (
        (df["SeniorCitizen"] == 1)
        & (df["Partner"] == "No")
        & (df["Dependents"] == "No")
    ).astype(int)

    # 8. Simple customer lifetime value proxy: tenure x monthly charges.
    df["clv_proxy"] = df["tenure"] * df["MonthlyCharges"]

    # 9. Charges per service - is this customer paying a premium for what
    #    they get? High value here can signal price-driven churn risk.
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_add_on_services"] + 1)

    return df


def build_feature_frame(raw_csv_path: str) -> pd.DataFrame:
    """Full pipeline: load -> clean -> engineer. Returns a model-ready frame
    (still contains the target column and customerID; those are dropped in
    train.py right before fitting)."""
    df = load_raw_data(raw_csv_path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "num_add_on_services",
    "clv_proxy",
    "charge_per_service",
]

BINARY_FEATURES = [
    "SeniorCitizen",
    "has_internet",
    "is_month_to_month",
    "risky_payment_profile",
    "senior_living_alone",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
