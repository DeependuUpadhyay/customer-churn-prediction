"""
train.py
--------
End-to-end training script:
  1. Load + engineer features (feature_engineering.py)
  2. Train/test split (stratified, so churn rate is preserved in both)
  3. Build a preprocessing pipeline (impute/scale/one-hot) inside a
     sklearn Pipeline so preprocessing is learned ONLY on the training
     fold at every cross-validation split (no leakage).
  4. Train three candidate models with 5-fold stratified CV +
     hyperparameter search: Logistic Regression, Random Forest, XGBoost.
  5. Select the best model by cross-validated ROC-AUC.
  6. Refit the winner on the full training set and save it (pipeline +
     model in a single artifact) to models/churn_pipeline.pkl.
  7. Hand off to evaluate.py to score the held-out test set and produce
     the evaluation report / charts.

Run:
    python src/train.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from feature_engineering import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_feature_frame,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def build_preprocessor() -> ColumnTransformer:
    """Numeric features -> median impute + scale.
    Categorical/binary features -> most-frequent impute + one-hot encode."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES + BINARY_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_model_search_space():
    """Return {name: (estimator, param_distributions)} for each candidate."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.05, 0.1, 0.3, 1, 3],
                "model__penalty": ["l2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [1, 4, 8],
                "model__max_features": ["sqrt"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=2,
                tree_method="hist",
            ),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__scale_pos_weight": [1, 2.77],  # ~ (1 - churn_rate)/churn_rate
            },
        ),
    }


def main():
    t0 = time.time()
    print("1/5  Loading data and engineering features ...")
    df = build_feature_frame(str(DATA_PATH))
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]
    print(f"     rows={len(df)}  churn_rate={y.mean():.3f}")

    print("2/5  Train / test split (stratified, 80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"     train={len(X_train)}  test={len(X_test)}")

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_searches = {}

    print("3/5  Randomized hyperparameter search (3-fold CV, ROC-AUC) per model ...")
    for name, (estimator, param_dist) in get_model_search_space().items():
        t_model = time.time()
        # SMOTE lives inside the CV pipeline so oversampling is fit
        # only on each training fold, never on the validation fold.
        pipe = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ])

        n_jobs = 2 if name == "xgboost" else -1
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=8,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"     ({name} took {time.time() - t_model:.1f}s)")
        fitted_searches[name] = search
        results[name] = {
            "best_cv_roc_auc": search.best_score_,
            "best_params": search.best_params_,
        }
        print(f"     {name:<20s} best CV ROC-AUC = {search.best_score_:.4f}")

    print("4/5  Selecting best model ...")
    best_name = max(results, key=lambda k: results[k]["best_cv_roc_auc"])
    best_search = fitted_searches[best_name]
    best_pipeline = best_search.best_estimator_
    print(f"     winner: {best_name}  (CV ROC-AUC={results[best_name]['best_cv_roc_auc']:.4f})")

    print("5/5  Saving artifacts ...")
    joblib.dump(best_pipeline, MODELS_DIR / "churn_pipeline.pkl")
    joblib.dump({"features": ALL_FEATURES}, MODELS_DIR / "feature_manifest.pkl")

    # Persist the held-out split so evaluate.py scores on data the model
    # never saw during training or hyperparameter selection.
    X_test.assign(**{TARGET_COL: y_test}).to_csv(
        MODELS_DIR / "holdout_test_set.csv", index=False
    )

    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "results": {
                    k: {
                        "best_cv_roc_auc": v["best_cv_roc_auc"],
                        "best_params": v["best_params"],
                    }
                    for k, v in results.items()
                },
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "churn_rate": float(y.mean()),
                "elapsed_seconds": round(time.time() - t0, 1),
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Done in {time.time() - t0:.1f}s. Model saved to {MODELS_DIR / 'churn_pipeline.pkl'}")
    print("Now run: python src/evaluate.py")


if __name__ == "__main__":
    main()
