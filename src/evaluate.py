"""
evaluate.py
-----------
Scores the saved pipeline on the held-out test set and produces the
evaluation artifacts a stakeholder (or a resume reviewer) would actually
want to see:

  - classification_report.txt   precision / recall / F1 / support
  - confusion_matrix.png
  - roc_curve.png                with AUC
  - precision_recall_curve.png   with average precision (better than ROC
                                  for an imbalanced target like churn)
  - lift_chart.png               decile lift + cumulative gains, i.e.
                                  "if we contact the top 20% highest-risk
                                  customers, how many actual churners do
                                  we catch, and how much better is that
                                  than contacting people at random?"
  - feature_importance.png       top drivers of churn from the model
  - metrics_summary.json         all headline numbers in one file

Run (after train.py):
    python src/evaluate.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from feature_engineering import ALL_FEATURES, TARGET_COL

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_artifacts():
    pipeline = joblib.load(MODELS_DIR / "churn_pipeline.pkl")
    test_df = pd.read_csv(MODELS_DIR / "holdout_test_set.csv")
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COL]
    return pipeline, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix (threshold = 0.5)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return cm


def plot_roc_curve(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return auc


def plot_precision_recall_curve(y_true, y_proba, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color="#dc2626", lw=2, label=f"Model (AP = {ap:.3f})")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline (churn rate = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return ap


def plot_lift_and_gains(y_true, y_proba, out_path, n_bins=10):
    """Business-facing chart: sort customers by predicted churn risk,
    split into decile buckets, and show (a) lift over random targeting
    and (b) cumulative % of actual churners captured."""
    df = pd.DataFrame({"y": y_true.values, "proba": y_proba})
    df = df.sort_values("proba", ascending=False).reset_index(drop=True)
    df["bucket"] = pd.qcut(df.index, n_bins, labels=False) + 1  # 1 = highest risk

    overall_rate = df["y"].mean()
    bucket_stats = df.groupby("bucket").agg(
        n=("y", "size"), churners=("y", "sum")
    )
    bucket_stats["churn_rate"] = bucket_stats["churners"] / bucket_stats["n"]
    bucket_stats["lift"] = bucket_stats["churn_rate"] / overall_rate
    bucket_stats["cum_churners"] = bucket_stats["churners"].cumsum()
    bucket_stats["cum_pct_churners_captured"] = (
        bucket_stats["cum_churners"] / bucket_stats["churners"].sum() * 100
    )
    bucket_stats["cum_pct_customers_contacted"] = (
        bucket_stats["n"].cumsum() / bucket_stats["n"].sum() * 100
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Lift-by-decile bar chart
    ax = axes[0]
    bars = ax.bar(bucket_stats.index, bucket_stats["lift"], color="#2563eb")
    ax.axhline(1.0, linestyle="--", color="gray", label="Random targeting")
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Lift over random")
    ax.set_title("Lift Chart by Decile")
    ax.set_xticks(range(1, n_bins + 1))
    ax.legend()
    for b, v in zip(bars, bucket_stats["lift"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}x",
                 ha="center", fontsize=8)

    # Cumulative gains chart
    ax2 = axes[1]
    x = [0] + bucket_stats["cum_pct_customers_contacted"].tolist()
    y = [0] + bucket_stats["cum_pct_churners_captured"].tolist()
    ax2.plot(x, y, color="#dc2626", lw=2, marker="o", markersize=3, label="Model")
    ax2.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random")
    ax2.set_xlabel("% of customers contacted (ranked by risk)")
    ax2.set_ylabel("% of actual churners captured")
    ax2.set_title("Cumulative Gains Chart")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return bucket_stats


def plot_feature_importance(pipeline, out_path, top_n=15):
    """Works for tree models (feature_importances_) and linear models
    (coef_) alike; whichever the winning pipeline uses."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        label = "|Coefficient|"
    else:
        return None

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df["feature"] = imp_df["feature"].str.replace("num__", "").str.replace("cat__", "")
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2563eb")
    ax.set_xlabel(label)
    ax.set_title(f"Top {top_n} Features Driving Churn Predictions")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return imp_df


def find_best_threshold(y_true, y_proba):
    """Pick the probability threshold that maximizes F1, as an example
    of tuning the decision threshold instead of blindly using 0.5."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    return best_t, max(f1s)


def main():
    print("Loading model + held-out test set ...")
    pipeline, X_test, y_test = load_artifacts()

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    print("Computing metrics at default threshold (0.5) ...")
    report_txt = classification_report(y_test, y_pred_default, target_names=["Stayed", "Churned"])
    (REPORTS_DIR / "classification_report.txt").write_text(report_txt)
    print(report_txt)

    print("Plotting confusion matrix ...")
    cm = plot_confusion_matrix(y_test, y_pred_default, REPORTS_DIR / "confusion_matrix.png")

    print("Plotting ROC curve ...")
    auc = plot_roc_curve(y_test, y_proba, REPORTS_DIR / "roc_curve.png")

    print("Plotting precision-recall curve ...")
    ap = plot_precision_recall_curve(y_test, y_proba, REPORTS_DIR / "precision_recall_curve.png")

    print("Plotting lift chart + cumulative gains ...")
    bucket_stats = plot_lift_and_gains(y_test, y_proba, REPORTS_DIR / "lift_chart.png")
    bucket_stats.to_csv(REPORTS_DIR / "lift_table.csv")

    print("Plotting feature importance ...")
    plot_feature_importance(pipeline, REPORTS_DIR / "feature_importance.png")

    best_threshold, best_f1 = find_best_threshold(y_test, y_proba)
    y_pred_tuned = (y_proba >= best_threshold).astype(int)

    summary = {
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "default_threshold_0.5": {
            "precision": round(float(precision_score(y_test, y_pred_default)), 4),
            "recall": round(float(recall_score(y_test, y_pred_default)), 4),
            "f1": round(float(f1_score(y_test, y_pred_default)), 4),
        },
        "tuned_threshold": {
            "threshold": round(float(best_threshold), 3),
            "precision": round(float(precision_score(y_test, y_pred_tuned)), 4),
            "recall": round(float(recall_score(y_test, y_pred_tuned)), 4),
            "f1": round(float(best_f1), 4),
        },
        "top_decile_lift": round(float(bucket_stats.loc[1, "lift"]), 2),
        "top_2_deciles_capture_pct": round(
            float(bucket_stats.loc[2, "cum_pct_churners_captured"]), 1
        ),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    }

    with open(REPORTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nAll charts + reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
