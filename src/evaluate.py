import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, clean_data, split_data

def main():
    df_raw   = load_data("data/raw_churn.csv")
    df_clean = clean_data(df_raw)
    _, X_val, X_test, _, y_val, y_test = split_data(df_clean)

    # Load saved model
    model = XGBClassifier()
    model.load_model("checkpoints/xgb_best_v1.json")

    # Validation metrics
    val_probs  = model.predict_proba(X_val)[:, 1]
    val_preds  = (val_probs >= 0.4).astype(int)
    val_auc    = roc_auc_score(y_val, val_probs)

    print(f"Validation AUC: {val_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, val_preds)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Error analysis - find the failure mode
    import pandas as pd
    val_df = X_val.copy()
    val_df["true_label"]  = y_val.values
    val_df["pred_label"]  = val_preds
    val_df["pred_proba"]  = val_probs

    # False negatives - churners we missed
    false_negatives = val_df[(val_df["true_label"]==1) & (val_df["pred_label"]==0)]
    print(f"\nFalse Negatives (missed churners): {len(false_negatives)}")
    print("Mean feature values for missed churners:")
    print(false_negatives[["tenure", "MonthlyCharges", "TotalCharges"]].mean())

    # Save confusion matrix plot
    os.makedirs("logs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retain","Churn"])
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix (threshold=0.4)")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=150)
    print("Saved: logs/confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, val_probs)
    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.plot(fpr, tpr, label=f"AUC = {val_auc:.4f}")
    ax2.plot([0,1],[0,1],"--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve - Validation Set")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("logs/roc_curve.png", dpi=150)
    print("Saved: logs/roc_curve.png")

if __name__ == "__main__":
    main()