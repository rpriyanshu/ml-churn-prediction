import os
import sys
import argparse
import numpy as np
import mlflow
import mlflow.xgboost

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, clean_data, split_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",           type=float, default=0.05)
    parser.add_argument("--n_estimators", type=int,   default=300)
    parser.add_argument("--max_depth",    type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and prepare data
    df_raw   = load_data("data/raw_churn.csv")
    df_clean = clean_data(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean, seed=args.seed)

    # Hyperparameter grid
    param_grid = {
        "n_estimators":  [100, 300, 500],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample":     [0.7, 0.9],
    }

    # Stratified CV - critical because churn is imbalanced (~26%)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=args.seed
    )

    print("Starting GridSearchCV (this takes ~5-10 minutes)...")
    grid_search = GridSearchCV(
        clf, param_grid, cv=cv,
        scoring="roc_auc", n_jobs=4, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_auc      = grid_search.best_score_

    # Evaluate on validation set
    val_preds = best_model.predict_proba(X_val)[:, 1]
    val_auc   = roc_auc_score(y_val, val_preds)
    val_preds_binary = (val_preds >= 0.4).astype(int)

    # Train AUC (to check overfitting)
    train_preds = best_model.predict_proba(X_train)[:, 1]
    train_auc   = roc_auc_score(y_train, train_preds)

    print(f"\n[RESULTS] CV AUC={cv_auc:.4f} | Train AUC={train_auc:.4f} | Val AUC={val_auc:.4f}")
    print(f"Best params: {best_params}")
    print(f"\nClassification Report (threshold=0.4):")
    print(classification_report(y_val, val_preds_binary, target_names=["Retain", "Churn"]))

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/xgb_best_v1.json"
    best_model.save_model(checkpoint_path)
    print(f"\nModel saved: {checkpoint_path}")

    # Log to MLflow
    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_auc",    cv_auc)
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("val_auc",   val_auc)
        mlflow.xgboost.log_model(best_model, "model")
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")

    # Save log line for the form
    os.makedirs("logs", exist_ok=True)
    with open("logs/train_results.log", "w") as f:
        f.write(f"[FINAL] val_auc={val_auc:.4f} | train_auc={train_auc:.4f} | "
                f"best_params={best_params} | checkpoint=xgb_best_v1.json | "
                f"mlflow_run_id={run_id}\n")

    print("\nDone. Check logs/train_results.log for final validation line.")

if __name__ == "__main__":
    main()