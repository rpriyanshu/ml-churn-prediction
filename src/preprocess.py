import pandas as pd
import numpy as np
import os

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def load_data(path=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print("Downloading dataset from IBM GitHub...")
        df = pd.read_csv(DATA_URL)
        df.to_csv("data/raw_churn.csv", index=False)
        print(f"Saved to data/raw_churn.csv | Shape: {df.shape}")
    return df

def clean_data(df):
    df = df.copy()

    # Drop customerID - not a feature
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Fix TotalCharges - has spaces, convert to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute TotalCharges nulls with median (data cleaning step for the form)
    median_val = df["TotalCharges"].median()
    null_count = df["TotalCharges"].isnull().sum()
    df["TotalCharges"].fillna(median_val, inplace=True)
    print(f"Imputed {null_count} null values in TotalCharges with median={median_val:.2f}")

    # Encode binary target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode binary Yes/No columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # One-hot encode remaining categoricals
    cat_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

def split_data(df, seed=42):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=seed
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Churn rate - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def impute_age(df):
    """Used in unit tests - idempotent imputation."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_val = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_val, inplace=True)
    return df

if __name__ == "__main__":
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    print(df_clean.head())
    print(f"Final shape: {df_clean.shape}")