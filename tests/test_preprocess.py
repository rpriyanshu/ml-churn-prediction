import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import impute_age, clean_data

@pytest.fixture
def sample_df():
    """50-row fixture with known nulls in TotalCharges."""
    np.random.seed(42)
    data = {
        "customerID":      [f"ID{i}" for i in range(50)],
        "gender":          np.random.choice(["Male","Female"], 50),
        "SeniorCitizen":   np.random.randint(0,2,50),
        "Partner":         np.random.choice(["Yes","No"], 50),
        "Dependents":      np.random.choice(["Yes","No"], 50),
        "tenure":          np.random.randint(1,72,50),
        "PhoneService":    np.random.choice(["Yes","No"], 50),
        "MultipleLines":   np.random.choice(["Yes","No","No phone service"], 50),
        "InternetService": np.random.choice(["DSL","Fiber optic","No"], 50),
        "OnlineSecurity":  np.random.choice(["Yes","No","No internet service"], 50),
        "OnlineBackup":    np.random.choice(["Yes","No","No internet service"], 50),
        "DeviceProtection":np.random.choice(["Yes","No","No internet service"], 50),
        "TechSupport":     np.random.choice(["Yes","No","No internet service"], 50),
        "StreamingTV":     np.random.choice(["Yes","No","No internet service"], 50),
        "StreamingMovies": np.random.choice(["Yes","No","No internet service"], 50),
        "Contract":        np.random.choice(["Month-to-month","One year","Two year"], 50),
        "PaperlessBilling":np.random.choice(["Yes","No"], 50),
        "PaymentMethod":   np.random.choice(["Electronic check","Mailed check"], 50),
        "MonthlyCharges":  np.random.uniform(20, 100, 50),
        "TotalCharges":    [str(np.random.uniform(100,5000)) if i % 5 != 0 else " " for i in range(50)],
        "Churn":           np.random.choice(["Yes","No"], 50),
    }
    return pd.DataFrame(data)

def test_no_nulls_after_impute(sample_df):
    """After impute_age, TotalCharges must have zero nulls."""
    result = impute_age(sample_df)
    assert result["TotalCharges"].isnull().sum() == 0, "Nulls still present after imputation"

def test_imputed_values_in_range(sample_df):
    """All TotalCharges values must be >= 0 after imputation."""
    result = impute_age(sample_df)
    assert (result["TotalCharges"] >= 0).all(), "Negative values found after imputation"

def test_imputation_is_idempotent(sample_df):
    """Calling impute_age twice gives identical results."""
    result_once  = impute_age(sample_df)
    result_twice = impute_age(result_once)
    pd.testing.assert_frame_equal(result_once, result_twice)

def test_clean_data_no_nulls(sample_df):
    """clean_data should return a DataFrame with no null values."""
    result = clean_data(sample_df)
    assert result.isnull().sum().sum() == 0, "Nulls remain after clean_data()"

def test_churn_column_binary(sample_df):
    """After cleaning, Churn column must contain only 0 and 1."""
    result = clean_data(sample_df)
    assert set(result["Churn"].unique()).issubset({0, 1}), "Churn column has values other than 0/1"