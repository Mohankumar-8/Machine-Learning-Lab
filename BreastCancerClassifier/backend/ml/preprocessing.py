"""Data loading and cleaning."""
import numpy as np
import pandas as pd


def load_dataset(csv_path):
    """Read a CSV and return (X dataframe, y array)."""
    df = pd.read_csv(csv_path)
    y = df["target"].values
    X = df.drop(columns=["target"])
    return X, y


def clean_data(X):
    """Replace inf with NaN, fill missing values with the column median."""
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True))
