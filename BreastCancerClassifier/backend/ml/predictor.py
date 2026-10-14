"""Load the saved model once and make predictions from a feature dict."""
import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, CLASSES

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def feature_names():
    return _load()["feature_names"]


def predict_from_dict(values):
    """values: {feature_name: number}. Returns a result dict."""
    bundle = _load()
    names = bundle["feature_names"]
    missing = [n for n in names if n not in values]
    if missing:
        raise ValueError(f"Missing features: {missing[:3]}...")
    row = pd.DataFrame([[float(values[n]) for n in names]], columns=names)
    if not np.isfinite(row.values).all() or (row.values < 0).any():
        raise ValueError("All feature values must be non-negative numbers.")
    proba = bundle["pipeline"].predict_proba(row)[0]
    idx = int(proba.argmax())
    return {
        "prediction": CLASSES[idx],
        "confidence": round(float(proba[idx]) * 100, 1),
        "malignant_prob": round(float(proba[0]) * 100, 1),
        "benign_prob": round(float(proba[1]) * 100, 1),
    }
