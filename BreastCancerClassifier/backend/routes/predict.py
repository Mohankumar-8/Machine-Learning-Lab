from flask import Blueprint, request, jsonify
import pandas as pd

from config import TEST_CSV
from database import log_prediction, get_history, clear_history
from ml.predictor import predict_from_dict, feature_names

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/features")
def features():
    return jsonify(feature_names())


@predict_bp.route("/sample/<kind>")
def sample(kind):
    """Return a real test-set record (benign / malignant) to try the system."""
    df = pd.read_csv(TEST_CSV)
    target = 1 if kind == "benign" else 0
    row = df[df["target"] == target].sample(1).iloc[0].drop("target")
    return jsonify({k: round(float(v), 5) for k, v in row.items()})


@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    label = str(data.get("patient_label") or "Anonymous")[:60]
    try:
        result = predict_from_dict(data.get("features", {}))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    log_prediction(label, result["prediction"], result["confidence"])
    return jsonify(result)


@predict_bp.route("/history", methods=["GET"])
def history():
    return jsonify(get_history())


@predict_bp.route("/history", methods=["DELETE"])
def history_clear():
    clear_history()
    return jsonify({"status": "cleared"})
