import json
from flask import Blueprint, jsonify

from config import METRICS_PATH

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Run train_model.py and evaluate_model.py first"}), 404
    return jsonify(json.loads(METRICS_PATH.read_text()))
