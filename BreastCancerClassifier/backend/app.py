"""Step 4 - Flask application entry point.   Run:  python app.py"""
from flask import Flask, render_template, send_from_directory

from config import OUTPUT_DIR
from routes.predict import predict_bp
from routes.metrics import metrics_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(predict_bp)
    app.register_blueprint(metrics_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/outputs/<path:name>")
    def outputs(name):
        return send_from_directory(OUTPUT_DIR, name)

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
