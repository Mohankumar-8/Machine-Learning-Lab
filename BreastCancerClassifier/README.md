# BreastCancerClassifier

Web-based Breast Cancer (Benign / Malignant) prediction system using a
Multi-Layer Perceptron (MLP) neural network, Flask and SQLite.

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) - 569 samples, 30 numeric
features computed from digitized FNA images of breast masses (bundled with
scikit-learn, no download needed).

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## Run (from inside the backend folder)
```bash
cd backend
python prepare_dataset.py      # Step 1: create dataset/train.csv and dataset/test.csv
python train_model.py          # Step 2: preprocess + feature selection + train MLP -> models/
python evaluate_model.py       # Step 3: metrics + confusion matrix + ROC -> outputs/
python app.py                  # Step 4: start web app at http://127.0.0.1:5000
```

## Folders
| Path | Purpose |
|------|---------|
| backend/app.py | Flask entry point |
| backend/database.py | SQLite helper (prediction history) |
| backend/routes/ | Flask blueprints (/predict, /history, /metrics) |
| backend/ml/ | preprocessing, feature selection, model code |
| backend/dataset/ | train.csv / test.csv |
| backend/models/ | saved model (.pkl) + metrics.json |
| backend/outputs/ | generated plots |
| backend/templates, static | HTML / CSS / JS front-end |
| database/ | breast_cancer.db (created automatically) |

Educational project - not a medical diagnostic tool.
