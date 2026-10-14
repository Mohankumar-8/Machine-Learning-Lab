"""Step 2 - Train the MLP and save it.

Trains epoch-by-epoch (partial_fit) so training / validation accuracy and loss
can be logged and plotted for every epoch.
"""
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from config import (TRAIN_CSV, MODEL_PATH, METRICS_PATH,
                    RANDOM_STATE)
from ml.preprocessing import load_dataset, clean_data
from ml.feature_engineering import build_pipeline, selected_feature_names

EPOCHS = 100


def main():
    X, y = load_dataset(TRAIN_CSV)
    X = clean_data(X)
    feature_names = list(X.columns)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipe = build_pipeline()
    scaler, select, mlp = (pipe.named_steps[k] for k in ("scaler", "select", "mlp"))
    Xt = select.fit_transform(scaler.fit_transform(X_tr), y_tr)
    Xv = select.transform(scaler.transform(X_val))

    classes = np.array([0, 1])
    y_tr = np.asarray(y_tr)
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    rng = np.random.RandomState(RANDOM_STATE)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    for epoch in range(1, EPOCHS + 1):
        idx = rng.permutation(len(Xt))
        mlp.partial_fit(Xt[idx], y_tr[idx], classes=classes)
        tl = log_loss(y_tr, mlp.predict_proba(Xt), labels=classes)
        vl = log_loss(y_val, mlp.predict_proba(Xv), labels=classes)
        ta = accuracy_score(y_tr, mlp.predict(Xt))
        va = accuracy_score(y_val, mlp.predict(Xv))
        for k, v in zip(history, (ta, va, tl, vl)):
            history[k].append(float(v))
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5} | {tl:>10.4f} | {ta:>9.4f} | {vl:>8.4f} | {va:>7.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": feature_names}, MODEL_PATH)

    sel = selected_feature_names(pipe, feature_names)
    METRICS_PATH.write_text(json.dumps({"history": history,
                                        "selected_features": sel,
                                        "epochs": EPOCHS}, indent=2))
    print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Selected features ({len(sel)}): {sel}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
