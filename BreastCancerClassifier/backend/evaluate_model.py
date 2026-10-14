"""Step 3 - Evaluate on the held-out test set and save plots + metrics."""
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

from config import TEST_CSV, MODEL_PATH, METRICS_PATH, OUTPUT_DIR, CLASSES
from ml.preprocessing import load_dataset, clean_data

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_history(h):
    ep = range(1, len(h["train_acc"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, h["train_acc"], label="Train"); ax[0].plot(ep, h["val_acc"], label="Validation")
    ax[0].set_title("Accuracy per epoch"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, h["train_loss"], label="Train"); ax[1].plot(ep, h["val_loss"], label="Validation")
    ax[1].set_title("Loss per epoch"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150); plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)


def plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}"); ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150); plt.close(fig)


def main():
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    X, y = load_dataset(TEST_CSV)
    X = clean_data(X)

    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 0]          # probability of Malignant (class 0)

    # "Positive" class = Malignant (the clinically important one): flip labels
    y_true_m = (y == 0).astype(int); y_pred_m = (y_pred == 0).astype(int)
    acc = accuracy_score(y_true_m, y_pred_m)
    prec = precision_score(y_true_m, y_pred_m)
    rec = recall_score(y_true_m, y_pred_m)
    f1 = f1_score(y_true_m, y_pred_m)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true_m, y_prob); roc_auc = auc(fpr, tpr)

    print(classification_report(y, y_pred, target_names=CLASSES))
    print(f"Accuracy  : {acc*100:.2f}%\nPrecision : {prec*100:.2f}%\n"
          f"Recall    : {rec*100:.2f}%\nF1-Score  : {f1*100:.2f}%\nAUC       : {roc_auc:.4f}")
    print("Confusion matrix [rows=actual, cols=pred] (Malignant, Benign):\n", cm)

    m = json.loads(METRICS_PATH.read_text())
    m["test"] = {"accuracy": round(acc*100, 2), "precision": round(prec*100, 2),
                 "recall": round(rec*100, 2), "f1": round(f1*100, 2),
                 "auc": round(roc_auc, 4), "confusion_matrix": cm.tolist(),
                 "n_test": int(len(y))}
    METRICS_PATH.write_text(json.dumps(m, indent=2))

    plot_history(m["history"]); plot_confusion(cm); plot_roc(fpr, tpr, roc_auc)
    print(f"Plots saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
