import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def plot_training_curves(history, outdir):
    epochs = history["epoch"]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training/Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_accuracy.png"), dpi=150)
    plt.close()

    if "val_auc" in history.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history["val_auc"], label="val_auc")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("Validation AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "val_auc.png"), dpi=150)
        plt.close()


def plot_confusion_and_curves(pred_df, outdir):
    y_true = pred_df["label"].astype(int).values
    y_prob = pred_df["prob"].astype(float).values
    y_pred = pred_df["pred"].astype(int).values

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label="real")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label="fake")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "probability_distribution.png"), dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    history = pd.read_csv(args.history)
    pred_df = pd.read_csv(args.predictions)
    plot_training_curves(history, args.outdir)
    plot_confusion_and_curves(pred_df, args.outdir)


if __name__ == "__main__":
    main()
