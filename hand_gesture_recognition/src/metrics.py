"""utils/metrics.py — Evaluation utilities: confusion matrix, per-class metrics."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, top_k_accuracy_score
)
from typing import List, Tuple


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: str = "cpu",
) -> dict:
    """
    Runs full evaluation on a DataLoader.
    Returns accuracy, per-class metrics, and confusion matrix.
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)

    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1]
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()
    top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTop-1 Accuracy : {accuracy:.4f}")
    print(f"Top-3 Accuracy : {top3_acc:.4f}")
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_acc,
        "report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = True,
):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Gesture Recognition — Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(history: dict, save_path: str = "training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    return total, trainable
