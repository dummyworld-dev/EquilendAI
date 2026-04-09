from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


def evaluate_classification_model(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Compute core classification metrics for a trained model.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        y_proba (Optional[np.ndarray]): Predicted probabilities for the positive
            class (shape: [n_samples]). If provided, AUC is computed.
        model_name (str): Name used in printed output.

    Returns:
        Dict[str, Any]: Dictionary with accuracy, F1, and (optionally) AUC.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")

    # AUC requires probabilities or decision scores.
    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            # In cases where only one class is present in y_true.
            auc = None

    print(f"\nEvaluation results for {model_name}:")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: N/A (probabilities not provided or invalid).")

    return {"accuracy": accuracy, "f1": f1, "auc": auc}


def plot_roc_curve(
    y_true,
    y_proba: np.ndarray,
    model_name: str = "model",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the ROC curve for a binary classifier.

    Args:
        y_true: Ground-truth labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        model_name (str): Used in the plot title.
        show (bool): Whether to display the plot immediately.
        save_path (Optional[str]): If provided, saves the figure to this path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "model",
    normalize: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a confusion matrix for a classifier.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        model_name (str): Used in the plot title.
        normalize (Optional[str]): Normalization mode ("true", "pred", "all"
            or None), passed directly to ConfusionMatrixDisplay.
        show (bool): Whether to display the plot immediately.
        save_path (Optional[str]): If provided, saves the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
    plt.title(f"Confusion Matrix - {model_name}")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

