from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    average: str = "macro",
    label_names: list[str] | None = None,
) -> dict[str, float | list[list[int]] | list[dict[str, float | int | str]]]:
    labels = list(range(len(label_names))) if label_names else sorted(set(y_true) | set(y_pred))
    per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class_metrics: list[dict[str, float | int | str]] = []
    for index, label in enumerate(labels):
        class_name = label_names[label] if label_names else str(label)
        per_class_metrics.append(
            {
                "class_index": label,
                "class_name": class_name,
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
        )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": per_class_metrics,
    }
