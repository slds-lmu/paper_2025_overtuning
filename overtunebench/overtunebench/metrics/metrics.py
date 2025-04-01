from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)


def compute_accuracy(
    y_true: np.array,
    y_pred: np.array,
    y_pred_proba: np.array,
    labels: List[int],
    multiclass: bool,
) -> float:
    """
    Compute accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def compute_balanced_accuracy(
    y_true: np.array,
    y_pred: np.array,
    y_pred_proba: np.array,
    labels: List[int],
    multiclass: bool,
) -> float:
    """
    Compute balanced accuracy score.
    """
    return balanced_accuracy_score(y_true, y_pred)


def compute_logloss(
    y_true: np.array,
    y_pred: np.array,
    y_pred_proba: np.array,
    labels: List[int],
    multiclass: bool,
) -> float:
    """
    Compute logloss score.
    """
    return log_loss(y_true, y_pred_proba, labels=labels)


def compute_auc(
    y_true: np.array,
    y_pred: np.array,
    y_pred_proba: np.array,
    labels: List[int],
    multiclass: bool,
) -> float:
    """
    Compute AUC score.
    """
    if multiclass:
        return roc_auc_score(
            y_true, y_pred_proba, average="macro", multi_class="ovo", labels=labels
        )
    else:
        return roc_auc_score(y_true, y_pred_proba[:, 1])


def compute_metric(
    y_true: np.array,
    y_pred: np.array,
    y_pred_proba: np.array,
    metric: str,
    labels: List[int],
    multiclass: bool,
) -> float:
    """
    Compute a metric.
    """
    if metric == "accuracy":
        return compute_accuracy(
            y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            labels=labels,
            multiclass=multiclass,
        )
    elif metric == "balanced_accuracy":
        return compute_balanced_accuracy(
            y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            labels=labels,
            multiclass=multiclass,
        )
    elif metric == "logloss":
        return compute_logloss(
            y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            labels=labels,
            multiclass=multiclass,
        )
    elif metric == "auc":
        return compute_auc(
            y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            labels=labels,
            multiclass=multiclass,
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")
