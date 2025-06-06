import numpy as np
from common.utils import setup_logging
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classification(y_prob, y_true):
    """
    Evaluate classification performance using balanced accuracy, F1 score, and ROC AUC.

    Parameters:
        y_prob (np.array): Predicted probabilities for the positive class.
        y_true (np.array): True labels.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    logger = setup_logging()
    y_pred = (y_prob[:, 1] > 0.5).astype(int)
    acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_prob[:, 1])

    logger.info(
        f"\t Balanced Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}"
    )

    return {
        "acc": acc,
        "f1": f1,
        "roc": roc_auc,
    }


def evaluate_regression(y_pred, y_true):
    """
    Evaluate regression performance using R^2 score.

    Parameters:
        y_pred (np.array): Predicted values.
        y_true (np.array): True values.

    Returns:
        float: R^2 score.
    """
    logger = setup_logging()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    logger.info(f"\t MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def evaluate_fidelity_classification(y_prob_student, y_prob_teacher):
    """
    Evaluate the fidelity of a student model compared to a teacher model in classification tasks.

    Parameters:
        y_prob_teacher (np.array): Predicted probabilities from the teacher model.
        y_prob_student (np.array): Predicted probabilities from the student model.

    Returns:
        dict: Dictionary containing evaluation metrics for fidelity.
    """
    logger = setup_logging()

    y_pred_student = (y_prob_student[:, 1] > 0.5).astype(int)

    if y_prob_teacher.ndim == 2:
        y_pred_teacher = (y_prob_teacher[:, 1] > 0.5).astype(int)
        kl_div = kl_divergence_soft(y_prob_student[:, 1], y_prob_teacher[:, 1])

    elif y_prob_teacher.ndim == 1:
        y_pred_teacher = y_prob_teacher.astype(int)
        kl_div = kl_divergence_hard(y_prob_student[:, 1], y_pred_teacher)

    else:
        raise ValueError(f"y_prob_teacher has unexpected shape: {y_prob_teacher.shape}")

    acc = balanced_accuracy_score(y_pred_teacher, y_pred_student)
    f1 = f1_score(y_pred_teacher, y_pred_student, average="macro")
    roc_auc = roc_auc_score(y_pred_teacher, y_prob_student[:, 1])

    logger.info(
        f"\t Fidelity - Balanced Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, KL Divergence: {kl_div:.4f}"
    )

    return {
        "fidelity_acc": acc,
        "fidelity_f1": f1,
        "fidelity_roc": roc_auc,
        "fidelity_kl_div": kl_div,
    }


def evaluate_fidelity_regression(y_pred_student, y_pred_teacher):
    """
    Evaluate the fidelity of a student model compared to a teacher model in regression tasks.

    Parameters:
        y_pred_teacher (np.array): Predicted values from the teacher model.
        y_pred_student (np.array): Predicted values from the student model.

    Returns:
        dict: Dictionary containing evaluation metrics for fidelity.
    """
    logger = setup_logging()

    mae = mean_absolute_error(y_pred_teacher, y_pred_student)
    mse = mean_squared_error(y_pred_teacher, y_pred_student)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_pred_teacher, y_pred_student)

    logger.info(
        f"\t Fidelity - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}"
    )

    return {
        "fidelity_mae": mae,
        "fidelity_mse": mse,
        "fidelity_rmse": rmse,
        "fidelity_r2": r2,
    }


def kl_divergence_hard(probs_pos, y_true):
    eps = 1e-7
    probs_pos = np.clip(probs_pos, eps, 1 - eps)

    log_likelihood = y_true * np.log(probs_pos) + (1 - y_true) * np.log(1 - probs_pos)
    result = -np.mean(log_likelihood)

    return result if not np.isnan(result) else 0.0


def kl_divergence_soft(p, q):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))
