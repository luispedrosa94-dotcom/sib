import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy between real and predicted target values.

    Parameters
    ----------
    y_true: np.ndarray
        Real target values.
    y_pred: np.ndarray
        Predicted target values.

    Returns
    -------
    float
        The accuracy value.
    """
    return float(np.sum(y_true == y_pred) / len(y_true))