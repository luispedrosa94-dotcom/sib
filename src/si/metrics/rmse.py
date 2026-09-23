import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between real and predicted target values.

    Parameters
    ----------
    y_true: np.ndarray
        Real target values.
    y_pred: np.ndarray
        Predicted target values.

    Returns
    -------
    float
        The RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))