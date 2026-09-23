import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance between a point x and a set of points y.

    Parameters
    ----------
    x: np.ndarray
        Point coordinates (1D array).
    y: np.ndarray
        Set of points coordinates (2D array).

    Returns
    -------
    np.ndarray
        Euclidean distances between x and each point in y.
    """
    return np.sqrt(np.sum((x - y) ** 2, axis=1))