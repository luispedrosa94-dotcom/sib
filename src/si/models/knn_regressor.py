from typing import Callable
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN Regressor
    The k-Nearest Neighbors regressor is an instance-based learning algorithm
    that estimates target values based on the mean of its k nearest neighbors.

    Parameters
    ----------
    k: int, default=1
        The number of nearest neighbors to consider.
    distance: Callable, default=euclidean_distance
        The distance function to calculate distances between instances.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> float:
        distances = self.distance(sample, self.dataset.X)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_values = self.dataset.y[k_nearest_indices]
        return float(np.mean(k_nearest_values))

    def _predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)