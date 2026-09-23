from typing import Callable, Union
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):
    """
    KNN Classifier
    The k-Nearest Neighbors classifier is an instance-based learning algorithm
    that classifies instances based on the majority class among its k nearest neighbors.

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

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        distances = self.distance(sample, self.dataset.X)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.dataset.y[k_nearest_indices]
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)