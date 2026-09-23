from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split.
    test_size: float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state: int, default=None
        Controls the shuffling applied to the data before applying the split.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training and testing datasets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)

    permutations = np.random.permutation(n_samples)
    test_indices = permutations[:n_test]
    train_indices = permutations[n_test:]

    train_X = dataset.X[train_indices]
    train_y = dataset.y[train_indices] if dataset.has_label() else None

    test_X = dataset.X[test_indices]
    test_y = dataset.y[test_indices] if dataset.has_label() else None

    train_dataset = Dataset(X=train_X, y=train_y, features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=test_X, y=test_y, features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset