from abc import ABC, abstractmethod
from si.base.estimator import Estimator
from si.data.dataset import Dataset


class Model(Estimator, ABC):
    """
    Abstract base class for models.
    A model is an object that can predict the target values of a Dataset object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model.
        """
        super().__init__(**kwargs)

    def predict(self, dataset: Dataset):
        """
        Predict the target values of the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        return self._predict(dataset)

    @abstractmethod
    def _predict(self, dataset: Dataset):
        """
        Abstract method to predict the target values of the dataset.
        """
        raise NotImplementedError

    def score(self, dataset: Dataset) -> float:
        """
        Compute the score of the model on the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on.

        Returns
        -------
        score: float
            The score of the model.
        """
        return self._score(dataset)

    @abstractmethod
    def _score(self, dataset: Dataset) -> float:
        """
        Abstract method to compute the score of the model on the dataset.
        """
        raise NotImplementedError