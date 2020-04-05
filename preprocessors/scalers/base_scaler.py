import numpy as np


class BaseScaler:
    def __init__(self, train_size: int):
        """
        Base scaler class.

        :param train_size: int
            First train_size number of observations to fit on.
        """

        self.train_size = train_size
        self.params = {}

    def __call__(self, dataset: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def inverse_transform(self, dataset: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
