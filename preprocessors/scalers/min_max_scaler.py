import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from preprocessors.scalers.base_scaler import BaseScaler


class MinMaxScalerData(BaseScaler):
    def __init__(self, train_size: int, feature_range: Tuple[int] = (0, 1)):
        """

        Performs MinMaxScale on the dataset given.
        Fit() is performed on first train_size number of observations.

        :param train_size: int
            First train_size number of observations to fit on.
        :param feature_range: Tuple[int]
            Parameter to be used in MinMaxScaler sklearn's class.

        """

        super().__init__(train_size=train_size)
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def __call__(self, dataset: np.ndarray) -> np.ndarray:
        train = dataset[: self.train_size, :]
        test = dataset[self.train_size :, :]
        train = self.scaler.fit_transform(train)
        test = self.scaler.transform(test)
        return np.concatenate((train, test))

    def inverse_transform(self, dataset: np.ndarray) -> np.ndarray:
        """

        Undoes transformation, gets the initial dataset back.

        :param dataset: np.ndarray
            Scaled by this very scaler dataset.
        :return: np.ndarray

        """

        return self.scaler.inverse_transform(dataset)
