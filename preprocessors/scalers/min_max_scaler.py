import datetime
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
        :param feature_range: Tuple[int], [default=(0, 1)]
            Parameter to be used in MinMaxScaler sklearn's class.
        """

        super().__init__(train_size=train_size)
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def __call__(self, dataset: np.ndarray) -> np.ndarray:
        self.params['d_size'] = dataset.shape[1]

        train = dataset[: self.train_size, :-1]
        test = dataset[self.train_size :, :-1]

        train = self.scaler.fit_transform(train)
        test = self.scaler.transform(test)
        data = np.concatenate((train, test))

        dates = np.array(
            list(
                map(
                    lambda x: datetime.datetime.strptime(x, '%m/%d/%y'),
                    dataset[:, -1],
                ),
            ),
        )[:, np.newaxis]

        return np.concatenate((data, dates), axis=1)

    def inverse_transform(self, dataset: np.ndarray) -> np.ndarray:
        """
        Undoes transformation, gets the initial dataset back.

        :param dataset: np.ndarray
            Scaled by this very scaler dataset.
        :return: np.ndarray
        """

        return self.scaler.inverse_transform(dataset[:, :-1])

    def inverse_transform_target(self, dataset: np.ndarray) -> np.ndarray:
        """
        Undoes transformation, gets the initial values for the first axis
        (target) back.

        :param dataset: np.ndarray
            Scaled by this very scaler dataset.
        :return: np.ndarray
        """

        ans = np.concatenate(
            (
                np.array(dataset)[:, np.newaxis],
                np.zeros((dataset.shape[0], self.params['d_size'] - 2)),
            ),
            axis=1,
        )
        return self.scaler.inverse_transform(ans)[:, 0]
