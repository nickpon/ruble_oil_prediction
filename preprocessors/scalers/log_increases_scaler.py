import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from preprocessors.scalers.base_scaler import BaseScaler


class LogIncreasesScalerData(BaseScaler):
    def __init__(self, train_size: int):
        """
        Performs LogIncreaseScale on the dataset given.
        Fit() is performed on first train_size number of observations.

        z[0] = ln(x[1]) - ln(x[0])

        :param train_size: int
            First train_size number of observations to fit on.
        """

        super().__init__(train_size=train_size)

    def __call__(self, dataset: np.ndarray) -> np.ndarray:
        self.params['d_size'] = dataset.shape[1]
        self.params['init_data'] = dataset[0, :]
        self.params['min_max_scaler'] = MinMaxScaler()

        data = np.diff(
            np.log(np.array(dataset[:, :-1], dtype=np.float)), axis=0,
        )
        train = self.params['min_max_scaler'].fit_transform(
            data[: self.train_size],
        )
        test = self.params['min_max_scaler'].transform(data[self.train_size :])
        data = np.concatenate((train, test), axis=0)

        dates = np.array(
            list(
                map(
                    lambda x: datetime.datetime.strptime(x, '%m/%d/%y'),
                    dataset[:, -1],
                ),
            ),
        )[:, np.newaxis]
        return np.concatenate((data, dates[1:, :]), axis=1)

    def inverse_transform(self, dataset: np.ndarray) -> np.ndarray:
        """
        Undoes transformation, gets the initial dataset back.

        :param dataset: np.ndarray
            Scaled by this very scaler dataset.
        """

        ans = self.params['init_data'][:-1][np.newaxis, :]
        dataset = self.params['min_max_scaler'].inverse_transform(
            np.array(dataset[:, :-1], dtype=np.float),
        )
        for i in range(1, dataset.shape[0]):
            new_elem = np.exp(dataset[i][np.newaxis, :]) * ans[-1, :]
            ans = np.concatenate((ans, new_elem))
        return ans

    def inverse_transform_target(self, dataset: np.ndarray) -> np.ndarray:
        """
        Undoes transformation, gets the initial values for the first axis
        (target) back.

        :param dataset: np.ndarray
            Scaled by this very scaler dataset.
        :return: np.ndarray
        """

        data = self.params['min_max_scaler'].inverse_transform(
            np.concatenate(
                (
                    np.array(dataset)[:, np.newaxis],
                    np.zeros((dataset.shape[0], self.params['d_size'] - 2)),
                ),
                axis=1,
            ),
        )[:, 0]

        ans = [self.params['init_data'][0]]
        for i in range(1, dataset.shape[0]):
            new_elem = np.exp(data[i]) * ans[-1]
            ans.append(new_elem)
        return np.array(ans)
