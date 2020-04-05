import numpy as np

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
        self.params['init_data'] = dataset[0, :]
        return np.diff(np.log(dataset), axis=0)

    def inverse_transform(self, dataset: np.ndarray) -> np.ndarray:
        """
        Undoes transformation, gets the initial dataset back.

        :param dataset: np.ndarray
            Scaled by this very scaler dataset.
        :return: np.ndarray
        """

        ans = self.params['init_data'][np.newaxis, :]
        for i in range(1, dataset.shape[0]):
            new_elem = np.exp(dataset[i, :][np.newaxis, :]) * ans[-1, :]
            ans = np.concatenate((ans, new_elem))
        return ans
