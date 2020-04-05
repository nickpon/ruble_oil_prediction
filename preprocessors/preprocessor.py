import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from utils import get_device

from preprocessors.splitters.train_test_splitter import TrainTestSplitter
from preprocessors.splitters.train_val_test_splitter import (
    TrainValTestSplitter,
)
from preprocessors.scalers.base_scaler import BaseScaler


class Preprocessor:
    def __init__(
            self,
            dataset: np.ndarray,
            max_pred_horizon: int,
            max_train_horizon: int,
            d_size: int,
    ):
        """
        Preprocessor class that makes data splits and scaling over the dataset
        give.

        :param dataset: np.ndarray
            Dataset to preprocess.
            dataset = [num_of_observation, d_size]
        :param max_pred_horizon: int
            Number of maximum prediction horizon for each period of time.
        :param max_train_horizon:
            Number of maximum train horizon for each period of time.
        :param d_size:
            Dimensions (number of rows) to explore.
        """

        self.dataset = dataset[:, :d_size]
        self.max_pred_horizon = max_pred_horizon
        self.max_train_horizon = max_train_horizon
        self.device = get_device()

        self.scaler = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def get_scaled_dataset(self) -> np.ndarray:
        """
        Returns scaled dataset if scale was used, else returns initial dataset.

        :return: np.ndarray
        """

        return self.dataset

    def get_initial_dataset(self) -> np.ndarray:
        """
        Returns initial dataset.

        :return: np.ndarray
        """

        if self.scaler is None:
            return self.dataset
        return self.scaler.inverse_transform(self.dataset)

    def plot_row(self, row: int):
        """
        Plots row of the initial dataset.

        :param row: int
            Index of the row to plot.
        """

        plt.title('{} row'.format(str(row)))
        plt.plot(self.get_initial_dataset()[:, row])
        if self.x_train is not None:
            plt.axvline(x=self.x_train.shape[0])
        if self.x_val is not None:
            plt.axvline(x=self.x_train.shape[0] + self.x_val.shape[0])
        plt.show()

    def get_train_dataloader(self, train_batch_size: int = 128) -> DataLoader:
        """
        Form torch Dataloder on train data.

        :param train_batch_size: int, [default=128]
            Batch-size to use in Dataloder.
        :return: DataLoader
        """

        return DataLoader(
            dataset=TensorDataset(self.x_train, self.y_train),
            shuffle=True,
            batch_size=train_batch_size,
        )

    def get_val_dataloader(self, val_batch_size: int = 128) -> DataLoader:
        """
        Form torch Dataloder on train data.

        :param val_batch_size: int, [default=128]
            Batch-size to use in Dataloder.
        :return: DataLoader
        """

        return DataLoader(
            dataset=TensorDataset(self.x_val, self.y_val),
            shuffle=False,
            batch_size=val_batch_size,
        )

    def get_test_dataloader(self, test_batch_size: int = 128) -> DataLoader:
        """
        Form torch Dataloder on train data.

        :param test_batch_size: int, [default=128]
            Batch-size to use in Dataloder.
        :return: DataLoader
        """

        return DataLoader(
            dataset=TensorDataset(self.x_test, self.y_test),
            shuffle=False,
            batch_size=test_batch_size,
        )

    def train_test_split(
            self,
            train_size: int,
            scaler: Optional[BaseScaler] = None,
            use_tqdm: bool = True,
    ):
        """
        Splits dataset into train and test part by train_size.
        Stores results in x_train, y_train, x_test, y_test field respectively.

        :param train_size: int
            First train_size number of observations to keep in training part.
            Others will be stored in test part.
        :param scaler: Optional[BaseScaler], [default=None]
            Instance of child-class of BaseScaler or None.
            If None, no transformation is performed, else transformation is
            performed with transform() method of the instance given.
        :param use_tqdm: bool, [default=True]
            Flag that indicates whether one
            should use tqdm progress bars or not.
        """

        if scaler is not None:
            self.scaler = scaler
            self.dataset = scaler(self.dataset)

        self.x_train, self.y_train, self.x_test, self.y_test = (
            TrainTestSplitter(
                train_size=train_size,
                dataset=self.dataset,
                use_tqdm=use_tqdm,
                max_pred_horizon=self.max_pred_horizon,
                max_train_horizon=self.max_train_horizon,
            )()
        )

    def train_val_test_split(
            self,
            train_size: int,
            val_size: int,
            scaler: Optional[BaseScaler],
            use_tqdm: bool = True,
    ):
        """
        Splits dataset into train and test part by train_size.
        Stores results in x_train, y_train, x_test, y_test field respectively.

        :param train_size: int
            First train_size number of observations to keep in training part.
        :param val_size: int
            First val_size number of observations to keep in validation part.
            Others will be stored in test part.
        :param scaler: Optional[BaseScaler]
            Instance of child-class of BaseScaler or None.
            If None, no transformation is performed, else transformation is
            performed with transform() method of the instance given.
        :param use_tqdm: bool, [default=True]
            Flag that indicates whether one
            should use tqdm progress bars or not.
        """

        if scaler is not None:
            self.scaler = scaler
            self.dataset = scaler(self.dataset)

        (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        ) = (
            TrainValTestSplitter(
                train_size=train_size,
                val_size=val_size,
                dataset=self.dataset,
                use_tqdm=use_tqdm,
                max_pred_horizon=self.max_pred_horizon,
                max_train_horizon=self.max_train_horizon,
            )()
        )
