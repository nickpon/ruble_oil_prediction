import numpy as np

from preprocessors.splitters import Splitter


class TrainValTestSplitter(Splitter):
    def __init__(
            self,
            train_size: int,
            val_size: int,
            dataset: np.ndarray,
            use_tqdm: bool = True,
            max_pred_horizon: int = 28,
            max_train_horizon: int = 30,
    ):
        """

        Splits into train and test part by train_size.

        x_train = [train_size, max_train_horizon, d_size],
        y_train = [train_size, max_pred_horizon, d_size],

        x_val = [val_size, max_train_horizon, d_size],
        y_val = [val_size, max_pred_horizon, d_size],

        x_test = [test_size, max_train_horizon, d_size],
        y_test = [test_size, max_pred_horizon, d_size],

        where
            test_size = dataset.shape[1] - max_train_horizon -
            max_pred_horizon - train_size - val_size

        :param train_size: int
            First train_size number of observations to keep in training part.
        :param val_size: int
            First val_size number of observations to keep in validation part.
            Others will be stored in test part.
        :param dataset: np.ndarray
            Initial dataset to be splitted.
        :param use_tqdm: bool
            Flag that indicates whether one
            should use tqdm progress bars or not.
        :param max_pred_horizon: int
            Number of maximum prediction horizon for each period of time.
        :param max_train_horizon:
            Number of maximum train horizon for each period of time.

        """

        super().__init__(
            dataset=dataset,
            use_tqdm=use_tqdm,
            max_pred_horizon=max_pred_horizon,
            max_train_horizon=max_train_horizon,
        )
        self.train_size = train_size
        self.val_size = val_size
        self._split()

    def _split(self):
        self.x_train = self._make_x_part(begin=0, end=self.train_size)
        self.y_train = self._make_y_part(begin=0, end=self.train_size)
        self.x_val = self._make_x_part(
            begin=self.train_size, end=self.train_size + self.val_size,
        )
        self.y_val = self._make_y_part(
            begin=self.train_size, end=self.train_size + self.val_size,
        )
        self.x_test = self._make_x_part(
            begin=self.train_size + self.val_size,
            end=(
                self.dataset.shape[0]
                - self.max_train_horizon
                - self.max_pred_horizon
            ),
        )
        self.y_test = self._make_y_part(
            begin=self.train_size + self.val_size,
            end=(
                self.dataset.shape[0]
                - self.max_train_horizon
                - self.max_pred_horizon
            ),
        )

    def __call__(self):
        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )
