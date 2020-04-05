import numpy as np
import tqdm


class Splitter:
    def __init__(
            self,
            dataset: np.ndarray,
            max_pred_horizon: int,
            max_train_horizon: int,
            use_tqdm: bool = True,
    ):
        """
        Base splitter class.

        :param dataset: np.ndarray
            Initial dataset to be splitted.
        :param max_pred_horizon: int
            Number of maximum prediction horizon for each period of time.
        :param max_train_horizon:
            Number of maximum train horizon for each period of time.
        :param use_tqdm: bool, [default=True]
            Flag that indicates whether one
            should use tqdm progress bars or not.
        """

        self.dataset = dataset
        self.use_tqdm = use_tqdm
        self.max_pred_horizon = max_pred_horizon
        self.max_train_horizon = max_train_horizon

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def _make_x_part(self, begin: int, end: int) -> np.ndarray:
        """
        Creates x_{train, val, test} part for dataset's preprocessor.
        Gets elements only with indexes in [begin, end).

        :param begin: int
            Index to begin with.
        :param end: int
            Index to stop at.
        :return: np.ndarray
            x_{train, val, test} part:
                [end-begin, max_train_horizon, d_size]
        """

        x = self.dataset[begin : begin + self.max_train_horizon, :][
            np.newaxis, :, :,
        ]
        if self.use_tqdm:
            for i in tqdm.tqdm(range(begin + 1, end)):
                new_elem = self.dataset[i : i + self.max_train_horizon, :][
                    np.newaxis, :, :,
                ]
                x = np.concatenate((x, new_elem))
        else:
            for i in range(begin + 1, end):
                new_elem = self.dataset[i : i + self.max_train_horizon, :][
                    np.newaxis, :, :,
                ]
                x = np.concatenate((x, new_elem))
        return x

    def _make_y_part(self, begin: int, end: int) -> np.ndarray:
        """
        Creates y_{train, val, test} part for dataset's preprocessor.
        Gets elements only with indexes in [begin, end).

        :param begin: int
            Index to begin with.
        :param end: int
            Index to stop at.
        :return: np.ndarray
            y_{train, val, test} part:
                [end-begin, max_pred_horizon, d_size]
        """

        y = self.dataset[
            begin
            + self.max_train_horizon : begin
            + self.max_train_horizon
            + self.max_pred_horizon,
            :,
        ][np.newaxis, :, :]
        if self.use_tqdm:
            for i in tqdm.tqdm(
                    range(
                        begin + 1 + self.max_train_horizon,
                        end + self.max_train_horizon,
                    ),
            ):
                new_elem = self.dataset[i : i + self.max_pred_horizon, :][
                    np.newaxis, :, :,
                ]
                y = np.concatenate((y, new_elem))
        else:
            for i in range(
                    begin + 1 + self.max_train_horizon,
                    end + self.max_train_horizon,
            ):
                new_elem = self.dataset[i : i + self.max_pred_horizon, :][
                    np.newaxis, :, :,
                ]
                y = np.concatenate((y, new_elem))
        return y
