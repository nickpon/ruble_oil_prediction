import numpy as np
from typing import Any, Dict, Optional

from catboost import CatBoostRegressor

from models.catboost.base_catboost import BaseCatBoostModel


class MultiCatBoost(BaseCatBoostModel):
    def __init__(
            self,
            use_gpu: bool = False,
            params: Optional[Dict[str, Any]] = None,
    ):
        """
        Implementation of the scikit-learn API for CatBoost regression for
        multitarget prediction.

        :param use_gpu: bool, [default=False]
            Flag that indicates whether use use_gpu or not.
        :param params: Optional[Dict[str, Any]], [default=None]
            Dict that contains parameters for CatBoostRegressor.
        """

        # The only function that is supported for multitarget task.
        loss_function = 'MultiRMSE'
        super().__init__(
            use_gpu=use_gpu, loss_function=loss_function, params=params,
        )

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
    ):
        """
        Fit CatBoost Model.

        :param x_train: np.ndarray
            x_train = [train_size, max_train_horizon, d_size]
        :param y_train: np.ndarray
            y_train = [train_size, max_pred_horizon, d_size]
        :param x_val: np.ndarray
            x_val = [val_size, max_train_horizon, d_size]
        :param y_val: np.ndarray
            y_val = [val_size, max_pred_horizon, d_size]
        :return:
        """

        eval_set = (
            (x_val[:, 0, :], y_val[:, 0, :])
            if x_val is not None or y_val is not None
            else None
        )

        self.regressor.fit(
            X=x_train[:, 0, :], y=y_train[:, 0, :], eval_set=eval_set,
        )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Prediction by trained CatBoost model.

        :param x_test: np.ndarray
            x_test = [test_size, max_train_horizon, d_size]
        :return: np.ndarray
            prediction = [test_size, d_size]
        """

        return self.regressor.predict(x_test[:, 0, :])


class GreedyMultiCatBoost(BaseCatBoostModel):
    def __init__(
            self,
            dimension: int,
            loss_function: str = 'RMSE',
            use_gpu: bool = False,
            params: Optional[Dict[str, Any]] = None,
    ):
        """
        Greedy multitarget implementation (create independent regressor
        for each row) of the scikit-learn API for CatBoost regression.

        :param dimension: int
            Number of rows to predict.
        :param use_gpu: bool, [default=False]
            Flag that indicates whether use use_gpu or not.
        :param params: Optional[Dict[str, Any]], [default=None]
            Dict that contains parameters for CatBoostRegressor.
        """

        super().__init__(
            use_gpu=use_gpu, loss_function=loss_function, params=params,
        )

        self.regressors = [
            CatBoostRegressor(loss_function=self.loss_function, **self.params)
            for _ in range(dimension)
        ]

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
    ):
        """
        Fit CatBoost Model.

        :param x_train: np.ndarray
            x_train = [train_size, max_train_horizon, d_size]
        :param y_train: np.ndarray
            y_train = [train_size, max_pred_horizon, d_size]
        :param x_val: np.ndarray
            x_val = [val_size, max_train_horizon, d_size]
        :param y_val: np.ndarray
            y_val = [val_size, max_pred_horizon, d_size]
        :return:
        """

        for row_ind, regressor in enumerate(self.regressors):
            eval_set = (
                (x_val[:, 0, :], y_val[:, 0, row_ind])
                if x_val is not None or y_val is not None
                else None
            )
            regressor.fit(
                X=x_train[:, 0, :],
                y=y_train[:, 0, row_ind],
                eval_set=eval_set,
            )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Prediction by trained CatBoost model.

        :param x_test: np.ndarray
            x_test = [test_size, max_train_horizon, d_size]
        :return: np.ndarray
            prediction = [test_size, d_size]
        """

        return np.transpose(
            np.array(
                [
                    regressor.predict(x_test[:, 0, :])
                    for regressor in self.regressors
                ],
            ),
            (1, 0),
        )
