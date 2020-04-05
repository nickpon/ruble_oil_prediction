import numpy as np
from typing import Any, Dict, Optional

from models.catboost.base_catboost import BaseCatBoostModel


class UnionCatBoost(BaseCatBoostModel):
    def __init__(
            self,
            row_ind: int,
            use_gpu: bool = False,
            loss_function: str = 'RMSE',
            params: Optional[Dict[str, Any]] = None,
    ):
        """
        Implementation of the scikit-learn API for CatBoost regression for
        uniontarget prediction regarding row_ind row index.

        :param row_ind: int,
            Row index to predict.
        :param use_gpu: bool, [default=False]
            Flag that idicates whether use use_gpu or not.
        :param loss_function: string, [default='RMSE']
            'RMSE'
            'MAE'
            'Quantile:alpha=value'
            'LogLinQuantile:alpha=value'
            'Poisson'
            'MAPE'
            'Lq:q=value'
        :param params: Optional[Dict[str, Any]], [default=None]
            Dict that contains parameters for CatBoostRegressor.
        """

        super().__init__(
            use_gpu=use_gpu, loss_function=loss_function, params=params,
        )
        self.row_ind = row_ind

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
            (x_val[:, 0, :], y_val[:, 0, self.row_ind])
            if x_val is not None or y_val is not None
            else None
        )

        self.regressor.fit(
            X=x_train[:, 0, :],
            y=y_train[:, 0, self.row_ind],
            eval_set=eval_set,
        )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Prediction by trained CatBoost model.

        :param x_test: np.ndarray
            x_test = [test_size, max_train_horizon, d_size]
        :return: np.ndarray
            Prediction for row_ind row.
            prediction = [test_size,]
        """

        return self.regressor.predict(x_test[:, 0, :])
