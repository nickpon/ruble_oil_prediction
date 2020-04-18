import numpy as np
from typing import Any, Dict, Optional

from catboost import CatBoostRegressor


def construct_params(
        params: Optional[Dict[str, Any]] = None, use_gpu: bool = True,
) -> Dict[str, Any]:
    if params is None:
        params = {}
    if 'verbose' not in params and 'silent' not in params:
        params['verbose'] = 100
    if use_gpu:
        params['task_type'] = 'GPU'
        if 'devices' not in params:
            params['devices'] = 'cuda:0'
    return params


class BaseCatBoostModel:
    def __init__(
            self,
            use_gpu: bool = False,
            loss_function: str = 'RMSE',
            params: Optional[Dict[str, Any]] = None,
    ):
        self.loss_function = loss_function
        self.params = construct_params(params, use_gpu)

        self.regressor = self._get_regressor()

    def _get_regressor(self) -> CatBoostRegressor:
        return CatBoostRegressor(
            loss_function=self.loss_function, **self.params,
        )

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: Optional[np.ndarray],
            y_val: Optional[np.ndarray],
    ):
        raise NotImplementedError()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_params(self):
        """
        Get parameters of the regressor.
        """

        return self.params
