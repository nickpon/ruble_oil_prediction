from datetime import date
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional

from preprocessors.preprocessor import Preprocessor

from models.catboost.base_catboost import BaseCatBoostModel


class Evaluator:
    def __init__(
            self,
            regressor: Callable,
            preprocessor: Preprocessor,
            function_x_train: Callable,
            function_y_train: Callable,
            function_x_val: Callable,
            function_y_val: Callable,
            function_x_test: Callable,
            function_y_test: Callable,
            train_horizons: List[int],
            prediction_horizons: List[int],
            metric: Callable,
            metric_name: str,
            loss_function: str,
            grid_search_params: Optional[Dict[str, List[Any]]] = None,
            use_gpu: bool = False,
    ):
        """
        Evaluator class for training and finding best params for CatBoost model.

        :param regressor: Callable
            Instance of BaseCatBoostModel class.
        :param preprocessor: Preprocessor
            Instance of Preprocessor class.
        :param function_x_train: Callable
            Function to reshape train x_part of dataset.
        :param function_y_train: Callable
            Function to reshape train y_part of dataset.
        :param function_x_val: Callable
            Function to reshape val x_part of dataset.
        :param function_y_val: Callable
            Function to reshape val y_part of dataset.
        :param function_x_test: Callable
            Function to reshape test x_part of dataset.
        :param function_y_test: Callable
            Function to reshape test y_part of dataset.
        :param train_horizons: int
            Number of train horizon for each period of time.
        :param prediction_horizons:
            Number of prediction horizon for each period of time.
        :param metric: Callable
            Function to evaluate corresponding metric.
        :param metric_name: str
            Name of metric function.
        :param loss_function: str
            Name of loss function to use in CatBoost estimator.
        :param grid_search_params: Optional[Dict[str, List[Any]]],
            [default=None]
            Params to perform grid-serach over.
        :param use_gpu: bool, [default=False]
            Flag that indicates whether use use_gpu or not.
        """

        self.regressor = regressor
        self.preprocessor = preprocessor
        self.function_x_train = function_x_train
        self.function_y_train = function_y_train
        self.function_x_val = function_x_val
        self.function_y_val = function_y_val
        self.function_x_test = function_x_test
        self.function_y_test = function_y_test
        self.train_horizons = train_horizons
        self.prediction_horizons = prediction_horizons
        self.metric = metric
        self.metric_name = metric_name
        self.loss_function = loss_function
        self.grid_search_params = grid_search_params
        if grid_search_params is None:
            self.grid_search_params = {}
        else:
            self.grid_search_params = grid_search_params
        self.use_gpu = use_gpu

        self.best_params = {}
        self.metric_values = {}
        self.best_model = {}
        self.shift = {}

    def _train_model(
            self,
            train_horizon: int,
            pred_horizon: int,
            params: Optional[Dict[str, Any]] = None,
    ) -> BaseCatBoostModel:
        """
        Train CatBoost model.

        :param train_horizon: int
            Number of train horizon for each period of time.
        :param pred_horizon:
            Number of prediction horizon for each period of time.
        :param params: Optional[Dict[str, Any]], [default=None]
            Additional params for CatBoost estimator.
        :return: BaseCatBoostModel
        """
        cur_regressor = self.regressor(
            use_gpu=self.use_gpu,
            loss_function=self.loss_function,
            params=params,
            dimension=pred_horizon,
        )
        cur_regressor.fit(
            x_train=self.function_x_train(
                self.preprocessor.get_x_part('train', train_horizon),
            ),
            y_train=self.function_y_train(
                self.preprocessor.get_y_part('train', pred_horizon)[0],
            ),
            x_val=self.function_x_val(
                self.preprocessor.get_x_part('val', train_horizon),
            ),
            y_val=self.function_y_val(
                self.preprocessor.get_y_part('val', pred_horizon)[0],
            ),
        )
        return cur_regressor

    def _get_best_shift(
            self,
            model: BaseCatBoostModel,
            train_horizon: int,
            pred_horizon: int,
    ) -> (int, float):
        prediction = model.predict(
            x_test=self.function_x_val(
                self.preprocessor.get_x_part('val', train_horizon),
            ),
        )
        y_val, dates_val = self.preprocessor.get_y_part('val', pred_horizon)

        y_val_work_days = []
        for ind, y_elem in enumerate(y_val):
            y_elements = []
            for num, value in enumerate(y_elem):
                if date.weekday(dates_val[ind][num]) not in [5, 6]:
                    y_elements.append(value)
            y_val_work_days.append(
                self.function_y_val(np.array(y_elements)[np.newaxis, :, :]),
            )
        y_val_work_days = np.array(y_val_work_days)

        if len(y_val_work_days.shape) == 3:
            y_val_work_days = y_val_work_days.squeeze(axis=1)

            pred_val_work_days = []
            for ind, y_elem in enumerate(prediction):
                y_elements = []
                for num, value in enumerate(y_elem):
                    if date.weekday(dates_val[ind][num]) not in [5, 6]:
                        y_elements.append(value)
                pred_val_work_days.append(y_elements)
            prediction = np.array(pred_val_work_days)

        print(y_val_work_days.shape)
        print(prediction.shape)

        ans = [self.metric(y_pred=prediction, y_true=y_val_work_days)]
        for k in range(1, 100):
            ans.append(
                self.metric(
                    y_pred=prediction[k:], y_true=y_val_work_days[:-k],
                ),
            )
        return ans.index(min(ans)), min(ans)

    def _get_model_metric_value(
            self,
            train_horizon: int,
            pred_horizon: int,
            params: Optional[Dict[str, Any]] = None,
    ) -> (int, float, BaseCatBoostModel):
        """
        Get model metric value.

        :param train_horizon: int
            Number of train horizon for each period of time.
        :param pred_horizon:
            Number of prediction horizon for each period of time.
        :param params: Optional[Dict[str, Any]], [default=None]
            Additional params for CatBoost estimator.
        :return: int, float, BaseCatBoostModel
            Returns shift, metric_value and trained model.
        """

        model = self._train_model(
            train_horizon=train_horizon,
            pred_horizon=pred_horizon,
            params=params,
        )
        shift, metric_value = self._get_best_shift(
            model=model,
            train_horizon=train_horizon,
            pred_horizon=pred_horizon,
        )
        print('Best shift:', shift)
        return shift, metric_value, model

    def evaluate_params(self):
        """
        Find best model for each parameter.
        """

        for train_hor in self.train_horizons:
            self.metric_values[train_hor] = {}
            self.best_params[train_hor] = {}
            self.best_model[train_hor] = {}
            self.shift[train_hor] = {}
            for pred_hor in self.prediction_horizons:
                self.metric_values[train_hor][pred_hor] = np.inf
                self.best_params[train_hor][pred_hor] = {}
                self.best_model[train_hor][pred_hor] = None
                self.shift[train_hor][pred_hor] = 0
                for values in list(
                        itertools.product(*self.grid_search_params.values()),
                ):
                    params = {}
                    for item, key in enumerate(self.grid_search_params.keys()):
                        params[key] = values[item]
                    print('Train_horizon:', train_hor)
                    print('Prediction_horizon:', pred_hor)
                    print('Estimator params:', params)
                    shift, metric_value, model = self._get_model_metric_value(
                        train_horizon=train_hor,
                        pred_horizon=pred_hor,
                        params=params,
                    )
                    if metric_value < self.metric_values[train_hor][pred_hor]:
                        self.metric_values[train_hor][pred_hor] = metric_value
                        self.best_params[train_hor][pred_hor] = params
                        self.best_model[train_hor][pred_hor] = model
                        self.shift[train_hor][pred_hor] = shift
                print(
                    'Best estimator params:',
                    self.best_params[train_hor][pred_hor],
                )

    def form_result_table(
            self,
            metric_names: List[str],
            metrics: List[Callable],
            part: str,
            is_real_data: bool,
    ):
        """
        Print result table with values of metric for the best estimators.

        :param metric_names: List[str]
            Names of metric functions.
        :param metrics: List[Callable]
            Functions to evaluate corresponding metrics.
        :param part: str
            Name of the part to get (train, val, test).
        :param is_real_data:
            Whether to evaluate on real data or preprocessed.
        """

        assert len(metric_names) == len(metrics)

        print('Loss function:', self.loss_function)

        for ind in range(len(metrics)):
            print('Metric:', metric_names[ind])

            data = []
            for train_hor in self.train_horizons:
                row = []
                for pred_hor in self.prediction_horizons:
                    model = self.best_model[train_hor][pred_hor]
                    if part == 'train':
                        x_test = self.function_x_train(
                            self.preprocessor.get_x_part('train', train_hor),
                        )
                        actual = self.function_y_train(
                            self.preprocessor.get_y_part('train', pred_hor)[0],
                        )
                    elif part == 'val':
                        x_test = self.function_x_val(
                            self.preprocessor.get_x_part('val', train_hor),
                        )
                        actual = self.function_y_val(
                            self.preprocessor.get_y_part('val', pred_hor)[0],
                        )
                    else:
                        x_test = self.function_x_test(
                            self.preprocessor.get_x_part('test', train_hor),
                        )
                        actual = self.function_y_test(
                            self.preprocessor.get_y_part('test', pred_hor)[0],
                        )
                    predicted = model.predict(x_test=x_test)
                    if is_real_data:
                        actual = (
                            self.preprocessor.scaler.inverse_transform_target(
                                actual,
                            )
                        )
                        predicted = (
                            self.preprocessor.scaler.inverse_transform_target(
                                predicted,
                            )
                        )
                    if self.shift[train_hor][pred_hor] != 0:
                        actual = actual[: -self.shift[train_hor][pred_hor]]
                        predicted = predicted[
                            self.shift[train_hor][pred_hor] :
                        ]
                    row.append(metrics[ind](actual, predicted))
                data.append(row)

            print(
                pd.DataFrame(
                    data=data,
                    index=pd.Index(self.train_horizons, 'cols'),
                    columns=pd.Index(self.prediction_horizons, 'rows'),
                ),
            )
            print('_____________________')

    def plot_union(
            self,
            train_horizon: int,
            pred_horizon: int,
            start_ind: int = 0,
            end_ind: Optional[int] = None,
    ):
        """
        Plot representation for union estimator.

        :param train_horizon: int
            Number of train horizon for each period of time.
        :param pred_horizon:
            Number of prediction horizon for each period of time.
        :param start_ind:
            Index to start from.
        :param end_ind:
            Index to end at.
        """

        if end_ind is None:
            end_ind = self.preprocessor.dataset.shape[0]
        model = self.best_model[train_horizon][pred_horizon]
        predicted = model.predict(
            x_test=self.function_x_train(
                self.preprocessor.get_x_part('train', train_horizon),
            ),
        )
        predicted = np.concatenate(
            [
                predicted,
                model.predict(
                    x_test=self.function_x_val(
                        self.preprocessor.get_x_part('val', train_horizon),
                    ),
                ),
            ],
        )
        predicted = np.concatenate(
            [
                predicted,
                model.predict(
                    x_test=self.function_x_test(
                        self.preprocessor.get_x_part('test', train_horizon),
                    ),
                ),
            ],
        )
        predicted = predicted[self.shift[train_horizon][pred_horizon] :]
        predicted = predicted[start_ind:end_ind]

        actual = self.function_y_train(
            self.preprocessor.get_y_part('train', pred_horizon)[0],
        )
        actual = np.concatenate(
            [
                actual,
                self.function_y_val(
                    self.preprocessor.get_y_part('val', pred_horizon)[0],
                ),
            ],
        )
        actual = np.concatenate(
            [
                actual,
                self.function_y_test(
                    self.preprocessor.get_y_part('test', pred_horizon)[0],
                ),
            ],
        )
        actual = actual[
            : actual.shape[0] - self.shift[train_horizon][pred_horizon]
        ]
        actual = actual[start_ind:end_ind]
        plt.plot(
            [i + start_ind for i in range(len(actual))],
            self.preprocessor.scaler.inverse_transform_target(actual),
            label='Actual',
        )
        plt.plot(
            [i + start_ind for i in range(len(actual))],
            self.preprocessor.scaler.inverse_transform_target(predicted),
            label='Predicted',
        )
        if start_ind < self.preprocessor.x_train.shape[0] < end_ind:
            plt.axvline(x=self.preprocessor.x_train.shape[0])
        if (
                start_ind
                < self.preprocessor.x_train.shape[0]
                + self.preprocessor.x_val.shape[0]
                < end_ind
        ):
            plt.axvline(
                x=self.preprocessor.x_train.shape[0]
                + self.preprocessor.x_val.shape[0],
            )
        plt.legend()
        plt.show()

    # TODO:
    def plot_multi(
            self, train_horizon: int, pred_horizon: int, start_ind: int,
    ):
        """
        Plot representation for multi-target estimator.

        :param train_horizon: int
            Number of train horizon for each period of time.
        :param pred_horizon:
            Number of prediction horizon for each period of time.
        :param start_ind:
            Index to look from.
        """

        train_size = self.preprocessor.x_train.shape[0]
        val_size = self.preprocessor.x_val.shape[0]
        model = self.best_model[train_horizon][pred_horizon]

        if start_ind < train_size:
            x_part = self.preprocessor.get_x_part('train', train_horizon)
            predicted = self.preprocessor.scaler.inverse_transform_target(
                model.predict(x_test=self.function_x_train(x_part))[start_ind],
            )
            actual = self.preprocessor.scaler.inverse_transform_target(
                self.function_y_train(
                    self.preprocessor.get_y_part('train', pred_horizon)[0],
                )[start_ind],
            )
            train = self.preprocessor.scaler.inverse_transform_target(
                x_part[start_ind, :, 0],
            )
        elif train_size <= start_ind < train_size + val_size:
            x_part = self.preprocessor.get_x_part('val', train_horizon)
            predicted = self.preprocessor.scaler.inverse_transform_target(
                model.predict(x_test=self.function_x_val(x_part))[
                    start_ind - train_size
                ],
            )
            actual = self.preprocessor.scaler.inverse_transform_target(
                self.function_y_val(
                    self.preprocessor.get_y_part('val', pred_horizon)[0],
                )[start_ind - train_size],
            )
            train = self.preprocessor.scaler.inverse_transform_target(
                x_part[start_ind - train_size, :, 0],
            )
        else:
            x_part = self.preprocessor.get_x_part('test', train_horizon)
            predicted = self.preprocessor.scaler.inverse_transform_target(
                model.predict(x_test=self.function_x_test(x_part))[
                    start_ind - train_size - val_size
                ],
            )
            actual = self.preprocessor.scaler.inverse_transform_target(
                self.function_y_test(
                    self.preprocessor.get_y_part('test', pred_horizon)[0],
                )[start_ind - train_size - val_size],
            )
            train = self.preprocessor.scaler.inverse_transform_target(
                x_part[start_ind - train_size - val_size, :, 0],
            )

        plt.plot(np.concatenate([train, actual]), label='Actual')
        plt.plot(np.concatenate([train, predicted]), label='Predicted')
        plt.axvline(x=train_horizon - 1)
        plt.legend()
        plt.show()
