import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from typing import Callable, Dict, Tuple, Union

import torch
from torch import optim
from torch.autograd import Variable

from preprocessors.preprocessor import Preprocessor

from models.darnn.decoder import Decoder
from models.darnn.encoder import Encoder


class DARNN:
    def __init__(
            self,
            preprocessor: Preprocessor,
            out_features: int,
            encoder_input_size: int,
            loss_function: Callable,
            metrics: Dict[str, Callable],
            target_function: Callable,
            train_horizon: int,
            pred_horizon: int,
            encoder_hidden_size: int = 64,
            decoder_hidden_size: int = 64,
            learning_rate: float = 0.001,
            plot_frequency: int = 5,
            path_to_save_weights: str = None,
            path_to_load_weights: str = None,
    ):
        """
        Usefull implementation of DARNN model
        (https://arxiv.org/pdf/1704.02971.pdf). It can be both used for
        univariate and multivariate time series.

        :param preprocessor: Preprocessor
            Instance of Preprocessor class.
        :param out_features: int
            Number of values to predics (1 for univariate series,
             > 1 for multivariate series).
        :param encoder_input_size: int
            Dimension of series to predict on.
        :param loss_function: Callable
            Loss function to use while learning,
        :param metrics: Dict[str, Callable],
            Dictionary of metric_name and it's callable implementation.
        :param target_function: Callable
            Target function to reshape dataset for specific task.
        :param train_horizon: int
            Number of train horizon for each period of time.
        :param pred_horizon: int
            Number of prediction horizon for each period of time.
        :param encoder_hidden_size: int, [default=64]
            Dimension of the hidden state in encoder.
        :param decoder_hidden_size: int, [default=64]
            Dimension of the hidden state in decoder.
        :param learning_rate:float, [default=0.001]
            Learning rate of encoder and decoder.
        :param plot_frequency: int, [default=5]
            After how many epochs result plot for univariate series ought to be
            shown. Every 10000 iterations it is decreased like this:
            lr_new = 0.9 * lr_old.
        :param path_to_save_weights: str, [default=None]
            Path to directory to save weights to (weights of encoder and
            decoder will be saved under encoder.pth and decoder.pth names
            correspondingly).
        :param path_to_load_weights: str, [default=None]
            Path to directory to load weights from (must contain encoder.pth
            and decoder.pth files).
        """

        self.preprocessor = preprocessor
        self.out_features = out_features
        self.loss_function = loss_function
        self.metrics = metrics
        self.target_function = target_function
        self.train_horizon = train_horizon
        self.pred_horizon = pred_horizon
        self.encoder_input_size = encoder_input_size
        self.plot_frequency = plot_frequency
        self.path_to_save_weights = path_to_save_weights
        self.path_to_load_weights = path_to_load_weights

        self.encoder = Encoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            t=train_horizon,
        )
        self.decoder = Decoder(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            t=train_horizon,
            out_features=self.out_features,
        )
        if self.path_to_load_weights is not None:
            self.encoder.load_state_dict(
                torch.load(
                    os.path.join(self.path_to_load_weights, 'encoder.pth'),
                ),
            )
            self.decoder.load_state_dict(
                torch.load(
                    os.path.join(self.path_to_load_weights, 'decoder.pth'),
                ),
            )
            print(f'Weights have been loaded from {self.path_to_load_weights}')

        self.encoder_optimizer = optim.Adam(
            params=filter(
                lambda p: p.requires_grad, self.encoder.parameters(),
            ),
            lr=learning_rate,
        )
        self.decoder_optimizer = optim.Adam(
            params=filter(
                lambda p: p.requires_grad, self.decoder.parameters(),
            ),
            lr=learning_rate,
        )

        self.epoch_losses = None
        self.best_geom_mean_metrics = np.inf
        self.best_metrics = {}

        (
            self.x_train,
            self.y_history_train,
            self.y_target_train,
        ) = self._get_data(part='train')
        self.x_val, self.y_history_val, self.y_target_val = self._get_data(
            part='val',
        )
        self.x_test, self.y_history_test, self.y_target_test = self._get_data(
            part='test',
        )

    def _get_data_dataloader(
            self,
            x: Union[torch.Tensor, np.ndarray],
            y: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(x, np.ndarray):
            x = x.numpy()
        if not isinstance(y, np.ndarray):
            y = y.numpy()
        y = y[:, :, 0]
        y_proc = []
        for t in range(self.train_horizon):
            y_proc.append(
                self.target_function(y[:, t : t + self.pred_horizon]),
            )
        if self.out_features > 1:
            y_proc = np.moveaxis(np.moveaxis(np.array(y_proc), 0, -1), 1, -1)
        else:
            y_proc = np.moveaxis(np.array(y_proc), 0, -1)
        y_history_out = y_proc[:, :-1]
        y_target_out = y_proc[:, -1]
        x_out = x[:, x.shape[1] - self.train_horizon + 1 :, 1:]
        return x_out, y_history_out, y_target_out

    def _get_train_data_dataloader(
            self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, y_histories, y_targets = [], [], []
        for X, y in self.preprocessor.get_train_dataloader(
                train_horizon=self.train_horizon,
                pred_horizon=self.train_horizon + self.pred_horizon,
        ):
            x_out, y_history_out, y_target_out = self._get_data_dataloader(
                X, y,
            )
            xs.append(x_out)
            y_histories.append(y_history_out)
            y_targets.append(y_target_out)
        return np.array(xs), np.array(y_histories), np.array(y_targets)

    def _get_data(
            self, part: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._get_data_dataloader(
            self.preprocessor.get_x_part(
                name=part, train_horizon=self.train_horizon,
            ),
            self.preprocessor.get_y_part(
                name=part, pred_horizon=self.train_horizon + self.pred_horizon,
            )[0],
        )

    def _get_predicted_and_actual_data(
            self, part: str,
    ) -> Tuple[torch.tensor, torch.tensor]:
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if part == 'train':
                x, y_history, y_target = (
                    self.x_train,
                    self.y_history_train,
                    self.y_target_train,
                )
            elif part == 'val':
                x, y_history, y_target = (
                    self.x_val,
                    self.y_history_val,
                    self.y_target_val,
                )
            else:
                x, y_history, y_target = (
                    self.x_test,
                    self.y_history_test,
                    self.y_target_test,
                )

            input_weighted, input_encoded = self.encoder(
                Variable(
                    torch.from_numpy(
                        x[:, :, : self.encoder_input_size].astype(np.float64),
                    ).type(torch.FloatTensor),
                ),
            )

            y_pred = (
                self.decoder(
                    input_encoded,
                    Variable(
                        torch.from_numpy(y_history.astype(np.float64)).type(
                            torch.FloatTensor,
                        ),
                    ),
                )
                .cpu()
                .data.numpy()
            )
            if self.out_features == 1:
                y_pred = y_pred[:, 0]
            return y_target, y_pred

    def _train_iteration(self, X, y_history, y_target):
        self.encoder.train()
        self.decoder.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor)),
        )
        y_pred = self.decoder(
            input_encoded,
            Variable(torch.from_numpy(y_history).type(torch.FloatTensor)),
        )
        y_true = Variable(
            torch.from_numpy(y_target).type(torch.FloatTensor),
        ).reshape(y_target.shape[0], self.out_features)
        # loss = self.loss_function(y_pred[:, 0], y_true[:, 0])
        loss = self.loss_function(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if len(loss.data.shape) > 0:
            return loss.data[0]
        return loss.data.reshape(-1, 1)[0]

    def _calculate_metrics(
            self, y_target: np.ndarray, y_pred: np.array, is_real_data: bool,
    ) -> Dict[str, float]:
        if is_real_data:
            if self.out_features > 1:
                y_target = np.moveaxis(
                    np.array(
                        [
                            self.preprocessor.scaler.inverse_transform_target(
                                y_target[:, i],
                            )
                            for i in range(self.out_features)
                        ],
                    ),
                    0,
                    -1,
                )
                y_pred = np.moveaxis(
                    np.array(
                        [
                            self.preprocessor.scaler.inverse_transform_target(
                                y_pred[:, i],
                            )
                            for i in range(self.out_features)
                        ],
                    ),
                    0,
                    -1,
                )
            else:
                y_target = self.preprocessor.scaler.inverse_transform_target(
                    y_target,
                )
                y_pred = self.preprocessor.scaler.inverse_transform_target(
                    y_pred,
                )
        return {
            metric_name: metric(y_target, y_pred)
            for metric_name, metric in self.metrics.items()
        }

    def train(self, n_epochs: int, is_real_data: bool):
        """
        Function to start training for both univariate and multivariate tasks.

        :param n_epochs: int
            Number of epochs to run over.
        :param is_real_data: bool
            Whether to evaluate on real data or preprocessed.
        """

        self.epoch_losses = np.zeros(n_epochs)
        n_iter = 0
        for i in range(n_epochs):
            j = 0
            iter_losses = []

            xs, y_histories, y_targets = self._get_train_data_dataloader()

            for step in range(len(xs)):
                x = xs[step]
                y_history = y_histories[step]
                y_target = y_targets[step]
                loss = self._train_iteration(x, y_history, y_target)
                iter_losses.append(loss)
                j += 1
                n_iter += 1

            if n_iter % 10000 == 0 and n_iter > 0:
                for param_group in self.encoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in self.decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                    print(f"Learning rate is decreased to {param_group['lr']}")

            self.epoch_losses[i] = np.mean(np.array(iter_losses))

            print('Epoch:', i, 'Epoch loss:', self.epoch_losses[i])

            y_target_val, y_pred_val = self._get_predicted_and_actual_data(
                part='val',
            )
            metrics_val = self._calculate_metrics(
                y_target=y_target_val,
                y_pred=y_pred_val,
                is_real_data=is_real_data,
            )

            geom_mean_metrics = stats.gmean(list(metrics_val.values()))
            if geom_mean_metrics < self.best_geom_mean_metrics:
                print(
                    f'Better model with geom mean metrics '
                    f'{geom_mean_metrics} found',
                )
                self.best_geom_mean_metrics = geom_mean_metrics
                (
                    y_target_test,
                    y_pred_test,
                ) = self._get_predicted_and_actual_data(part='test')

                self.best_metrics = self._calculate_metrics(
                    y_target=y_target_test,
                    y_pred=y_pred_test,
                    is_real_data=is_real_data,
                )
                self.best_metrics = {
                    metric_name: (
                        metric_value[0]
                        if isinstance(metric_value, tuple)
                        else metric_value
                    )
                    for metric_name, metric_value in self.best_metrics.items()
                }
                print('Metrics:', self.best_metrics)
                if self.path_to_save_weights is not None:
                    torch.save(
                        self.encoder.state_dict(),
                        os.path.join(self.path_to_save_weights, 'encoder.pth'),
                    )
                    torch.save(
                        self.decoder.state_dict(),
                        os.path.join(self.path_to_save_weights, 'decoder.pth'),
                    )
                    print(
                        f'Model has been saved in {self.path_to_save_weights}',
                    )

            if i % self.plot_frequency == 0 and self.out_features == 1:
                self.plot_union()

    def plot_union(self):
        """
        Plots results for univariate series.
        """

        assert self.out_features == 1
        y_target_train, y_pred_train = self._get_predicted_and_actual_data(
            part='train',
        )
        y_target_val, y_pred_val = self._get_predicted_and_actual_data(
            part='val',
        )
        y_target_test, y_pred_test = self._get_predicted_and_actual_data(
            part='test',
        )
        plt.plot(
            np.concatenate((y_target_train, y_target_val, y_target_test)),
            label='true',
        )
        plt.plot(y_pred_train, label='pred_train')
        plt.plot(
            range(len(y_pred_train), len(y_pred_train) + len(y_pred_val)),
            y_pred_val,
            label='pred_val',
        )
        plt.plot(
            range(
                len(y_pred_train) + len(y_pred_val),
                len(y_pred_train) + len(y_pred_val) + len(y_pred_test),
            ),
            y_pred_test,
            label='pred_test',
        )
        plt.legend()
        plt.show()

    def plot_multi(self, part: str, index: int, is_real_data: bool):
        """
        Plots representations for multivariate estimators.

        :param part: str
            Indicates which part of the dataset to use.
        :param index: int
            Index to look from.
        :param is_real_data: bool
            Whether to evaluate on real data or preprocessed.
        """

        assert self.out_features > 1
        y_target, y_pred = self._get_predicted_and_actual_data(part=part)
        if is_real_data:
            y_target = self.preprocessor.scaler.inverse_transform_target(
                y_target[index],
            )
            y_pred = self.preprocessor.scaler.inverse_transform_target(
                y_pred[index],
            )
        else:
            y_target = y_target[index]
            y_pred = y_pred[index]
        plt.plot(y_target, label='target')
        plt.plot(y_pred, label='predicted')

        plt.legend()
        plt.show()
