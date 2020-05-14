from datetime import date
import numpy as np
from typing import Tuple


def x_flat_piecewise(
        x_data: np.ndarray, y_data: np.ndarray, train_horizon: int,
) -> np.ndarray:
    """
    Extrapolate values on saturday and sunday same as on friday.

    :param x_data: np.ndarray
        x_part of the dataset (including values for days on the last axis).
    :param y_data: np.ndarray
        y_part of the dataset (including values for days on the last axis).
    :param train_horizon: int
        Number of timestamps to predict on.
    """

    output_data = []
    for i in range(x_data.shape[0]):
        cur_pos = len(x_data[i]) - 1
        counter = 0
        data = []
        while counter < train_horizon:
            if date.weekday(x_data[i][cur_pos][-1]) == 5:
                data.append(x_data[i][cur_pos - 1][:-1])
            elif date.weekday(x_data[i][cur_pos][-1]) == 6:
                data.append(x_data[i][cur_pos - 2][:-1])
            else:
                data.append(x_data[i][cur_pos][:-1])
            counter += 1
            cur_pos -= 1
        output_data.append(np.array(data[::-1]))
    return np.array(output_data)


def y_flat_piecewise(
        x_data: np.ndarray, y_data: np.ndarray, pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrapolate values on saturday and sunday same as on friday.

    :param x_data: np.ndarray
        x_part of the dataset (including values for days on the last axis).
    :param y_data: np.ndarray
        y_part of the dataset (including values for days on the last axis).
    :param pred_horizon: int
        Number of timestamps to make prediction for.
    """

    output_data = []
    output_dates = []
    for i in range(y_data.shape[0]):
        cur_pos = 0
        counter = 0
        data = []
        dates = []
        while counter < pred_horizon:
            if date.weekday(y_data[i][cur_pos][-1]) == 5:
                if cur_pos == 0:
                    data.append(x_data[i][-1][:-1])
                else:
                    data.append(y_data[i][cur_pos - 1][:-1])
            elif date.weekday(y_data[i][cur_pos][-1]) == 6:
                if cur_pos == 0:
                    data.append(x_data[i][-2][:-1])
                elif cur_pos == 1:
                    data.append(x_data[i][-1][:-1])
                else:
                    data.append(y_data[i][cur_pos + 1][:-1])
            else:
                data.append(y_data[i][cur_pos][:-1])
            dates.append(y_data[i][cur_pos][-1])
            counter += 1
            cur_pos += 1
        output_data.append(np.array(data))
        output_dates.append(np.array(dates))
    return np.array(output_data), np.array(output_dates)


def x_linear_piecewise(
        x_data: np.ndarray, y_data: np.ndarray, train_horizon: int,
) -> np.ndarray:
    """
    Extrapolate values on saturday and sunday by linear function from friday
    to monday.

    :param x_data: np.ndarray
        x_part of the dataset (including values for days on the last axis).
    :param y_data: np.ndarray
        y_part of the dataset (including values for days on the last axis).
    :param train_horizon: int
        Number of timestamps to predict on.
    """

    output_data = []
    for i in range(x_data.shape[0]):
        data = np.concatenate((x_data[i], y_data[i]))
        output = []
        j = 2
        while j < len(x_data[i]) + 1:
            if date.weekday(data[j][-1]) == 5:
                left_data = data[j - 1][:-1]
                right_data = data[j + 2][:-1]

                # y(x) = k * x + b
                k = (right_data - left_data) / 3
                b = left_data - k * (j - 1)
                output.append(k * j + b)
                output.append(k * (j + 1) + b)

                j += 2
            else:
                output.append(data[j][:-1])
                j += 1
        output_data.append(
            np.array(output)[
                len(x_data[i]) - train_horizon - 2 : len(x_data[i]) - 2
            ],
        )
    return np.array(output_data)


def y_linear_piecewise(
        x_data: np.ndarray, y_data: np.ndarray, pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrapolate values on saturday and sunday by linear function from friday
    to monday.

    :param x_data: np.ndarray
        x_part of the dataset (including values for days on the last axis).
    :param y_data: np.ndarray
        y_part of the dataset (including values for days on the last axis).
    :param pred_horizon: int
        Number of timestamps to make prediction for.
    """

    output_data = []
    output_dates = []
    for i in range(x_data.shape[0]):
        data = np.concatenate((x_data[i], y_data[i]))
        output = []
        j = len(x_data[i]) - 2
        while j < len(data) - 2:
            if date.weekday(data[j][-1]) == 5:
                left_data = data[j - 1][:-1]
                right_data = data[j + 2][:-1]

                # y(x) = k * x + b
                k = (right_data - left_data) / 3
                b = left_data - k * (j - 1)
                output.append(k * j + b)
                output.append(k * (j + 1) + b)

                j += 2
            else:
                output.append(data[j][:-1])
                j += 1
        output_data.append(np.array(output)[2 : pred_horizon + 2])
        output_dates.append(
            data[len(x_data[i]) : len(x_data[i]) + pred_horizon, -1],
        )
    return np.array(output_data), np.array(output_dates)


def x_sparse_piecewise(
        x_data: np.ndarray, y_data: np.ndarray, train_horizon: int,
) -> np.ndarray:
    """
    Skip values for saturday and sunday. Note: use prediction and train horizons
    multiple of 5, and multiple of 7 for other methods to perform compatible
    results.

    :param x_data: np.ndarray
        x_part of the dataset (including values for days on the last axis).
    :param y_data: np.ndarray
        y_part of the dataset (including values for days on the last axis).
    :param train_horizon: int
        Number of timestamps to predict on.
    """

    output_data = []
    for i in range(x_data.shape[0]):
        counter = 0
        cur_pos = len(x_data[i]) - 1
        data = []
        while counter < train_horizon:
            if date.weekday(x_data[i][cur_pos][-1]) not in [5, 6]:
                data.append(x_data[i][cur_pos][:-1])
                counter += 1
            cur_pos -= 1
        output_data.append(np.array(data)[::-1])
    return np.array(output_data)


def y_sparse_piecewise(
        x_data: np.ndarray, y_data: np.ndarray, pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skip values for saturday and sunday. Note: use prediction and train horizons
    multiple of 5, and multiple of 7 for other methods to perform compatible
    results.

    :param x_data: np.ndarray
        x_part of the dataset (including values for days on the last axis).
    :param y_data: np.ndarray
        y_part of the dataset (including values for days on the last axis).
    :param pred_horizon: int
        Number of timestamps to make prediction for.
    """

    output_data = []
    output_dates = []
    for i in range(x_data.shape[0]):
        counter = 0
        cur_pos = 0
        data = []
        dates = []
        while counter < pred_horizon:
            if date.weekday(y_data[i][cur_pos][-1]) not in [5, 6]:
                counter += 1
                data.append(y_data[i][cur_pos][:-1])
                dates.append(y_data[i][cur_pos][-1])
            cur_pos += 1
        output_data.append(np.array(data))
        output_dates.append(np.array(dates))
    return np.array(output_data), np.array(output_dates)
