import numpy as np
from sklearn.metrics import mean_squared_error
import torch


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        print('Using cpu')
    else:
        print('Using {} GPUs'.format(torch.cuda.get_device_name(0)))
    return device


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
