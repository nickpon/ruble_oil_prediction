import torch


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        print('Using cpu')
    else:
        print('Using {} GPUs'.format(torch.cuda.get_device_name(0)))
    return device
