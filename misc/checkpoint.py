import os
import torch


def save_checkpoint(path, epoch, model, optimizer, params=None):
    """
    Save a PyTorch checkpoint.

    Args:
        path (str): path where the checkpoint will be saved.
        epoch (int): current epoch.
        model (torch.nn.Module): model whose parameters will be saved.
        optimizer (torch.optim.Optimizer): optimizer whose parameters will be saved.
        params (dict): other parameters. Optional.
            Default: None

    """
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint.pth')
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        },
        path
    )


def load_checkpoint(path, model, optimizer=None, device=None):
    """
    Load a PyTorch checkpoint.

    Args:
        path (str): checkpoint path.
        model (torch.nn.Module): model whose parameters will be loaded.
        optimizer (torch.optim.Optimizer): optimizer whose parameters will be loaded. Optional.
            Default: None
        device (torch.device): device to be used.
            Default: None

    Returns:
        epoch (int): saved epoch
        model (torch.nn.Module): reference to `model`
        optimizer (torch.nn.Optimizer): reference to `optimizer`
        params (dict): other saved params

    """
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint.pth')
    checkpoint = torch.load(path, map_location=device)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    params = checkpoint['params']

    return epoch, model, optimizer, params
