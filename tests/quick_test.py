import torch


def dice_new(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-5):
    intersection = torch.sum(target * input, dim=reduce_axis)
    ground_o = torch.sum(target, dim=reduce_axis)
    pred_o = torch.sum(input, dim=reduce_axis)
    denominator = ground_o + pred_o

    dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    return dice


def soft_dice(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Function to compute soft dice loss

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py#L22

    Args:
        y_true: the shape should be BCH(WD)
        y_pred: the shape should be BCH(WD)

    Returns:
        dice loss
    """
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth)
    soft_dice: torch.Tensor = 1.0 - coeff
    return soft_dice
