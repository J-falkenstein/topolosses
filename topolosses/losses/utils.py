import torch
from torch import Tensor

import enum
import numpy as np
from typing import List, Optional
import torch.nn.functional as F


def compute_default_dice_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_axis: Optional[List[int]] = None,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Function to compute the (weighted) Dice loss with default settings for the default base loss

    Args:
        input (torch.Tensor): The predicted segmentation map with shape (N, C, ...),
                            where N is batch size, C is the number of classes.
        target (torch.Tensor): The ground truth segmentation map with the same shape as `input`.
        reduce_axis (List[int]): The axes along which to reduce the loss computation.
                            To decide whether to sum the intersection and union areas over the batch dimension before the dividing.

    Returns:
        torch.Tensor: The Dice loss as a scalar

    """
    if reduce_axis == None:
        reduce_axis = list(range(2, len(input.shape)))

    intersection = torch.sum(target * input, dim=reduce_axis)
    ground_o = torch.sum(target, dim=reduce_axis)
    pred_o = torch.sum(input, dim=reduce_axis)
    denominator = ground_o + pred_o
    dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

    dice = torch.mean(dice)

    return dice


# TODO for all these enums and function check if they are used accross losses or only in topograph, move accordingly
class AggregationType(enum.Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    CE = "ce"
    RMS = "rms"
    LEG = "leg"


class ThresholdDistribution(enum.Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    NONE = "none"


def new_compute_diffs(paired_img_batch: torch.Tensor):
    h_diff = paired_img_batch[:, :-1, :] - paired_img_batch[:, 1:, :]
    v_diff = paired_img_batch[:, :, :-1] - paired_img_batch[:, :, 1:]
    h_diff = h_diff != 0
    v_diff = v_diff != 0
    return h_diff, v_diff


def new_compute_diag_diffs(paired_img_batch: torch.Tensor, th: int = 11):
    weight = torch.tensor([[1, -1], [-1, 1]], device=paired_img_batch.device).unsqueeze(0).unsqueeze(0)
    diag_connections = F.conv2d(paired_img_batch.unsqueeze(1).float(), weight.float()).squeeze(1)
    diagr = diag_connections > th
    diagl = diag_connections < -th
    special_case_r = torch.logical_or(diag_connections == 7, diag_connections == 4)
    special_case_l = torch.logical_or(diag_connections == -7, diag_connections == -4)
    # special_case_r = torch.zeros_like(diagr)
    # special_case_l = torch.zeros_like(diagl)

    return diagr, diagl, special_case_r, special_case_l


def fill_adj_matr(adj_matrix, h_edges, v_edges):
    adj_matrix[tuple(h_edges)] = True
    adj_matrix[tuple(h_edges[::-1])] = True  # Add the transposed edges
    adj_matrix[tuple(v_edges)] = True
    adj_matrix[tuple(v_edges[::-1])] = True  # Add the transposed edges

    # remove self loops
    np.fill_diagonal(adj_matrix, False)

    return adj_matrix


# TODO Check if only used in betti matching if so move accordingly
class FiltrationType(enum.Enum):
    SUPERLEVEL = "superlevel"
    SUBLEVEL = "sublevel"
    BOTHLEVELS = "bothlevels"
