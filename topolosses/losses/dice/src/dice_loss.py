import warnings
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from ...utils import convert_to_one_vs_rest


class DiceLoss(_Loss):
    """Computes the Dice loss between two tensors."""

    def __init__(
        self,
        iter_=3,
        alpha=0.5,
        smooth=1e-5,
        sigmoid=False,
        softmax=False,
        convert_to_one_vs_rest=False,
        batch=False,
        include_background=True,
        weights: Optional[Tensor] = None,
    ):
        """Initializes the DiceLoss object.

        Args:
            smooth (float): Smoothing factor to avoid division by zero added to numerator and denominator. Defaults to 1e-5.
            sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the loss.
                Defaults to `False`.
            softmax (bool): If `True`, applies a softmax activation to the input before computing the loss.
                Defaults to `False`.
            convert_to_one_vs_rest (bool): If `True`, converts the input into a one-vs-rest format for
                multi-class segmentation. Defaults to `False`.
            batch (bool): If `True`, reduces the loss across the batch dimension by summing intersection and union areas before division.
                Defaults to `False`, where the loss is computed independently for each item for the Dice calculation and reduced afterwards.
            include_background (bool): If `False`, channel index 0 (background class) is excluded from the calculation.
                Defaults to `False`.
            weights (Tensor, optional): A 1D tensor of class-wise weights, with length equal to the number of classes (adjusted for background inclusion).
                It allows emphasizing or ignoring classes. Defaults to `None` (unweighted).


        Raises:
            ValueError: If more than one of `sigmoid`, `softmax`, or `convert_to_one_vs_rest` is set to `True`.

        """

        if sum([sigmoid, softmax, convert_to_one_vs_rest]) > 1:
            raise ValueError(
                "At most one of [sigmoid, softmax, convert_to_one_vs_rest] can be set to True. "
                "You can only choose one of these options at a time or none if you already pass probabilites."
            )

        super(DiceLoss, self).__init__()

        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.batch = batch
        self.include_background = include_background
        self.register_buffer("weights", weights)
        self.weights: Optional[Tensor]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the Dice loss between two tensors.

        Args:
            input (torch.Tensor): Predicted segmentation map of shape BC[spatial dimensions],
                where C is the number of classes, and [spatial dimensions] represent height, width, and optionally depth.
            target (torch.Tensor): Ground truth segmentation map of shape BC[spatial dimensions]

        Returns:
            torch.Tensor: The Dice loss as a scalar.

        Raises:
            ValueError: If the shape of the ground truth is different from the input shape.
            ValueError: If softmax=True and the number of channels for the prediction is 1.
        """
        if target.shape != input.shape:
            raise ValueError(f"Ground truth has different shape ({target.shape}) from input ({input.shape})")
        if self.weights is not None:
            if len(self.weights.shape) != 1:
                raise ValueError("weights must be a 1-dimensional tensor (vector).")
            if len(self.weights) != (input.shape[1] - (0 if self.include_background or input.shape[1] == 1 else 1)):
                raise ValueError(
                    f"Wrong shape of weight vector: Number of class weights ({len(self.weights)}) must match the number of classes."
                    f"({'including' if self.include_background else 'excluding'} background) ({input.shape[1]})."
                )
            non_zero_weights_mask = self.weights != 0
            input = input[:, non_zero_weights_mask]
            target = target[:, non_zero_weights_mask]

        starting_class = 0 if self.include_background else 1

        if input.shape[1] == 1:
            if self.softmax:
                raise ValueError(
                    "softmax=True requires multiple channels for class probabilities, but received a single-channel input."
                )
            if not self.include_background:
                warnings.warn(
                    "Single-channel prediction detected. The `include_background=False` setting  will be ignored."
                )
                starting_class = 0

        input = input[:, starting_class:]
        target = target[:, starting_class:]

        if self.sigmoid:
            input = torch.sigmoid(input)
        elif self.softmax:
            input = torch.softmax(input, 1)
        elif self.convert_to_one_vs_rest:
            input = convert_to_one_vs_rest(input)

        reduce_axis: List[int] = [0] * self.batch + list(range(2, len(input.shape)))

        intersection = torch.sum(target * input, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # Weights are normalized to keep scales consistent
        # This is different to the monai implementation of weighted dice loss
        if self.weights is not None:
            weighted_dice = dice * (self.weights[non_zero_weights_mask] / self.weights[non_zero_weights_mask].sum())
            dice = torch.mean(weighted_dice.sum(dim=1)) if not self.batch else weighted_dice.sum()
        else:
            dice = torch.mean(dice)

        return dice
