from typing import List
import torch
from .utils import convert_to_one_vs_rest
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import warnings

# if typing.TYPE_CHECKING:
from jaxtyping import Float


# TODO Adjust to dice
class DiceLoss(_Loss):
    """
    DiceCLDice is a loss function for segmentation tasks that combines a Dice component and a CLDice componenent defined in:

        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)


    Attributes:
        iter_ (int): Number of iterations for soft skeleton computation. Higher values refine the skeleton but increase computation time. Default is 3.
        smooth (float): Smoothing factor to avoid division by zero in Dice and CLDice calculations. Default is 1e-5.
        alpha (float): Weighting factor for the CLDice loss. Setting `alpha=0` makes the loss equivalent to the standard Dice loss. Default is 0.5.
        sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the loss. Default is `False`.
        softmax (bool): If `True`, applies a softmax activation to the input before computing the loss. Default is `False`.
        convert_to_one_vs_rest (bool): If `True`, converts the input into a one-vs-rest format for multi-class segmentation. Default is `False`.
        batch (bool): If `True`, the loss is reduced across the batch dimension. Default is `False`.
        include_background (bool): If `True`, includes the background class in CLDice computation. Default is `False`.
                                   Note: Background inclusion in the Dice component should be controlled using `weights` instead.
        weights (List[float] or None): Class-wise weights for the Dice component, allowing emphasis on specific classes.
                                       Default is `None` (unweighted). Weights are **only applied to the Dice component**,
                                       not the CLDice component. This can be used to ignore the background in the Dice loss.

    Methods:
        forward(input, target): Computes the CLDice loss and Dice loss for the given input and target.

    """

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
        weights: List[Float] = None,
    ):
        """Initializes the DiceLoss object.

        Args:
            smooth (float): Smoothing factor to avoid division by zero added to numeraotr and denominator. Defaults to 1e-5.
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
            weights (List[float], optional): Class-wise weights with length equal to the number of classes, allowing emphasis on specific classes or ignoring classes.
                Defaults to `None` (unweighted).

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
        self.weights = None if weights == None else torch.tensor(weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the Dice loss.

        Args:
            input (torch.Tensor): Predicted segmentation map of shape BC[spatial dimensions],
                where C is the number of classes, and [spatial dimensions] represent height, width, and optionally depth.
            target (torch.Tensor): Ground truth segmentation map of shape BC[spatial dimensions]

        Returns:
            torch.Tensor: The Dice loss as a scalar value.

        Raises:
            ValueError: If the shape of the ground truth is different from the input shape.
            ValueError: If softmax=True and the number of channels for the prediction is 1.
        """
        if target.shape != input.shape:
            raise ValueError(f"Ground truth has different shape ({target.shape}) from input ({input.shape})")
        if input.shape[1] == 1 and not self.include_background:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
            self.include_background = True
        if self.softmax and input.shape[1] == 1:
            raise ValueError("softmax=True, but the number of channels for the prediction is 1.")
        if self.weights is not None and len(self.weights) != (input.shape[1] - (0 if self.include_background else 1)):
            raise ValueError(
                f"Wrong shape of weight vector: Number of class weights ({len(self.weights)}) must match the number of classes."
                f"({'including' if self.include_background else 'excluding'} background) ({input.shape[1]})."
            )

        if self.weights is not None:
            self.weights = self.weights.to(input.device)
            non_zero_weights_mask = self.weights != 0
            input = input[:, non_zero_weights_mask]
            target = target[:, non_zero_weights_mask]

        if not self.include_background:
            input = input[:, 1:]
            target = target[:, 1:]

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
