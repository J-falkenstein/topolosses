from typing import List
import torch
from .utils import convert_to_one_vs_rest
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

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
        include_background=False,
        weights: List[Float] = None,
    ):
        """
        Args:
            iter_ (int): Number of iterations for soft skeleton computation. Higher values refine the skeleton but increase computation time. Default is 3.
            smooth (float): Smoothing factor to avoid division by zero in Dice and CLDice calculations. Default is 1e-5.
            alpha (float): Weighting factor for the CLDice and the Dice components. Default is 0.5.
            sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the loss. Default is `False`.
            softmax (bool): If `True`, applies a softmax activation to the input before computing the loss. Default is `False`.
            convert_to_one_vs_rest (bool): If `True`, converts the input into a one-vs-rest format for multi-class segmentation. Default is `False`.
            batch (bool): If `True`, the loss is reduced across the batch dimension. Default is `False`.
            include_background (bool): If `True`, includes the background class in CLDice computation. Default is `False`.
                                    Note: Background inclusion in the Dice component should be controlled using `weights` instead.
            weights (List[float] or None): Class-wise weights for the Dice component, allowing emphasis on specific classes or ignoring classes.
                                       Default is `None` (unweighted). Weights are **only applied to the Dice component**,
                                       not the CLDice component. This can be used to ignore the background in the Dice loss.

        Raises:
            ValueError: If more than one of [sigmoid, softmax, convert_to_one_vs_rest] is set to True.
            ValueError: If the length of the weight vector does not match the number of classes in the input.

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

    # TODO integrate to the class
    def _compute_dice_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice loss. If weights is not None, the loss is weighted.

        Args:
            input (torch.Tensor): The predicted segmentation map with shape (N, C, ...),
                                where N is batch size, C is the number of classes.
            target (torch.Tensor): The ground truth segmentation map with the same shape as `input`.
            reduce_axis (List[int]): The axes along which to reduce the loss computation.

        Returns:
            torch.Tensor: The Dice loss as a scalar

        """
        reduce_axis: List[int] = [0] * self.batch + list(range(2, len(input.shape)))

        intersection = torch.sum(target * input, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if self.weights is not None:
            dice = (dice * (self.weights / self.weights.sum())).sum()
        else:
            dice = torch.mean(dice)
        return dice
