import warnings
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from ...utils import compute_default_dice_loss


class CLDiceLoss(_Loss):
    """A loss function for segmentation that combines a base loss and a CLDice component.

    The loss has been defined in:
        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

    By default the cl dice component is combined with a (weighted) default dice loss.
    For more flexibility a custom base loss function can be passed.
    """

    def __init__(
        self,
        iter_: int = 3,
        smooth: float = 1e-5,
        batch: bool = False,
        include_background: bool = False,
        alpha: float = 0.5,
        sigmoid: bool = False,
        softmax: bool = False,
        use_base_loss: bool = True,
        base_loss: Optional[_Loss] = None,
    ) -> None:
        """
        Args:
            iter_ (int): Number of iterations for soft skeleton computation. Higher values refine
                the skeleton but increase computation time. Defaults to 3.
            smooth (float): Smoothing factor to avoid division by zero in CLDice and the default base dice calculations. Defaults to 1e-5.
            batch (bool): If `True`, reduces the loss across the batch dimension by summing intersection and union areas before division.
                Defaults to `False`, where the loss is computed independently for each item for the CLDice and default base dice component calculation.
            include_background (bool): If `True`, includes the background class in CLDice computation. Defaults to `False`.
            alpha (float): Weighting factor for combining the CLDice component (i.e.: base_loss + alpha*cldice_loss).
                Defaults to 0.5.
            sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the CLDice and the default dice component.
                Sigmoid is not applied before passing it to a custom base loss function. Defaults to `False`.
            softmax (bool): If `True`, applies a softmax activation to the input before computing the CLDice loss.
                Softmax is not applied before passing it to a custom base loss function. Defaults to `False`.
            use_base_component (bool): if false the loss only consists of the CLDice component. A forward call will return the full CLDice component.
                base_loss and alpha will be ignored if this flag is set to false.
            base_loss (_Loss, optional): The base loss function to be used alongside the CLDice loss.
                Defaults to `None`, meaning a Dice component with default parameters will be used.

        Raises:
            ValueError: If more than one of [sigmoid, softmax] is set to True.
        """

        if sum([sigmoid, softmax]) > 1:
            raise ValueError(
                "At most one of [sigmoid, softmax, convert_to_one_vs_rest] can be set to True. "
                "You can only choose one of these options at a time or none if you already pass probabilites."
            )

        super(CLDiceLoss, self).__init__()

        self.iter_ = iter_
        self.smooth = smooth
        self.batch = batch
        self.include_background = include_background
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.use_base_component = use_base_loss
        self.base_loss = base_loss

        if not self.use_base_component:
            if base_loss is not None:
                warnings.warn("base_loss is ignored beacuse use_base_component is set to false")
            if self.alpha != 1:
                warnings.warn(
                    "Alpha < 1 has no effect when no base component is used. The full ClDice loss will be returned."
                )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the CLDice loss and base loss for the given input and target.

        Args:
            input (torch.Tensor): Predicted segmentation map of shape BC[spatial dimensions],
                where C is the number of classes, and [spatial dimensions] represent height, width, and optionally depth.
            target (torch.Tensor): Ground truth segmentation map of shape BC[spatial dimensions]

        Returns:
            tuple: A tuple containing the total DiceCLDice loss and a dictionary of individual loss components. # TODO define the return type

        Raises:
            ValueError: If the shape of the ground truth is different from the input shape.
            ValueError: If softmax=True and the number of channels for the prediction is 1.

        """

        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        if len(input.shape) < 4:
            raise ValueError(
                "Invalid input tensor shape. Expected at least 4 dimensions in the format (batch, channel, [spatial dims]), "
                "where 'spatial dims' must be at least 2D (height, width). "
                f"Received shape: {input.shape}."
            )

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

        # Avoiding applying transformations like sigmoid, softmax, or one-vs-rest before passing the input to the base loss function
        # These settings have to be controlled by the user when initializing the base loss function
        base_loss = torch.tensor(0.0)
        if self.alpha < 1 and self.use_base_component and self.base_loss is not None:
            base_loss = self.base_loss(input, target)

        if self.sigmoid:
            input = torch.sigmoid(input)
        elif self.softmax:
            input = torch.softmax(input, 1)

        reduce_axis: List[int] = [0] * self.batch + list(range(2, len(input.shape)))

        if self.alpha < 1 and self.use_base_component and self.base_loss is None:
            base_loss = compute_default_dice_loss(
                input,
                target,
                reduce_axis,
                self.smooth,
            )

        cl_dice = torch.tensor(0.0)
        if self.alpha > 0:
            cl_dice = compute_cldice_loss(
                input[:, starting_class:].float(),
                target[:, starting_class:].float(),
                self.smooth,
                self.iter_,
                reduce_axis,
            )

        # TODO do not multiple base loss by 1-alpha, this has an effect on the current test cases
        base_cl_dice_loss = (
            cl_dice if not self.use_base_component else (1 - self.alpha) * base_loss + self.alpha * cl_dice
        )

        return base_cl_dice_loss  # , {"base": (1 - self.alpha) * base_loss, "cldice": self.alpha * cl_dice}


# TODO think about putting in class to not pass all the arguments (bc in topograph it makes sense there might be too many arguments to pass for it to be beautiful)
def compute_cldice_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    smooth: float,
    iter_: int,
    reduce_axis: List[int],
) -> torch.Tensor:
    """Computes the CLDice loss.

    Args:
        input (torch.Tensor): The predicted segmentation map with shape (N, C, ...),
                            where N is batch size, C is the number of classes.
        target (torch.Tensor): The ground truth segmentation map with the same shape as `input`.
        smooth (float): Smoothing factor to avoid division by zero.
        iter_ (int): Number of iterations for soft skeleton computation.
        reduce_axis (List[int]): The axes along which to reduce the loss computation.
                            It decides whether to sum the intersection and union areas over the batch dimension before the dividing.

    Returns:
        torch.Tensor: The CLDice loss as a scalar tensor.
    """

    pred_skeletons = soft_skel(input, iter_)
    target_skeletons = soft_skel(target, iter_)

    tprec = (
        torch.sum(
            torch.multiply(pred_skeletons, target),
            dim=reduce_axis,
        )
        + smooth
    ) / (torch.sum(pred_skeletons, dim=reduce_axis) + smooth)

    tsens = (
        torch.sum(
            torch.multiply(target_skeletons, input),
            dim=reduce_axis,
        )
        + smooth
    ) / (torch.sum(target_skeletons, dim=reduce_axis) + smooth)

    return torch.mean(1.0 - 2.0 * (tprec * tsens) / (tprec + tsens))


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Erode the input image by shrinking objects using max pooling"""
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    else:
        raise ValueError("input tensor must have 4D with shape: (batch, channel, height, width)")


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Perform soft dilation on the input image using max pooling."""
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    else:
        raise ValueError("input tensor must have 4D with shape: (batch, channel, height, width)")


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Apply opening: erosion followed by dilation."""
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int) -> torch.Tensor:
    """Generate a soft skeleton by iteratively applying erosion and opening."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel
