from typing import List, Optional
import torch
from .utils import convert_to_one_vs_rest
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import warnings

# if typing.TYPE_CHECKING:
from jaxtyping import Float


class DiceCLDiceLoss(_Loss):
    """A loss function for segmentation tasks that combines a Dice and CLDice componenent.

    The loss has been defined in:
        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

    This class is for a straightforward use with a (weighted) Dice loss as the base loss.
    For more advanced configurations of the base loss, use the `BaseCLDice` class, where you can provide an initialized base loss (e.g., cross-entropy).
    """

    def __init__(
        self,
        iter_: int = 3,
        alpha: float = 0.5,
        smooth: float = 1e-5,
        sigmoid: bool = False,
        softmax: bool = False,
        convert_to_one_vs_rest: bool = False,
        batch: bool = False,
        include_background: bool = False,
        weights: List[Float] = None,
    ):
        """Initializes the DiceCLDiceLoss object.

        Args:
            iter_ (int): Number of iterations for soft skeleton computation. Higher values refine
                the skeleton but increase computation time. Defaults to 3.
            alpha (float): Weighting factor for the CLDice and Dice components. Defaults to 0.5.
            smooth (float): Smoothing factor to avoid division by zero in Dice and CLDice calculations.
                Defaults to 1e-5.
            sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the loss.
                Defaults to `False`.
            softmax (bool): If `True`, applies a softmax activation to the input before computing the loss.
                Defaults to `False`.
            convert_to_one_vs_rest (bool): If `True`, converts the input into a one-vs-rest format for
                multi-class segmentation. Defaults to `False`.
            batch (bool): If `True`, reduces the loss across the batch dimension by summing intersection and union areas before division.
                Defaults to `False`, where the loss is computed independently for each item for the Dice and CLDice component calculation.
            include_background (bool): If `True`, includes the background class in CLDice component. Defaults to `False`.
                Background inclusion in the Dice component should be controlled using `weights` instead.
            weights (List[float], optional): Class-wise weights for the Dice component, allowing emphasis
                on specific classes or ignoring classes. Defaults to `None` (unweighted). Weights are **only
                applied to the Dice component**, not the CLDice component.

        Raises:
            ValueError: If more than one of `sigmoid`, `softmax`, or `convert_to_one_vs_rest` is set to `True`.

        """

        if sum([sigmoid, softmax, convert_to_one_vs_rest]) > 1:
            raise ValueError(
                "At most one of [sigmoid, softmax, convert_to_one_vs_rest] can be set to True. "
                "You can only choose one of these options at a time or none if you already pass probabilites."
            )

        super(DiceCLDiceLoss, self).__init__()

        self.iter_ = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.batch = batch
        self.include_background = include_background
        self.weights = None if weights == None else torch.tensor(weights)

    # TODO check that i do not change variables here that affect the next time forward is called
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the CLDice loss and Dice loss for the given input and target.

        Args:
            input (torch.Tensor): Predicted segmentation map of shape BC[spatial dimensions],
                where C is the number of classes, and [spatial dimensions] represent height, width, and optionally depth.
            target (torch.Tensor): Ground truth segmentation map of shape BC[spatial dimensions]

        Returns:
            tuple: A tuple containing the total DiceCLDice loss and a dictionary of individual loss components.

        Raises:
            ValueError: If the shape of the ground truth is different from the input shape.
            ValueError: If softmax=True and the number of channels for the prediction is 1.
            ValueError: If single channel prediction is used and include_background=False.

        """

        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        if input.shape[1] == 1 and not self.include_background:
            warnings.warn("Single-channel prediction detected. Automatically setting `include_background=True`.")
            self.include_background = True
        if self.softmax and input.shape[1] == 1:
            raise ValueError("softmax=True, but the number of channels for the prediction is 1.")
        if self.weights is not None and len(self.weights) != input.shape[1]:
            # Weights shape is independent of inlcude_background because they apply only to the Dice component, while `include_background` affects only CLDice calculations.
            raise ValueError(
                f"Wrong shape of weight vector: Number of class weights ({len(self.weights)}) must match the number of classes ({input.shape[1]})."
            )

        if self.weights is not None:
            self.weights = self.weights.to(input.device)

        if self.sigmoid:
            input = torch.sigmoid(input)
        elif self.softmax:
            input = torch.softmax(input, 1)
        elif self.convert_to_one_vs_rest:
            input = convert_to_one_vs_rest(input)

        reduce_axis: List[int] = [0] * self.batch + list(range(2, len(input.shape)))
        starting_class = 0 if self.include_background else 1

        cl_dice = torch.tensor(0.0)
        if self.alpha > 0:
            cl_dice = compute_cldice_loss(
                input[:, starting_class:].float(),
                target[:, starting_class:].float(),
                self.smooth,
                self.iter_,
                reduce_axis,
            )
        dice = torch.tensor(0.0)
        if self.alpha < 1:
            dice = self._compute_dice_loss(input, target, reduce_axis)

        dice_cl_dice_loss = (1 - self.alpha) * dice + self.alpha * cl_dice

        return dice_cl_dice_loss, {"dice": (1 - self.alpha) * dice, "cldice": self.alpha * cl_dice}

    # Could possibly be reused for other topo losses that are by defaul combined wiht dice loss
    def _compute_dice_loss(self, input: torch.Tensor, target: torch.Tensor, reduce_axis: List[int]) -> torch.Tensor:
        """Simplified function to compute the (weighted) Dice loss as part of the DiceCLDice loss.

        Args:
            input (torch.Tensor): The predicted segmentation map with shape (N, C, ...),
                                where N is batch size, C is the number of classes.
            target (torch.Tensor): The ground truth segmentation map with the same shape as `input`.
            reduce_axis (List[int]): The axes along which to reduce the loss computation.
                                To decide whether to sum the intersection and union areas over the batch dimension before the dividing.

        Returns:
            torch.Tensor: The Dice loss as a scalar

        """
        # TODO if weights = 0 at some channels, we can reduce the number of channels in the computation
        intersection = torch.sum(target * input, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # Weights are normalized to keep scales consistent
        # This is different to the monai implementation of weighted dice loss
        if self.weights is not None:
            weighted_dice = dice * (self.weights / self.weights.sum())
            dice = torch.mean(weighted_dice.sum(dim=1)) if not self.batch else weighted_dice.sum()
        else:
            dice = torch.mean(dice)

        return dice


class BaseCLDiceLoss(_Loss):
    """A loss function for segmentation that combines a base loss and a CLDice component.

    The loss has been defined in:
        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

    This class allows flexibility in choosing a custom base loss function.
    For a simpler implementation with a Dice base loss, use the `DiceCLDiceLoss` class.
    """

    def __init__(
        self,
        iter_: int = 3,
        alpha: float = 0.5,
        smooth: float = 1e-5,
        sigmoid: bool = False,
        softmax: bool = False,
        convert_to_one_vs_rest: bool = False,
        batch: bool = False,
        include_background: bool = False,
        base_loss: Optional[_Loss] = None,
    ):
        """
        Args:
            iter_ (int): Number of iterations for soft skeleton computation. Higher values refine
                the skeleton but increase computation time. Defaults to 3.
            smooth (float): Smoothing factor to avoid division by zero in CLDice calculations. Defaults to 1e-5.
            alpha (float): Weighting factor for combining the CLDice and base loss components. Defaults to 0.5.
            sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the CLDice loss.
                Defaults to `False`.
            softmax (bool): If `True`, applies a softmax activation to the input before computing the CLDice loss.
                Defaults to `False`.
            convert_to_one_vs_rest (bool): If `True`, converts the input into a one-vs-rest format for multi-class
                segmentation. Defaults to `False`.
            batch (bool): If `True`, reduces the loss across the batch dimension by summing intersection and union areas before division.
                Defaults to `False`, where the loss is computed independently for each item for the CLDice component calculation.
            include_background (bool): If `True`, includes the background class in CLDice computation. Defaults to `False`.
                Background inclusion in the Dice component should be controlled using `weights` instead.
            base_loss (_Loss, optional): The base loss function (e.g., cross-entropy) to be used alongside the CLDice loss.
                Defaults to `None`, meaning only the CLDice loss will be used.

        Raises:
            ValueError: If more than one of [sigmoid, softmax, convert_to_one_vs_rest] is set to True.
        """

        if sum([sigmoid, softmax, convert_to_one_vs_rest]) > 1:
            raise ValueError(
                "At most one of [sigmoid, softmax, convert_to_one_vs_rest] can be set to True. "
                "You can only choose one of these options at a time or none if you already pass probabilites."
            )

        super(BaseCLDiceLoss, self).__init__()

        self.iter_ = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.batch = batch
        self.include_background = include_background
        self.base_loss = base_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the CLDice loss and base loss for the given input and target.

        Args:
            input (torch.Tensor): The predicted segmentation map.
            target (torch.Tensor): The ground truth segmentation map.

        Returns:
            tuple: A tuple containing the total BaseCLDice loss and a dictionary of individual loss components.

        Raises:
            ValueError: If the shape of the ground truth is different from the input shape.
            ValueError: If softmax=True and the number of channels for the prediction is 1.
            ValueError: If single channel prediction is used and include_background=False.

        """

        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        if not self.include_background and input.shape[1] == 1:
            raise ValueError("single channel prediction, `include_background=False` is not a valid combination.")
        if self.softmax and input.shape[1] == 1:
            raise ValueError("softmax=True, but the number of channels for the prediction is 1.")

        # We avoid applying transformations like sigmoid, softmax, or one-vs-rest before passing the input to the base loss function
        # Such settings have to be controlled by the user when initializing the base loss function
        base_loss = torch.tensor(0.0)
        if self.base_loss is not None and self.alpha < 1:
            base_loss = self.base_loss(input, target)

        if self.sigmoid:
            input = torch.sigmoid(input)
        elif self.softmax:
            input = torch.softmax(input, 1)
        elif self.convert_to_one_vs_rest:
            input = convert_to_one_vs_rest(input)

        reduce_axis: List[int] = [0] * self.batch + list(range(2, len(input.shape)))
        starting_class = 0 if self.include_background else 1

        cl_dice = torch.tensor(0.0)
        if self.alpha > 0:
            cl_dice = compute_cldice_loss(
                input[:, starting_class:].float(),
                target[:, starting_class:].float(),
                self.smooth,
                self.iter_,
                reduce_axis,
            )

        base_cl_dice_loss = (1 - self.alpha) * base_loss + self.alpha * cl_dice

        return base_cl_dice_loss, {"base": (1 - self.alpha) * base_loss, "cldice": self.alpha * cl_dice}


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
                            To decide whether to sum the intersection and union areas over the batch dimension before the dividing.

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
