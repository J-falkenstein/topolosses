from __future__ import annotations
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

# TODO will this opencv library somehow intefere with the opencv c++ library?
# Could i use one installation for both use cases, the opencv-python did not work for c++ topograph
import cv2
from scipy import ndimage

from torch.nn.modules.loss import _Loss

from ...utils import compute_default_dice_loss


class WarpingLoss(_Loss):
    """A topology-aware loss function that emphasizes structurally critical pixels during segmentation.

    The loss has been defined in:
        Hu (2022) Structure-Aware Image Segmentation with Homotopy Warping (NeurIPS).

    This loss identifies topologically sensitive false positives and false negatives using distance transforms,
    then selectively applies a cross-entropy loss on these critical points to preserve object connectivity
    and structure. It is especially suited for applications requiring high topological fidelity.
    """

    def __init__(
        self,
        eight_connectivity: bool = True,
        alpha: float = 0.5,
        softmax: bool = False,
        sigmoid: bool = False,
        use_base_loss: bool = True,
        base_loss: Optional[_Loss] = None,
    ) -> None:
        """
        Args:
            eight_connectivity (bool): Determines whether to use 8-connectivity for foreground components (i.e., diagonal adjacent pixels form a single connected component)
                versus 4-connectivity when building the component graph. Defaults to 8-connectivity.
            alpha (float): Weighting factor for combining the base loss and the topology loss
                (i.e.: base_loss + alpha*topology_loss). Defaults to 0.5.
            sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the loss.
                Sigmoid is not applied before passing it to a custom base loss function. Defaults to `False`.
            softmax (bool): If `True`, applies a softmax activation to the input before computing the loss.
                Softmax is not applied before passing it to a custom base loss function. Defaults to `False`.
            use_base_loss (bool): If `False`, the loss only consists of the topology component.
                The base_loss and alpha will be ignored if this flag is set to false. Defaults to `True`.
            base_loss (_Loss, optional): The base loss function to be used alongside the topology loss.
                Defaults to `None`, meaning a standard cross-entropy loss will be used.

        Raises:
            ValueError: If more than one of [sigmoid, softmax] is set to True.
        """
        if sum([sigmoid, softmax]) > 1:
            raise ValueError(
                "At most one of [sigmoid, softmax] can be set to True. "
                "You can only choose one of these options at a time or none if you already pass probabilites."
            )

        super(WarpingLoss, self).__init__()
        if eight_connectivity:
            self.fg_connectivity = 8
            self.bg_connectivity = 4
        else:
            self.fg_connectivity = 4
            self.bg_connectivity = 8

        # is not used in Warp but will be part of the parent class
        self.include_background = True
        self.alpha = alpha
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.use_base_loss = use_base_loss
        self.base_loss = base_loss

        if not self.use_base_loss:
            if base_loss is not None:
                warnings.warn("base_loss is ignored beacuse use_base_component is set to false")
            if self.alpha != 1:
                warnings.warn("Alpha < 1 has no effect when no base component is used.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the forward pass of the Mosin Loss.

        Args:
            input (Tensor): Input tensor of shape (batch_size, num_classes, H, W).
            target (Tensor): Target tensor of shape (batch_size, num_classes, H, W).

        Returns:
            Tensor: The calculated betti matching loss.

        Raises:
            ValueError: If the shape of the ground truth is different from the input shape.
            ValueError: If the number of classe is smaller than 2.
        """
        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # will always be 0 but makes it comparable to other losses -> move to parent class
        starting_class = 0 if self.include_background else 1
        num_classes = input.shape[1]

        if num_classes == 1:
            raise ValueError(
                "Invalid input: Warp loss requires at least two class channels (e.g., foreground and background)."
                "Got only one channel."
            )

        # will never be reached but relevant for parent class later on
        if num_classes == 1:
            if self.softmax:
                raise ValueError(
                    "softmax=True requires multiple channels for class probabilities, but received a single-channel input."
                )
            if not self.include_background:
                warnings.warn(
                    "Single-channel prediction detected. The `include_background=False` setting  will be ignored."
                )
                starting_class = 0

        # Avoiding applying transformations sigmoid and softmax before passing the input to the base loss function
        # These settings have to be controlled by the user when initializing the base loss function
        base_loss = torch.tensor(0.0)
        if self.alpha < 1 and self.use_base_loss and self.base_loss is not None:
            base_loss = self.base_loss(input, target)

        if self.sigmoid:
            input = torch.sigmoid(input)
        elif self.softmax:
            input = torch.softmax(input, 1)

        if self.alpha < 1 and self.use_base_loss and self.base_loss is None:
            base_loss = compute_default_dice_loss(input, target)

        mosin_loss = torch.tensor(0.0)
        if self.alpha > 0:
            mosin_loss = self.compute_warping_loss(
                input[:, starting_class:].float(),
                target[:, starting_class:].float(),
            )

        total_loss = mosin_loss if not self.use_base_loss else base_loss + self.alpha * mosin_loss

        return total_loss

    def _decide_simple_point(self, target, x, y):
        """Flip the pixel at (x, y) if it’s a topologically ‘simple’ point in the 3×3 patch."""
        if x < 1 or y < 1 or x >= target.shape[0] - 1 or y >= target.shape[1] - 1:
            return target  # TODO: decide what to do
        patch = target[x - 1 : x + 2, y - 1 : y + 2]

        ccs_fg, _ = cv2.connectedComponents(patch, self.fg_connectivity)
        ccs_bg, _ = cv2.connectedComponents(patch, self.bg_connectivity)

        label = (ccs_fg - 1) * (ccs_bg - 1)
        if label == 1:
            target[x, y] = 1 - target[x, y]  # flip

        return target

    def _update_simple_point(self, distance, target):
        """Iterate over pixels by descending distance, flipping any simple points in the target."""
        non_zero_distance = np.nonzero(distance)
        idx = np.unravel_index(np.argsort(-distance, axis=None), distance.shape)

        for i in range(len(non_zero_distance[0])):
            x = idx[0][len(non_zero_distance[0]) - i - 1]
            y = idx[0][len(non_zero_distance[0]) - i - 1]

            target = self._decide_simple_point(target, x, y)

        return target

    def compute_warping_loss(self, input, target):
        """Compute cross-entropy loss only on pixels critical to preserving segmentation topology."""
        target = target.float()
        assert len(target.shape) == 4
        assert len(input.shape) == 4
        B, C, H, W = target.shape

        probs = F.softmax(input, dim=1)

        pred = torch.argmax(probs, dim=1)
        if C == 2:  # TODO: probably unnecessary
            target = torch.unsqueeze(target[:, 0, :, :], dim=1)

        predictions = pred.detach().cpu().numpy()
        predictions_c = predictions.copy()
        target_np = target.detach().cpu().numpy()
        target_c = target_np.copy()

        critical_points = np.zeros((B, H, W))
        for i in range(B):
            fp = ((predictions_c[i, :, :] - target_c[i, :, :]) == 1).astype(int)
            fn = ((target_c[i, :, :] - predictions_c[i, :, :]) == 1).astype(int)

            fn_distance_gt = ndimage.distance_transform_edt(target_c[i, :, :]) * fn
            fp_distance_gt = ndimage.distance_transform_edt(1 - target_c[i, :, :]) * fp

            target_warp = self._update_simple_point(fn_distance_gt, target_c[i, :, :])
            target_warp = self._update_simple_point(fp_distance_gt, target_warp)

            fn_distance_pre = (
                ndimage.distance_transform_edt(1 - predictions_c[i, :, :]) * fn
            )  # grow gt while keep unconnected
            fp_distance_pre = (
                ndimage.distance_transform_edt(predictions_c[i, :, :]) * fp
            )  # shrink pre while keep connected

            pre_warp = self._update_simple_point(fp_distance_pre, predictions_c[i, :, :])
            pre_warp = self._update_simple_point(fn_distance_pre, pre_warp)

            critical_points[i, :, :] = np.logical_or(
                np.not_equal(predictions[i, :, :], target_warp), np.not_equal(target_np[i, :, :], pre_warp)
            ).astype(int)

        critical_points = torch.from_numpy(critical_points).to(device=input.device)
        masked_input = input * torch.unsqueeze(critical_points, dim=1)
        masked_target = (target * torch.unsqueeze(critical_points, dim=1)).long()

        # TODO no include background at the moment and needs at least two class channels, might want add binary cross entropy?
        warping_loss = F.cross_entropy(masked_input, torch.squeeze(masked_target, dim=1)) * len(
            np.nonzero(critical_points)[0]
        )

        return warping_loss
