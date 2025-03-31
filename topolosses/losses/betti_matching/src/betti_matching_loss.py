from __future__ import annotations
import warnings
from typing import List, Optional

import enum
import torch
from torch.nn.modules.loss import _Loss
from functools import partial
import numpy as np

# from monai.losses.dice import DiceLoss
# import sys, os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, "ext/Betti-Matching-3D/build"))
# sys.path.append(os.path.join(current_dir, "ext/Betti-Matching-3D-standalone-branch/build"))
# import betti_matching  # C++ Implementation
from . import betti_matching

# from losses.dice_losses import Multiclass_CLDice
from ...utils import compute_default_dice_loss
from ...utils import FiltrationType


class BettiMatchingLoss(_Loss):
    """TODO"""

    def __init__(
        self,
        filtration_type: FiltrationType = FiltrationType.SUPERLEVEL,
        num_processes: int = 1,
        push_unmatched_to_1_0: bool = False,
        barcode_length_threshold: float = 0.0,
        topology_weights: tuple[float, float] = (
            1.0,
            1.0,
        ),
        sphere: bool = False,
        include_background: bool = False,
        alpha: float = 0.5,
        softmax: bool = False,
        sigmoid: bool = False,
        use_base_loss: bool = True,
        base_loss: Optional[_Loss] = None,
    ) -> None:
        """
        Args:
            TODO
            topology_weight: TODO weights for the topology classes in the following order: [matched, unmatched_pred, unmatched_target]
            include_background (bool): If `True`, includes the background class in the topograph computation.
                Background inclusion in the base loss component should be controlled independently.
            alpha (float): Weighting factor for the topograph loss component. Is only applied if a base loss is used. Defaults to 0.1.
            sigmoid (bool): If `True`, applies a sigmoid activation to the input before computing the CLDice loss.
                Typically used for binary segmentation. Defaults to `False`.
            softmax (bool): If `True`, applies a softmax activation to the input before computing the CLDice loss.
                This is useful for multi-class segmentation tasks. Defaults to `False`.
                For other activation functions set sigmoid and softmax to false and apply the transformation before passing inputs to the loss.
            use_base_component (bool): if false the loss only consists of the Topograph component.
                A forward call will return the full Topograph component.
                base_loss, weights, and alpha will be ignored if this flag is set to false.
            base_loss (_Loss, optional): The base loss function to be used alongside the Topograph loss.
                Defaults to `None`, meaning a Dice component with default parameters will be used.  .

        Raises:
            ValueError: If more than one of [sigmoid, softmax] is set to True.
            ValueError: If topology_weights is not a list of lenght 2
        """
        if sum([sigmoid, softmax]) > 1:
            raise ValueError(
                "At most one of [sigmoid, softmax] can be set to True. "
                "You can only choose one of these options at a time or none if you already pass probabilites."
            )
        if len(topology_weights) != 2:
            raise ValueError(
                "Topology weights must be a list of length 2, where the first element is the weight for matched pairs and the second for unmatched pairs in the prediction."
            )

        super(BettiMatchingLoss, self).__init__()

        self.filtration_type = filtration_type
        self.num_processes = num_processes
        self.push_unmatched_to_1_0 = push_unmatched_to_1_0
        self.barcode_length_threshold = barcode_length_threshold
        self.include_background = include_background
        self.topology_weights = topology_weights
        self.sphere = sphere
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
        """Calculates the forward pass of the betti matching loss.

        Args:
            input (Tensor): Input tensor of shape (batch_size, num_classes, H, W).
            target (Tensor): Target tensor of shape (batch_size, num_classes, H, W).

        Returns:
            Tensor: The calculated betti matching loss.

        Raises:
            ValueError: If the shape of the ground truth is different from the input shape.
            ValueError: If softmax=True and the number of channels for the prediction is 1.

        """
        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        starting_class = 0 if self.include_background else 1
        num_classes = input.shape[1]

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

        # Avoiding applying transformations like sigmoid, softmax, or one-vs-rest before passing the input to the base loss function
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

        betti_matching_loss = torch.tensor(0.0)
        if self.alpha > 0:
            betti_matching_loss, losses = self._compute_batched_betti_matching_loss(
                input[:, starting_class:].float(),
                target[:, starting_class:].float(),
            )

        total_loss = betti_matching_loss if not self.use_base_loss else base_loss + self.alpha * betti_matching_loss

        return total_loss

    def _compute_batched_betti_matching_loss(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """TODO"""
        # Flatten out channel dimension to treat each channel as a separate instance for multiclass prediction
        input = torch.flatten(input, start_dim=0, end_dim=1).unsqueeze(1)
        target = torch.flatten(target, start_dim=0, end_dim=1).unsqueeze(1)
        if self.sphere:
            # add padding of 1 to all sides of the spatial dimensions with correct predicted background
            input = torch.nn.functional.pad(input, (1, 1, 1, 1), mode="constant", value=0)
            target = torch.nn.functional.pad(target, (1, 1, 1, 1), mode="constant", value=0)
        if self.filtration_type == FiltrationType.SUPERLEVEL:
            # Using (1 - ...) to allow binary sorting optimization on the label, which expects values [0, 1]
            input = 1 - input
            target = 1 - target
        if self.filtration_type == FiltrationType.BOTHLEVELS:
            # Just duplicate the number of elements in the batch, once with sublevel, once with superlevel
            input = torch.concat([input, 1 - input])
            target = torch.concat([target, 1 - target])

        split_indices = np.arange(self.num_processes, input.shape[0], self.num_processes)
        predictions_list_numpy = np.split(input.detach().cpu().numpy().astype(np.float64), split_indices)
        targets_list_numpy = np.split(target.detach().cpu().numpy().astype(np.float64), split_indices)

        num_dimensions = input.ndim - 2
        num_matched_by_dim = torch.zeros((num_dimensions,), device=input.device)
        num_unmatched_prediction_by_dim = torch.zeros((num_dimensions,), device=input.device)

        losses = []

        current_instance_index = 0
        for predictions_cpu_batch, targets_cpu_batch in zip(predictions_list_numpy, targets_list_numpy):
            predictions_cpu_batch, targets_cpu_batch = list(predictions_cpu_batch.squeeze(1)), list(
                targets_cpu_batch.squeeze(1)
            )
            if not (
                all(a.data.contiguous for a in predictions_cpu_batch)
                and all(a.data.contiguous for a in targets_cpu_batch)
            ):
                warnings.warn(
                    f"WARNING! Non-contiguous arrays encountered. Shape: {predictions_cpu_batch[0].shape}",
                    RuntimeWarning,
                )
                global ENCOUNTERED_NONCONTIGUOUS
                ENCOUNTERED_NONCONTIGUOUS = True
            predictions_cpu_batch = [np.ascontiguousarray(a) for a in predictions_cpu_batch]
            targets_cpu_batch = [np.ascontiguousarray(a) for a in targets_cpu_batch]

            results = betti_matching.compute_matching(predictions_cpu_batch, targets_cpu_batch)

            for result_arrays in results:
                losses.append(
                    self._betti_matching_loss(
                        input[current_instance_index].squeeze(0),
                        target[current_instance_index].squeeze(0),
                        result_arrays,
                    )
                )

                num_matched_by_dim += torch.tensor(result_arrays.num_matched, device=input.device, dtype=torch.long)
                num_unmatched_prediction_by_dim += torch.tensor(
                    result_arrays.num_unmatched_input1, device=input.device, dtype=torch.long
                )

                current_instance_index += 1

        return torch.mean(torch.concatenate(losses)), losses

    def _betti_matching_loss(
        self,
        prediction: torch.Tensor,  # *spatial_dimensions
        target: torch.Tensor,  # *spatial_dimensions
        betti_matching_result: betti_matching.return_types.BettiMatchingResult,
    ) -> torch.Tensor:  # one_dimension
        """TODO"""
        # Combine all birth and death coordinates from prediction and target into one array
        losses_by_dim = torch.zeros(
            (len(betti_matching_result.input1_matched_birth_coordinates),),
            device=prediction.device,
            dtype=torch.float32,
        )
        dims = len(betti_matching_result.input1_matched_birth_coordinates)
        # Iterate over all dimensions
        for dim in range(dims):
            # Combine all birth and death coordinates from prediction and target into one array
            (
                prediction_matches_birth_coordinates,
                prediction_matches_death_coordinates,
                target_matches_birth_coordinates,
                target_matches_death_coordinates,
                prediction_unmatched_birth_coordinates,
                prediction_unmatched_death_coordinates,
            ) = [
                (
                    torch.tensor(array, device=prediction.device, dtype=torch.long)
                    if array.strides[-1] > 0
                    else torch.zeros(0, len(prediction.shape), device=prediction.device, dtype=torch.long)
                )
                for array in [
                    betti_matching_result.input1_matched_birth_coordinates[dim],
                    betti_matching_result.input1_matched_death_coordinates[dim],
                    betti_matching_result.input2_matched_birth_coordinates[dim],
                    betti_matching_result.input2_matched_death_coordinates[dim],
                    betti_matching_result.input1_unmatched_birth_coordinates[dim],
                    betti_matching_result.input1_unmatched_death_coordinates[dim],
                ]
            ]

            # Get the Barcode interval of the matched pairs from the prediction using the coordinates
            # (M, 2) tensor of matched persistence pairs for prediction
            prediction_matched_pairs = torch.stack(
                [
                    prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
                    for coords in [prediction_matches_birth_coordinates, prediction_matches_death_coordinates]
                ],
                dim=1,
            )

            # Get the Barcode interval of the matched pairs from the target using the coordinates
            # (M, 2) tensor of matched persistence pairs for target
            target_matched_pairs = torch.stack(
                [
                    target[tuple(coords[:, i] for i in range(coords.shape[1]))]
                    for coords in [target_matches_birth_coordinates, target_matches_death_coordinates]
                ],
                dim=1,
            )

            # Get the Barcode interval of all unmatched pairs  in the prediction using the coordinates
            # (M, 2) tensor of unmachted persistence pairs for prediction
            prediction_unmatched_pairs = torch.stack(
                [
                    prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
                    for coords in [prediction_unmatched_birth_coordinates, prediction_unmatched_death_coordinates]
                ],
                dim=1,
            )

            # Get the Barcode interval of all unmatched pairs in the target using the coordinates
            # (M, 2) tensor of unmatched persistence pairs for target
            target_unmatched_pairs = torch.stack(
                [
                    target[tuple(coords[:, i] for i in range(coords.shape[1]))]
                    for coords in [
                        betti_matching_result.input2_unmatched_birth_coordinates[dim],
                        betti_matching_result.input2_unmatched_death_coordinates[dim],
                    ]
                ],
                dim=1,
            )

            # filter all pairs where abs(birth - death) < threshold
            prediction_unmatched_pairs = prediction_unmatched_pairs[
                torch.abs(prediction_unmatched_pairs[:, 0] - prediction_unmatched_pairs[:, 1])
                > self.barcode_length_threshold
            ]

            # sum over ||(birth_pred_i, death_pred_i), (birth_target_i, death_target_i)||²
            loss_matched = (
                2 * ((prediction_matched_pairs - target_matched_pairs) ** 2).sum() * self.topology_weights[0]
            )

            # sum over ||(birth_pred_i, death_pred_i), 1/2*(birth_pred_i+death_pred_i, birth_pred_i+death_pred_i)||²
            # reformulated as (birth_pred_i^2 / 4 + death_pred_i^2/4 - birth_pred_i*death_pred_i/2)
            if self.push_unmatched_to_1_0:
                loss_unmatched_pred = (
                    2
                    * ((prediction_unmatched_pairs[:, 0] - 1) ** 2 + prediction_unmatched_pairs[:, 1] ** 2).sum()
                    * self.topology_weights[1]
                )
                loss_unmatched_target = (
                    2 * ((target_unmatched_pairs[:, 0] - 1) ** 2 + target_unmatched_pairs[:, 1] ** 2).sum()
                )
            else:
                loss_unmatched_pred = (
                    (prediction_unmatched_pairs[:, 0] - prediction_unmatched_pairs[:, 1]) ** 2
                ).sum() * self.topology_weights[1]
                loss_unmatched_target = ((target_unmatched_pairs[:, 0] - target_unmatched_pairs[:, 1]) ** 2).sum()

            losses_by_dim[dim] = loss_matched + loss_unmatched_pred + loss_unmatched_target

        return torch.sum(losses_by_dim).reshape(1)
