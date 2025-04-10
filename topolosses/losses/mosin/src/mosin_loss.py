from __future__ import annotations
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torchvision

from ...utils import compute_default_dice_loss


class MosinLoss(_Loss):
    """TODO"""

    def __init__(
        self,
        include_background: bool = False,
        alpha: float = 0.5,
        softmax: bool = False,
        sigmoid: bool = False,
        use_base_loss: bool = True,
        base_loss: Optional[_Loss] = None,
    ) -> None:
        """TODO"""
        if sum([sigmoid, softmax]) > 1:
            raise ValueError(
                "At most one of [sigmoid, softmax] can be set to True. "
                "You can only choose one of these options at a time or none if you already pass probabilites."
            )

        super(MosinLoss, self).__init__()

        # TODO add parameters to allow the user to choose model
        # requires minimum image size of 32x32 pixels
        self.vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_layers = [2, 7, 16]
        self.layer_names = ["conv1_2", "conv2_2", "conv3_4"]
        self.activation = {}
        for i, name in zip(self.feature_layers, self.layer_names):
            self.vgg[i].register_forward_hook(self.get_activation(name))
        self.vgg.eval()
        self.vgg.requires_grad_(False)

        self.include_background = include_background
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

        mosin_loss = torch.tensor(0.0)
        if self.alpha > 0:
            self.vgg.to(input.device)
            mosin_loss = self.compute_mosin_loss(
                input[:, starting_class:].float(),
                target[:, starting_class:].float(),
            )
            # mosin_loss = torch.mean(torch.concatenate(mosin_loss))

        total_loss = mosin_loss if not self.use_base_loss else base_loss + self.alpha * mosin_loss

        return total_loss

    def compute_mosin_loss(self, prediction, target):

        # for multi class input
        prediction = torch.flatten(prediction, start_dim=0, end_dim=1).unsqueeze(1)
        target = torch.flatten(target, start_dim=0, end_dim=1).unsqueeze(1)

        prediction = prediction.expand(-1, 3, -1, -1)
        target = target.argmax(dim=1, keepdim=True).to(torch.float32).expand(-1, 3, -1, -1)

        pred_features = self.get_features(prediction)
        target_features = self.get_features(target)

        loss = 0
        for layer_name in self.layer_names:
            loss += F.mse_loss(pred_features[layer_name], target_features[layer_name])

        return loss

    def get_activation(self, name):
        """Hook to save activation for a given layer"""

        def hook(model, input, output):
            self.activation[name] = output

        return hook

    def get_features(self, x):
        """Extract features from specified VGG layers"""
        self.activation = {}  # Clear previous activations
        self.vgg(x)  # Forward pass through VGG
        return self.activation.copy()
