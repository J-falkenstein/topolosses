import torch


# this is more of a one-vs-max strategy
def convert_to_one_vs_rest(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Converts a multi-class prediction tensor into a one-vs-rest format by building
    the softmax over each class (one) and the max of all other classes (rest).

    Args:
        prediction (torch.Tensor): The input prediction tensor of shape (batch, channel, *spatial_dimensions).

    Returns:
        torch.Tensor: The converted prediction tensor of shape (batch, channel, *spatial_dimensions).
    """
    converted_prediction = torch.zeros_like(prediction)

    for channel in range(prediction.shape[1]):
        # Get logits for the channel class
        channel_logits = prediction[:, channel].unsqueeze(1)

        # For each pixel, get the class with the highest probability but exclude the channel class
        rest_logits = torch.max(prediction[:, torch.arange(prediction.shape[1]) != channel], dim=1).values.unsqueeze(1)

        # Apply softmax to get probabilities and select the probability of the channel class
        converted_prediction[:, channel] = torch.softmax(torch.cat([rest_logits, channel_logits], dim=1), dim=1)[:, 1]

    return converted_prediction
