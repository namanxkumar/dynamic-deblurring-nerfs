import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange


def camera_initialization_loss(
    predicted_camera_poses: torch.Tensor, ground_truth_camera_poses: torch.Tensor
):
    """
    Args:
        predicted_camera_poses: (B, 3, 3)
        ground_truth_camera_poses: (B, 3, 3)
    Returns:
        loss: (1)
    """
    return torch.mean((predicted_camera_poses - ground_truth_camera_poses) ** 2)


def blur_combination_loss(
    predicted_sharp_colors: torch.Tensor, ground_truth_blurred_colors: torch.Tensor
):
    """
    Args:
        predicted_sharp_colors: (B, R, 3)
        ground_truth_blurred_colors: (B, 3)
    Returns:
        loss: (1)
    """

    # Combine the predicted sharp images into a blurred image
    predicted_blurred_image = torch.mean(predicted_sharp_colors, dim=-2)

    # Compute the loss
    return torch.mean((predicted_blurred_image - ground_truth_blurred_colors) ** 2)


def sharp_image_initialization_loss(
    predicted_sharp_colors: torch.Tensor, ground_truth_sharp_colors: torch.Tensor
):
    """
    Args:
        predicted_sharp_colors: (B, R, 3)
        ground_truth_sharp_colors: (B, R, 3)
    Returns:
        loss: (1)
    """
    return torch.mean((predicted_sharp_colors - ground_truth_sharp_colors) ** 2)
