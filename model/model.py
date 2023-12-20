import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from .render import Render


class Model(nn.Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        num_time_steps_per_frame: int,
        num_frames: int,
        num_coarse_samples_per_ray: int,
        num_fine_samples_per_ray: int,
        near_depth: float,
        far_depth: float,
        num_scene_trajectory_basis_coefficients: int,
        num_camera_trajectory_basis_coefficients: int,
        num_voxels_per_axis: int,
        min_bound_per_axis: float,
        max_bound_per_axis: float,
        voxel_dim: int,
        color_model_hidden_dim: int,
        device: torch.device,
        focal_length: float = 0.005,
    ):
        super().__init__()
        self.device = device
        self.num_time_steps_per_frame = num_time_steps_per_frame
        self.total_num_time_steps = num_time_steps_per_frame * num_frames
        self.camera_intrinsics = torch.tensor(
            [
                [focal_length, 0, image_width / 2],
                [0, focal_length, image_height / 2],
                [0, 0, 1],
            ]
        )
        self.model = Render(
            camera_intrinsics=self.camera_intrinsics,
            num_time_steps=self.total_num_time_steps,
            num_coarse_samples_per_ray=num_coarse_samples_per_ray,
            num_fine_samples_per_ray=num_fine_samples_per_ray,
            near_depth=near_depth,
            far_depth=far_depth,
            num_scene_trajectory_basis_coefficients=num_scene_trajectory_basis_coefficients,
            num_camera_trajectory_basis_coefficients=num_camera_trajectory_basis_coefficients,
            num_voxels_per_axis=num_voxels_per_axis,
            min_bound_per_axis=min_bound_per_axis,
            max_bound_per_axis=max_bound_per_axis,
            voxel_dim=voxel_dim,
            color_model_hidden_dim=color_model_hidden_dim,
            device=self.device,
        )

    def forward(
        self,
        time_step: torch.Tensor,
        warmup: bool = False,
    ):
        time_step_batch = torch.ones(
            time_step.shape[0], self.num_time_steps_per_frame
        ).to(self.device)
        time_step_batch = (
            torch.arange(-(self.num_time_steps_per_frame//2), (self.num_time_steps_per_frame//2) + 1) * time_step_batch
        )  # Shape: (batch_size, num_time_steps_per_frame)
        time_step_batch = (
            time_step_batch + time_step[:, None]
        )  # Shape: (batch_size, num_time_steps_per_frame)
        time_step_batch = rearrange(time_step_batch, "b t -> (b t)")
        
        images = self.model(time_step_batch, warmup=warmup)
        images = rearrange(
            images,
            "(b t) c h w -> b t c h w",
            b=time_step.shape[0],
        )
        return images


def initialize_model(
    focal_length: float,
    image_width: int,
    image_height: int,
    num_time_steps_per_frame: int,
    num_frames: int,
    num_coarse_samples_per_ray: int,
    num_fine_samples_per_ray: int,
    near_depth: float,
    far_depth: float,
    num_scene_trajectory_basis_coefficients: int,
    num_camera_trajectory_basis_coefficients: int,
    num_voxels_per_axis: int,
    min_bound_per_axis: float,
    max_bound_per_axis: float,
    voxel_dim: int,
    color_model_hidden_dim: int,
) -> Render:
    total_num_time_steps = num_time_steps_per_frame * num_frames
    camera_intrinsics = torch.tensor(
        [
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1],
        ]
    )
    return Render(
        camera_intrinsics=camera_intrinsics,
        num_time_steps=total_num_time_steps,
        num_coarse_samples_per_ray=num_coarse_samples_per_ray,
        num_fine_samples_per_ray=num_fine_samples_per_ray,
        near_depth=near_depth,
        far_depth=far_depth,
        num_scene_trajectory_basis_coefficients=num_scene_trajectory_basis_coefficients,
        num_camera_trajectory_basis_coefficients=num_camera_trajectory_basis_coefficients,
        num_voxels_per_axis=num_voxels_per_axis,
        min_bound_per_axis=min_bound_per_axis,
        max_bound_per_axis=max_bound_per_axis,
        voxel_dim=voxel_dim,
        color_model_hidden_dim=color_model_hidden_dim,
    )


def render_anchor_frame(
    model: Render,
    num_time_steps_per_frame: int,
    frame_index: torch.Tensor,
    warmup: bool = False,
) -> torch.Tensor:
    """
    Args:
        model: (Render)
        num_time_steps_per_frame: (int)
        frame_index: (torch.Tensor) Shape: (batch_size)
    Returns:
        images: (torch.Tensor) Shape: (batch_size, num_time_steps_per_frame, 3, image_height, image_width)
    """
    initial_time_step = frame_index * num_time_steps_per_frame

    time_step_batch = torch.ones(frame_index.shape[0], num_time_steps_per_frame)
    time_step_batch = (
        torch.arange(0, num_time_steps_per_frame) * time_step_batch
    )  # Shape: (batch_size, num_time_steps_per_frame)
    time_step_batch = (
        time_step_batch + initial_time_step[:, None]
    )  # Shape: (batch_size, num_time_steps_per_frame)
    time_step_batch = rearrange(time_step_batch, "b t -> (b t)")

    images = model(time_step_batch, warmup=warmup)
    images = rearrange(
        images,
        "(b t) c h w -> b t c h w",
        b=frame_index.shape[0],
    )
    return images


def render_anchor_frame_at_specific_pixels(
    model: Render,
    num_time_steps_per_frame: int,
    frame_index: torch.Tensor,
    pixel_indices: torch.Tensor,
    warmup: bool = False,
) -> torch.Tensor:
    """
    Args:
        model: (Render)
        num_time_steps_per_frame: (int)
        frame_index: (torch.Tensor) Shape: (num_pixels)
        pixel_indices: (torch.Tensor) Shape: (num_pixels, 2)
    Returns:
        pixels: (torch.Tensor) Shape: (num_pixels, num_time_steps_per_frame, 3)
    """
    initial_time_step = frame_index * num_time_steps_per_frame

    time_step_batch = torch.ones(
        pixel_indices.shape[0], num_time_steps_per_frame
    )  # Shape: (num_pixels, num_time_steps_per_frame)
    time_step_batch = (
        torch.arange(0, num_time_steps_per_frame) * time_step_batch
    )  # Shape: (num_pixels, num_time_steps_per_frame)
    time_step_batch = (
        time_step_batch + initial_time_step[:, None]
    )  # Shape: (num_pixels, num_time_steps_per_frame)
    time_step_batch = rearrange(
        time_step_batch, "p t -> (p t)"
    )  # Shape: (num_pixels * num_time_steps_per_frame)

    pixel_x = pixel_indices[:, 0]  # Shape: (num_pixels)
    pixel_y = pixel_indices[:, 1]  # Shape: (num_pixels)
    pixel_x = pixel_x[:, None]  # Shape: (num_pixels, 1)
    pixel_y = pixel_y[:, None]  # Shape: (num_pixels, 1)
    pixel_x = pixel_x.repeat(
        1, num_time_steps_per_frame
    )  # Shape: (num_pixels, num_time_steps_per_frame)
    pixel_y = pixel_y.repeat(
        1, num_time_steps_per_frame
    )  # Shape: (num_pixels, num_time_steps_per_frame)
    pixel_x = rearrange(
        pixel_x, "p t -> (p t)"
    )  # Shape: (num_pixels * num_time_steps_per_frame)
    pixel_y = rearrange(
        pixel_y, "p t -> (p t)"
    )  # Shape: (num_pixels * num_time_steps_per_frame)

    pixels = model(
        time_step_batch,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        warmup=warmup,
    )  # Shape: (num_pixels * num_time_steps_per_frame, 3)

    pixels = rearrange(
        pixels,
        "(p t) c -> p t c",
        p=pixel_indices.shape[0],
        t=num_time_steps_per_frame,
    )  # Shape: (num_pixels, num_time_steps_per_frame, 3)
    return pixels
