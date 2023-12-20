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

    def _render_entire_image(self, time_step: torch.Tensor, warmup: bool = False):
        """
        Args:
            time_step: (torch.Tensor) Shape: (batch_size)
        Returns:
            images: (torch.Tensor) Shape: (batch_size, num_time_steps_per_frame, 3, image_height, image_width)
        """
        time_step_batch = torch.ones(
            time_step.shape[0], self.num_time_steps_per_frame
        ).to(self.device)
        time_step_batch = (
            torch.arange(
                -(self.num_time_steps_per_frame // 2),
                (self.num_time_steps_per_frame // 2) + 1,
            ).to(self.device)
            * time_step_batch
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

    def _render_specific_pixels(
        self, time_step: torch.Tensor, pixel_indices: torch.Tensor, warmup: bool = False
    ):
        """
        Args:
            time_step: (torch.Tensor) Shape: (batch_size)
            pixel_indices: (torch.Tensor) Shape: (batch_size, num_pixels, 2)
        Returns:
            pixels: (torch.Tensor) Shape: (batch_size, num_pixels, num_time_steps_per_frame, 3)
        """
        num_pixels = pixel_indices.shape[1]
        time_step = time_step.repeat(
            num_pixels
        )
        pixel_indices = pixel_indices.reshape(-1, 2)  # Shape: (num_pixels, 2)
        time_step_batch = torch.ones(
            pixel_indices.shape[0], self.num_time_steps_per_frame
        )  # Shape: (num_pixels, num_time_steps_per_frame)
        time_step_batch = (
            torch.arange(
                -(self.num_time_steps_per_frame // 2),
                (self.num_time_steps_per_frame // 2) + 1,
            ).to(self.device)
            * time_step_batch
        )  # Shape: (num_pixels, num_time_steps_per_frame)
        time_step_batch = (
            time_step_batch + time_step[:, None]
        )  # Shape: (num_pixels, num_time_steps_per_frame)
        time_step_batch = rearrange(
            time_step_batch, "p t -> (p t)"
        )  # Shape: (num_pixels * num_time_steps_per_frame)

        pixel_x = pixel_indices[:, 0]  # Shape: (num_pixels)
        pixel_y = pixel_indices[:, 1]  # Shape: (num_pixels)
        pixel_x = pixel_x[:, None]  # Shape: (num_pixels, 1)
        pixel_y = pixel_y[:, None]  # Shape: (num_pixels, 1)
        pixel_x = pixel_x.repeat(
            1, self.num_time_steps_per_frame
        )  # Shape: (num_pixels, num_time_steps_per_frame)
        pixel_y = pixel_y.repeat(
            1, self.num_time_steps_per_frame
        )  # Shape: (num_pixels, num_time_steps_per_frame)
        pixel_x = rearrange(
            pixel_x, "p t -> (p t)"
        )  # Shape: (num_pixels * num_time_steps_per_frame)
        pixel_y = rearrange(
            pixel_y, "p t -> (p t)"
        )  # Shape: (num_pixels * num_time_steps_per_frame)

        pixels = self.model(
            time_step_batch,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            warmup=warmup,
        )  # Shape: (num_pixels * num_time_steps_per_frame, 3)

        pixels = rearrange(
            pixels,
            "(b p t) c -> b p t c",
            p=num_pixels,
            t=self.num_time_steps_per_frame,
        )  # Shape: (num_pixels, num_time_steps_per_frame, 3)
        return pixels

    def forward(
        self,
        time_step: torch.Tensor,
        pixel_indices: torch.Tensor = None,
        warmup: bool = False,
    ):
        """
        Args:
            time_step: (torch.Tensor) Shape: (batch_size)
            pixel_indices: (torch.Tensor) Shape: (batch_size, num_pixels, 2)
        Returns:
            images: (torch.Tensor) Shape: (batch_size, num_time_steps_per_frame, 3, image_height, image_width) or (batch_size, num_pixels, num_time_steps_per_frame, 3)
        """
        if pixel_indices is None:
            return self._render_entire_image(time_step, warmup=warmup)
        else:
            return self._render_specific_pixels(time_step, pixel_indices, warmup=warmup)
