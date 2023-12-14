import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from .render_image import RenderImage


class Model(nn.Module):
    def __init__(
        self,
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
    ) -> None:
        """
        Initializes the Model module.

        Args:
        - focal_length (float): Focal length.
        - image_width (int): Image width.
        - image_height (int): Image height.
        - num_time_steps_per_frame (int): Number of time steps per frame.
        - num_frames (int): Number of frames.
        - num_coarse_samples_per_ray (int): Number of coarse samples per ray.
        - num_fine_samples_per_ray (int): Number of fine samples per ray.
        - near_depth (float): Near depth.
        - far_depth (float): Far depth.
        - num_scene_trajectory_basis_coefficients (int): Number of coefficients for the scene trajectory basis.
        - num_camera_trajectory_basis_coefficients (int): Number of coefficients for the camera trajectory basis.
        - num_voxels_per_axis (int): Number of voxels per axis.
        - min_bound_per_axis (float): Minimum bound per axis.
        - max_bound_per_axis (float): Maximum bound per axis.
        - voxel_dim (int): Dimension of the voxel.
        - color_model_hidden_dim (int): Hidden dimension of the color model.
        """
        super().__init__()
        self.num_steps_per_frame = num_time_steps_per_frame
        self.total_num_time_steps = num_time_steps_per_frame * num_frames
        camera_intrinsics = torch.tensor(
            [
                [focal_length, 0, image_width / 2],
                [0, focal_length, image_height / 2],
                [0, 0, 1],
            ]
        )
        self.render_image = RenderImage(
            camera_intrinsics=camera_intrinsics,
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
        )

    def forward(self, frame_index: torch.Tensor, warmup: bool = False) -> torch.Tensor:
        """
        Returns num_steps_per_frame images for the given frame index.

        Args:
        - frame_index (torch.Tensor): Shape: (batch_size)

        Returns:
        - image (torch.Tensor): Rendered image. Shape: (num_steps_per_frame, 3, image_height, image_width)
        """
        initial_time_step = frame_index * self.num_steps_per_frame
        time_step_batch = torch.arange(
            initial_time_step,
            initial_time_step + self.num_steps_per_frame,
        )
        return self.render_image(time_step_batch, warmup=warmup)
