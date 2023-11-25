import torch
from torch import nn

from einops import rearrange

from .motion_models import SceneMotionModel
from .spatial_model import (
    SceneSpatialVoxelModel,
    SceneSpatialDensityModel,
    SceneSpatialColorModel,
)


class SamplePointsOnRay(nn.Module):
    def __init__(self, num_samples) -> None:
        """
        Initializes the SamplePointsOnRay module.

        Args:
        - num_samples (int): Number of samples on the ray.
        """
        super().__init__()
        self.num_samples = num_samples

    def forward(self, ray_start: torch.Tensor, ray_end: torch.Tensor) -> torch.Tensor:
        """
        Computes the points sampled along the ray.

        Args:
        - ray_start (torch.Tensor): Starting point of the ray. Shape: (batch_size, 3)
        - ray_end (torch.Tensor): Ending point of the ray. Shape: (batch_size, 3)

        Returns:
        - points_on_ray (torch.Tensor): Points sampled on the ray. Shape: (batch_size, num_samples, 3)
        """
        batch_size = ray_start.shape[0]
        t = torch.linspace(0, 1, self.num_samples)
        t = t.view(1, self.num_samples, 1).repeat(batch_size, 1, 1)
        points_on_ray = ray_start[:, None, :] + t * (
            ray_end[:, None, :] - ray_start[:, None, :]
        )
        return points_on_ray


class RenderRay(nn.Module):
    def __init__(
        self,
        num_time_steps: int,
        num_samples: int,
        num_scene_trajectory_basis_coefficients: int,
        num_voxels_per_axis: int,
        min_bound_per_axis: float,
        max_bound_per_axis: float,
        voxel_dim: int,
        color_model_hidden_dim: int,
    ) -> None:
        """
        Initializes the RenderRay module.

        Args:
        - num_samples (int): Number of samples on the ray.
        - num_scene_trajectory_basis_coefficients (int): Number of coefficients for the scene trajectory basis.
        - num_time_steps (int): Number of time steps.
        - num_voxels_per_axis (int): Number of voxels per voxel grid axis.
        - min_bound_per_axis (float): Minimum bound per voxel grid axis.
        - max_bound_per_axis (float): Maximum bound per voxel grid axis.
        - voxel_dim (int): Dimension of the voxel.
        - color_model_hidden_dim (int): Hidden dimension of the color model.
        """
        super().__init__()
        self.num_samples = num_samples
        self.sample_points_on_ray = SamplePointsOnRay(num_samples)
        self.scene_motion_model = SceneMotionModel(
            num_basis_coefficients=num_scene_trajectory_basis_coefficients,
            num_time_steps=num_time_steps,
        )
        self.spatial_voxel_model = SceneSpatialVoxelModel(
            num_voxels_per_axis=num_voxels_per_axis,
            min_bound_per_axis=min_bound_per_axis,
            max_bound_per_axis=max_bound_per_axis,
            voxel_dim=voxel_dim,
        )
        self.spatial_density_model = SceneSpatialDensityModel()
        self.spatial_color_model = SceneSpatialColorModel(
            voxel_feature_dim=voxel_dim, hidden_dim=color_model_hidden_dim
        )

    def forward(
        self,
        ray_start: torch.Tensor,
        ray_end: torch.Tensor,
        time_step: torch.Tensor,
        view_direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs forward pass of the RenderRay module.

        Args:
        - ray_start (torch.Tensor): Starting point of the ray. Shape: (batch_size, 3)
        - ray_end (torch.Tensor): Ending point of the ray. Shape: (batch_size, 3)
        - time_step (torch.Tensor): Time step of the ray. Shape: (batch_size, 1)
        - view_direction (torch.Tensor): View direction of the ray. Shape: (batch_size, 3)

        Returns:
        - rgb_map (torch.Tensor): RGB values of the rendered scene. Shape: (batch_size, 3)
        """
        batch_size = ray_start.shape[0]
        points_on_ray = self.sample_points_on_ray(ray_start, ray_end)
        warped_points_on_ray = self.scene_motion_model(
            rearrange(
                points_on_ray,
                "b n c -> (b n) c",
                b=batch_size,
                n=self.num_samples,
                c=3,
            ),
            time_step,
        )
        spatial_voxel_features = self.spatial_voxel_model(warped_points_on_ray)
        density = self.spatial_density_model(spatial_voxel_features)
        color = self.spatial_color_model(spatial_voxel_features, view_direction)
        rgb_map = density * color
        rgb_map = rearrange(
            rgb_map, "(b n) c -> b n c", b=batch_size, n=self.num_samples
        )
        rgb_map = torch.sum(rgb_map, dim=1)
        return rgb_map
