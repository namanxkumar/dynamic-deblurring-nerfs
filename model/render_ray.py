import torch
from torch import nn

from einops import rearrange

from .motion_models import SceneMotionModel
from .spatial_model import (
    SceneSpatialVoxelModel,
    SceneSpatialDensityModel,
    SceneSpatialColorModel,
)


class RenderRay(nn.Module):
    def __init__(
        self,
        num_time_steps: int,
        num_samples: int,
        near_depth: float,
        far_depth: float,
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
        - near_depth (float): Near plane of the camera.
        - far_depth (float): Far plane of the camera.
        - num_time_steps (int): Number of time steps.
        - num_voxels_per_axis (int): Number of voxels per voxel grid axis.
        - min_bound_per_axis (float): Minimum bound per voxel grid axis.
        - max_bound_per_axis (float): Maximum bound per voxel grid axis.
        - voxel_dim (int): Dimension of the voxel.
        - color_model_hidden_dim (int): Hidden dimension of the color model.
        """
        super().__init__()
        self.num_samples = num_samples

        assert near_depth < far_depth, "Near depth must be less than far depth."
        assert (
            near_depth > 0 and far_depth > 0
        ), "Near depth and far depth must be positive."

        self.near_depth = near_depth
        self.far_depth = far_depth

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

    def _sample_points_on_ray(
        self, ray_origin: torch.Tensor, ray_direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples points on the ray.

        Args:
        - ray_origin (torch.Tensor): Origin of the ray. Shape: (batch_size, 3)
        - ray_direction (torch.Tensor): Direction of the ray. Shape: (batch_size, 3)

        Returns:
        - points_on_ray (torch.Tensor): Points sampled on the ray. Shape: (batch_size, num_samples, 3)
        """
        near_depth = self.near_depth * torch.ones_like(ray_direction[:, 2])
        far_depth = self.far_depth * torch.ones_like(ray_direction[:, 2])

        step = (far_depth - near_depth) / (self.num_samples - 1)
        z_values = torch.stack(
            [near_depth + i * step for i in range(self.num_samples)], dim=1
        )

        ray_direction = ray_direction.unsqueeze(1).repeat(
            1, self.num_samples, 1
        )  # (B, N, 3)
        ray_origin = ray_origin.unsqueeze(1).repeat(1, self.num_samples, 1)  # (B, N, 3)

        return ray_origin + ray_direction * z_values.unsqueeze(-1)

    def forward(
        self,
        ray_origin: torch.Tensor,
        ray_direction: torch.Tensor,
        time_step: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs forward pass of the RenderRay module.

        Args:
        - ray_origin (torch.Tensor): Origin of the ray. Shape: (batch_size, 3)
        - ray_direction (torch.Tensor): Direction of the ray. Shape: (batch_size, 3)
        - time_step (torch.Tensor): Time step of the ray. Shape: (batch_size, 1)
        - view_direction (torch.Tensor): View direction of the ray. Shape: (batch_size, 3)

        Returns:
        - rgb_map (torch.Tensor): RGB values of the rendered scene. Shape: (batch_size, 3)
        """
        batch_size = ray_origin.shape[0]
        points_on_ray = self._sample_points_on_ray(ray_origin, ray_direction)
        warped_points_on_ray = self.scene_motion_model(
            points_on_ray.reshape(-1, 3), time_step
        )
        spatial_voxel_features = self.spatial_voxel_model(warped_points_on_ray)
        density = self.spatial_density_model(spatial_voxel_features)
        color = self.spatial_color_model(spatial_voxel_features, ray_direction)
        rgb_map = density * color
        rgb_map = rearrange(
            rgb_map, "(b n) c -> b n c", b=batch_size, n=self.num_samples
        )
        rgb_map = torch.sum(rgb_map, dim=1)
        return rgb_map
