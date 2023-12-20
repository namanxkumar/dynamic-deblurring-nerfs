import torch
from torch import nn

from einops import rearrange

from .motion_models import SceneMotionModel
from .spatial_model import (
    SceneSpatialVoxelModel,
    SceneSpatialDensityModel,
    SceneSpatialColorModel,
)

from typing import Optional


class RenderRay(nn.Module):
    def __init__(
        self,
        num_time_steps: int,
        num_coarse_samples: int,
        num_fine_samples: int,
        near_depth: float,
        far_depth: float,
        num_scene_trajectory_basis_coefficients: int,
        num_voxels_per_axis: int,
        min_bound_per_axis: float,
        max_bound_per_axis: float,
        voxel_dim: int,
        color_model_hidden_dim: int,
        device: torch.device,
    ) -> None:
        """
        Initializes the RenderRay module.

        Args:
        - num_time_steps (int): Number of time steps.
        - num_coarse_samples (int): Number of coarse samples.
        - num_fine_samples (int): Number of fine samples.
        - near_depth (float): Near plane of the camera.
        - far_depth (float): Far plane of the camera.
        - num_scene_trajectory_basis_coefficients (int): Number of coefficients for the scene trajectory basis.
        - num_voxels_per_axis (int): Number of voxels per voxel grid axis.
        - min_bound_per_axis (float): Minimum bound per voxel grid axis.
        - max_bound_per_axis (float): Maximum bound per voxel grid axis.
        - voxel_dim (int): Dimension of the voxel.
        - color_model_hidden_dim (int): Hidden dimension of the color model.
        """
        super().__init__()
        self.device = device
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples

        assert near_depth < far_depth, "Near depth must be less than far depth."
        assert (
            near_depth > 0 and far_depth > 0
        ), "Near depth and far depth must be positive."

        self.near_depth = near_depth
        self.far_depth = far_depth

        self.scene_motion_model = SceneMotionModel(
            num_basis_coefficients=num_scene_trajectory_basis_coefficients,
            num_time_steps=num_time_steps,
            device=self.device,
        )
        self.spatial_voxel_model = SceneSpatialVoxelModel(
            num_voxels_per_axis=num_voxels_per_axis,
            min_bound_per_axis=min_bound_per_axis,
            max_bound_per_axis=max_bound_per_axis,
            voxel_dim=voxel_dim,
            device=self.device,
        )
        self.spatial_density_model = SceneSpatialDensityModel(
            device=self.device,
        )
        self.spatial_color_model = SceneSpatialColorModel(
            voxel_feature_dim=voxel_dim,
            hidden_dim=color_model_hidden_dim,
            device=self.device,
        )

    @staticmethod
    def _sample_probability_distribution(
        bins: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Samples from a probability distribution of coarse samples. Copied from NeRF.
        """
        N_samples = num_samples
        M = weights.shape[1]
        weights += 1e-5
        # Get pdf
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
        cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
        cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)  # [N_rays, M+1]

        # Take uniform samples
        if deterministic:
            print(N_samples)
            u = torch.linspace(0.0, 1.0, N_samples, device=bins.device)
            u = u.unsqueeze(0).repeat(bins.shape[0], 1)  # [N_rays, N_samples]
        else:
            u = torch.rand(bins.shape[0], N_samples, device=bins.device)

        # Invert CDF
        above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
        for i in range(M):
            above_inds += (u >= cdf[:, i : i + 1]).long()

        # random sample inside each bin
        below_inds = torch.clamp(above_inds - 1, min=0)
        inds_g = torch.stack((below_inds, above_inds), dim=2)  # [N_rays, N_samples, 2]

        cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
        cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

        bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
        bins_g = torch.gather(
            input=bins, dim=-1, index=inds_g
        )  # [N_rays, N_samples, 2]

        # fix numeric issue
        denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[:, :, 0]) / denom

        samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

        return samples

    def _sample_points_on_ray(
        self,
        ray_origin: torch.Tensor,
        ray_direction: torch.Tensor,
        depth_values: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Samples points on the ray.

        Args:
        - ray_origin (torch.Tensor): Origin of the ray. Shape: (batch_size, 3)
        - ray_direction (torch.Tensor): Direction of the ray. Shape: (batch_size, 3)
        - depth_values (torch.Tensor): Depth values of the points sampled on the ray. Shape: (batch_size, num_samples)
        - weights (torch.Tensor): Weights of the probability distribution. Shape: (batch_size, num_samples)

        Returns:
        - points_on_ray (torch.Tensor): Points sampled on the ray. Shape: (batch_size, num_samples, 3)
        - depth_values (torch.Tensor): Depth values of the points sampled on the ray. Shape: (batch_size, num_samples)
        """
        if weights is None and depth_values is None:
            num_samples = self.num_coarse_samples
            total_samples = num_samples

            near_depth = self.near_depth * torch.ones_like(ray_direction[:, 2]).to(
                self.device
            )
            far_depth = self.far_depth * torch.ones_like(ray_direction[:, 2]).to(
                self.device
            )

            step = (far_depth - near_depth) / (num_samples - 1)
            depth_values = torch.stack(
                [near_depth + i * step for i in range(num_samples)], dim=1
            ).to(
                self.device
            )  # (B, N)
        else:
            num_samples = self.num_fine_samples
            total_samples = self.num_coarse_samples + num_samples
            mid_depth_values = 0.5 * (
                depth_values[..., 1:] + depth_values[..., :-1]
            )  # (B, N-1)
            weights = weights[:, 1:-1]  # (B, N-2)
            sampled_depth_values = self._sample_probability_distribution(
                bins=mid_depth_values, weights=weights, num_samples=num_samples
            )  # (B, N)
            depth_values, _ = torch.sort(
                torch.cat((depth_values, sampled_depth_values), dim=-1).to(self.device),
                dim=-1,
            )  # (B, 2N-1)

        ray_direction = ray_direction.unsqueeze(1).repeat(
            1, total_samples, 1
        )  # (B, N, 3)
        ray_origin = ray_origin.unsqueeze(1).repeat(1, total_samples, 1)  # (B, N, 3)

        return ray_origin + (ray_direction * depth_values.unsqueeze(-1)), depth_values

    def _process_raw_spatial_outputs(
        self,
        raw_density: torch.Tensor,
        raw_color: torch.Tensor,
        depth_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Processes raw spatial outputs.

        Args:
        - raw_density (torch.Tensor): Raw density. Shape: (batch_size, num_samples, 1)
        - raw_color (torch.Tensor): Raw color. Shape: (batch_size, num_samples, 3)
        - depth_values (torch.Tensor): Depth values of the points sampled on the ray. Shape: (batch_size, num_samples)

        Returns:
        - density (torch.Tensor): Density. Shape: (batch_size, 1)
        - color (torch.Tensor): Color. Shape: (batch_size, 3)
        - weights (torch.Tensor): Weights. Shape: (batch_size, num_samples)
        """
        distances = depth_values[..., 1:] - depth_values[..., :-1]
        distances = torch.cat(
            [
                distances,
                torch.Tensor([1e10])
                .expand(distances[:, :1].shape)
                .to(self.device),  # final distance segment is ~infinite
            ],
            dim=-1,
        )  # (B, N)

        alpha = 1.0 - torch.exp(-raw_density.squeeze(-1) * distances)  # (B, N)
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[:, :-1]
        T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)
        weights = (T * alpha).to(self.device)

        depth = torch.sum(weights * depth_values, dim=-1).unsqueeze(-1)  # (B, 1)
        color = torch.sum(weights.unsqueeze(-1) * raw_color, dim=-2)  # (B, 3)

        return depth, color, weights

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
        - color_map (torch.Tensor): RGB values of the rendered scene. Shape: (batch_size, 3)
        """
        batch_size = ray_origin.shape[0]

        with torch.no_grad():
            coarse_points_on_ray, coarse_depth_values = self._sample_points_on_ray(
                ray_origin, ray_direction
            )
            coarse_warped_points_on_ray = self.scene_motion_model(
                coarse_points_on_ray.reshape(-1, 3),
                time_step.repeat(self.num_coarse_samples, 1),
            )
            coarse_spatial_voxel_features = self.spatial_voxel_model(
                coarse_warped_points_on_ray
            )
            coarse_raw_density = self.spatial_density_model(
                coarse_spatial_voxel_features
            )
            coarse_raw_color = self.spatial_color_model(
                coarse_spatial_voxel_features,
                ray_direction.repeat(self.num_coarse_samples, 1),
            )
            _, _, coarse_weights = self._process_raw_spatial_outputs(
                rearrange(
                    coarse_raw_density,
                    "(b n) 1 -> b n 1",
                    b=batch_size,
                    n=self.num_coarse_samples,
                ),
                rearrange(
                    coarse_raw_color,
                    "(b n) c -> b n c",
                    b=batch_size,
                    n=self.num_coarse_samples,
                    c=3,
                ),
                coarse_depth_values,
            )

        points_on_ray, depth_values = self._sample_points_on_ray(
            ray_origin,
            ray_direction,
            coarse_depth_values,
            coarse_weights.clone().detach(),
        )
        warped_points_on_ray = self.scene_motion_model(
            points_on_ray.reshape(-1, 3),
            time_step.repeat(self.num_coarse_samples + self.num_fine_samples, 1),
        )
        spatial_voxel_features = self.spatial_voxel_model(warped_points_on_ray)
        raw_density = self.spatial_density_model(spatial_voxel_features)
        raw_color = self.spatial_color_model(
            spatial_voxel_features,
            ray_direction.repeat(self.num_coarse_samples + self.num_fine_samples, 1),
        )
        _, color_map, _ = self._process_raw_spatial_outputs(
            rearrange(raw_density, "(b n) 1 -> b n 1", b=batch_size),
            rearrange(raw_color, "(b n) c -> b n c", b=batch_size, c=3),
            depth_values,
        )
        return color_map
