import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from .utils import PositionalEmbedding

from typing import Tuple


class SceneSpatialVoxelModel(nn.Module):
    def __init__(
        self,
        num_voxels_per_axis: int,
        min_bound_per_axis: float,
        max_bound_per_axis: float,
        voxel_dim: int,
    ) -> None:
        """
        Initializes the SceneSpatialVoxelModel module.

        Args:
        - num_voxels_per_axis (int): Number of voxels per axis.
        - min_bound_per_axis (float): Minimum bound per axis.
        - max_bound_per_axis (float): Maximum bound per axis.
        - voxel_dim (int): Dimension of the voxel.
        """
        super().__init__()
        self.min_bound = min_bound_per_axis
        self.max_bound = max_bound_per_axis
        self.voxel_dim = voxel_dim

        self.voxel_grid = torch.nn.Parameter(
            torch.zeros(
                (
                    1,
                    voxel_dim,
                    num_voxels_per_axis,
                    num_voxels_per_axis,
                    num_voxels_per_axis,
                ),
                dtype=torch.float32,
            )
        )

    def forward(
        self, warped_sample_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Queries the voxel grid at the given location with trilinear interpolation

        Args:
        - warped_sample_points (torch.Tensor): Warped sample points tensor of shape (batch_size, 3).

        Returns:
        - features (torch.Tensor): Trilinearly interpolated spatial voxel at the given location.
                                  Shape: (batch_size, voxel_dim)
        """
        normalized_sample_points = (
            (warped_sample_points - self.min_bound) / (self.max_bound - self.min_bound)
        ).flip((-1,)) * 2 - 1

        features = torch.squeeze(
            F.grid_sample(
                self.voxel_grid,
                normalized_sample_points[None, None, None, ...],
                mode="bilinear",
                align_corners=True,
            )
        )

        return rearrange(features, "c b -> b c")


class SceneSpatialDensityModel(nn.Module):
    def __init__(self) -> None:
        """
        Initializes the SceneSpatialDensityModel module.
        """
        super().__init__()

    def forward(self, spatial_voxel_features) -> torch.Tensor:
        """
        Computes the density from voxel features.

        Args:
        - spatial_voxel_features (torch.Tensor): Spatial voxel features. Shape: (batch_size, voxel_dim)

        Returns:
        - density (torch.Tensor): Density at the given location and time step. Shape: (batch_size, 1)
        """
        return F.relu(spatial_voxel_features)


class SceneSpatialColorModel(nn.Module):
    def __init__(
        self,
        voxel_feature_dim,
        hidden_dim,
        voxel_feature_positional_embedding_dim=6,
        view_direction_positional_embedding_dim=6,
    ) -> None:
        """
        Initializes the SceneSpatialColorModel module.

        Args:
        - voxel_feature_dim (int): Dimension of the voxel feature.
        - hidden_dim (int): Dimension of the hidden layer.
        - voxel_feature_positional_embedding_dim (int): Dimension of the positional embedding for voxel features.
        - view_direction_positional_embedding_dim (int): Dimension of the positional embedding for view directions.
        """
        super().__init__()
        self.voxel_feature_positional_embedding = PositionalEmbedding(
            voxel_feature_positional_embedding_dim
        )
        self.view_direction_positional_embedding = PositionalEmbedding(
            view_direction_positional_embedding_dim
        )

        self.interstitial_feature_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                voxel_feature_dim + voxel_feature_positional_embedding_dim,
                hidden_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.view_mlp = torch.nn.Linear(
            hidden_dim + view_direction_positional_embedding_dim, 3
        )

    def forward(self, spatial_voxel_features, view_direction):
        """
        Computes the color from voxel features in a given view direction.

        Args:
        - spatial_voxel_features (torch.Tensor): Spatial voxel features. Shape: (batch_size, voxel_dim)
        - view_direction (torch.Tensor): View directions. Shape: (batch_size, 3)

        Returns:
        - color (torch.Tensor): Color at the given location. Shape: (batch_size, 3)
        """
        feature_encoding = torch.cat(
            (
                spatial_voxel_features,
                self.voxel_feature_positional_embedding(spatial_voxel_features),
            ),
            dim=-1,
        )
        interstitial_features = self.interstitial_feature_mlp(feature_encoding)

        view_direction_encoding = torch.cat(
            (
                view_direction,
                self.view_direction_positional_embedding(view_direction),
            ),
            dim=-1,
        )

        view_features = self.view_mlp(
            torch.cat(
                (
                    interstitial_features,
                    view_direction_encoding,
                ),
                dim=-1,
            )
        )

        return torch.sigmoid(view_features)


# Code for rodynrf implemented voxel field using TensorRF
#
# self.num_density_components = num_density_components
# self.num_color_components = num_color_components
# self.color_dim = color_dim

# self.plane_coefficients = torch.nn.Parameter(
#     0.1
#     * torch.randn(
#         (
#             3,
#             self.num_color_components + self.num_density_components,
#             self.num_voxels,
#             self.num_voxels,
#         )
#     )
# )
# self.line_coefficients = torch.nn.Parameter(
#     0.1
#     * torch.randn(
#         (
#             3,
#             self.num_color_components + self.num_density_components,
#             self.num_voxels,
#             1,
#         )
#     )
# )
# self.basis_matrix = torch.nn.Linear(
#     self.num_color_components * 3, self.color_dim, bias=False
# )

# def _compute_voxel_features(self, sample_point: torch.Tensor):
#     coordinate_plane = torch.stack((sample_point[..., [0, 1]], sample_point[..., [0, 2]], sample_point[..., [1, 2]])).detach()
#     coordinate_line = torch.stack((sample_point[..., 0], sample_point[..., 1], sample_point[..., 2]))
#     coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

#     plane_features = F.grid_sample(
#         self.plane_coefficients[:, -self.num_density_components:], coordinate_plane, align_corners=True
#     ).view(-1, *sample_point.shape[:1])
#     line_features = F.grid_sample(
#         self.line_coefficients[:, -self.num_density_components:], coordinate_line, align_corners=True
#     ).view(-1, *sample_point.shape[:1])

#     density_features = torch.sum(plane_features * line_features, dim=0)

#     plane_features = F.grid_sample(
#         self.plane_coefficients[:, :self.num_color_components], coordinate_plane, align_corners=True
#     ).view(3 * self.num_color_components, -1)
#     line_features = F.grid_sample(
#         self.line_coefficients[:, :self.num_color_components], coordinate_line, align_corners=True
#     ).view(3 * self.num_color_components, -1)

#     color_features = self.basis_matrix((plane_features * line_features).T)

#     return density_features, color_features
