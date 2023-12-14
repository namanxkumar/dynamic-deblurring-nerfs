import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from .render_ray import RenderRay
from .motion_models import CameraMotionModel


class RenderImage(nn.Module):
    def __init__(
        self,
        camera_intrinsics: torch.Tensor,
        num_time_steps: int,
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
        Initializes the RenderImage module.

        Args:
        - camera_intrinsics (torch.Tensor): Camera intrinsics. Shape: (3, 3)
        - num_time_steps (int): Number of time steps.
        - num_coarse_samples_per_ray (int): Number of coarse samples per ray.
        - num_fine_samples_per_ray (int): Number of fine samples per ray.
        - num_scene_trajectory_basis_coefficients (int): Number of coefficients for the scene trajectory basis.
        - num_camera_trajectory_basis_coefficients (int): Number of coefficients for the camera trajectory basis.
        - num_voxels_per_axis (int): Number of voxels per axis.
        - min_bound_per_axis (float): Minimum bound per axis.
        - max_bound_per_axis (float): Maximum bound per axis.
        - voxel_dim (int): Dimension of the voxel.
        - color_model_hidden_dim (int): Hidden dimension of the color model.
        """
        super().__init__()
        self.camera_intrinsics = camera_intrinsics
        self.inverse_camera_intrinsics = torch.inverse(camera_intrinsics)

        self.num_time_steps = num_time_steps
        self.camera_motion_model = CameraMotionModel(
            num_basis_coefficients=num_camera_trajectory_basis_coefficients,
            num_time_steps=num_time_steps,
        )
        self.render_ray = RenderRay(
            num_time_steps=num_time_steps,
            num_coarse_samples=num_coarse_samples_per_ray,
            num_fine_samples=num_fine_samples_per_ray,
            near_depth=near_depth,
            far_depth=far_depth,
            num_scene_trajectory_basis_coefficients=num_scene_trajectory_basis_coefficients,
            num_voxels_per_axis=num_voxels_per_axis,
            min_bound_per_axis=min_bound_per_axis,
            max_bound_per_axis=max_bound_per_axis,
            voxel_dim=voxel_dim,
            color_model_hidden_dim=color_model_hidden_dim,
        )

    def _taylor_expansion_A(self, x, n_terms: int = 10) -> torch.Tensor:
        """
        Gets the Taylor expansion for sinx/x.

        Args:
        - x (torch.Tensor): Input. Shape: (batch_size, 1)
        - n_terms (int): Number of terms.

        Returns:
        - A (torch.Tensor): Taylor expansion. Shape: (batch_size, 1)
        """
        A = torch.zeros_like(x)
        denominator = torch.tensor([1], dtype=torch.float32)
        for i in range(n_terms + 1):
            if i > 0:
                denominator *= (2 * i + 1) * (2 * i)
            A = A + (-1) ** i * x ** (2 * i) / denominator
        return A

    def _taylor_expansion_B(self, x, n_terms: int = 10) -> torch.Tensor:
        """
        Gets the Taylor expansion for (1-cosx)/(x^2).

        Args:
        - x (torch.Tensor): Input. Shape: (batch_size, 1)
        - n_terms (int): Number of terms.

        Returns:
        - B (torch.Tensor): Taylor expansion. Shape: (batch_size, 1)
        """
        B = torch.zeros_like(x)
        denominator = torch.tensor([1], dtype=torch.float32)
        for i in range(n_terms + 1):
            denominator *= (2 * i + 1) * (2 * i + 2)
            B = B + (-1) ** i * x ** (2 * i) / denominator
        return B

    def _taylor_expansion_C(self, x, n_terms: int = 10) -> torch.Tensor:
        """
        Gets the Taylor expansion for (x-sinx)/x^3.

        Args:
        - x (torch.Tensor): Input. Shape: (batch_size, 1)
        - n_terms (int): Number of terms.

        Returns:
        - C (torch.Tensor): Taylor expansion. Shape: (batch_size, 1)
        """
        C = torch.zeros_like(x)
        denominator = torch.tensor([1], dtype=torch.float32)
        for i in range(n_terms + 1):
            denominator *= (2 * i + 2) * (2 * i + 3)
            C = C + (-1) ** i * x ** (2 * i) / denominator
        return C

    def _so3_to_wx(self, so3: torch.Tensor) -> torch.Tensor:
        """
        Converts so(3) to w_x.

        Args:
        - so3 (torch.Tensor): so(3) matrix. Shape: (batch_size, 3, 3)

        Returns:
        - w_x (torch.Tensor): Skew symmetric matrix. Shape: (batch_size, 3, 3)
        """
        batch_size = so3.shape[0]
        w_x = torch.zeros((batch_size, 3, 3))
        w_x[:, 0, 1] = -so3[:, 2]
        w_x[:, 0, 2] = so3[:, 1]
        w_x[:, 1, 0] = so3[:, 2]
        w_x[:, 1, 2] = -so3[:, 0]
        w_x[:, 2, 0] = -so3[:, 1]
        w_x[:, 2, 1] = so3[:, 0]
        return w_x

    def _wx_to_SO3(self, w_x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Converts w_x to SO(3).

        Args:
        - w_x (torch.Tensor): Skew symmetric matrix. Shape: (batch_size, 3, 3)

        Returns:
        - SO3 (torch.Tensor): SO(3) matrix. Shape: (batch_size, 3, 3)
        """
        batch_size = w_x.shape[0]
        SO3 = (
            torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
            + self._taylor_expansion_A(theta).unsqueeze(-1).repeat(1, 3, 3) * w_x
            + self._taylor_expansion_B(theta).unsqueeze(-1).repeat(1, 3, 3)
            * (w_x @ w_x)
        )
        return SO3

    def _se3_to_SE3(self, se3: torch.Tensor) -> torch.Tensor:
        """
        Converts SE(3) to SE(3).

        Args:
        - se3 (torch.Tensor): SE(3) matrix. Shape: (batch_size, 6)

        Returns:
        - SE3 (torch.Tensor): SE(3) matrix. Shape: (batch_size, 4, 4)
        """
        batch_size = se3.shape[0]
        u, w = se3[:, :3], se3[:, 3:]

        w_x = self._so3_to_wx(w)
        theta = torch.norm(w, dim=1, keepdim=True)

        SE3 = torch.zeros((batch_size, 4, 4))
        SE3[:, :3, :3] = self._wx_to_SO3(w_x=w_x, theta=theta)

        V = (
            torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
            + self._taylor_expansion_B(theta).unsqueeze(-1).repeat(1, 3, 3) * w_x
            + self._taylor_expansion_C(theta).unsqueeze(-1).repeat(1, 3, 3)
            * (w_x @ w_x)
        )
        SE3[:, :3, 3] = torch.matmul(V, u)
        SE3[:, 3, 3] = 1
        return SE3

    def _get_ray(self, camera_pose: torch.Tensor) -> torch.Tensor:
        """
        Gets the ray directions.

        Args:
        - camera_pose (torch.Tensor): Camera pose in SE(3). Shape: (batch_size, 4, 4)

        Returns:
        - origin (torch.Tensor): Ray origins. Shape: (batch_size*image_height*image_width, 3)
        - direction (torch.Tensor): Ray directions. Shape: (batch_size*image_height*image_width, 3)
        """
        batch_size = camera_pose.shape[0]

        xx, yy = torch.meshgrid(
            torch.arange(self.camera_intrinsics[0, 2] * 2),
            torch.arange(self.camera_intrinsics[1, 2] * 2),
            indexing="xy",
        )

        xx, yy = xx.reshape(-1), yy.reshape(-1)
        direction = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
        direction = direction.unsqueeze(0).repeat(batch_size, 1, 1)
        direction = torch.bmm(
            torch.bmm(
                camera_pose[:, :3, :3].transpose(1, 2),
                self.inverse_camera_intrinsics.unsqueeze(0).repeat(batch_size, 1, 1),
            ),
            direction,
        ).reshape(-1, 3)
        origin = (
            camera_pose[:, :3, 3]
            .unsqueeze(1)
            .repeat(1, direction.shape[0], 1)
            .reshape(-1, 3)
        )

        return origin, direction

    def forward(self, time_step: torch.Tensor, warmup: bool = False) -> torch.Tensor:
        """
        Performs forward pass of the RenderImage module.

        Args:
        - time_step (torch.Tensor): Time step. Shape: (batch_size)

        Returns:
        - image (torch.Tensor): Rendered image. Shape: (batch_size, 3, image_height, image_width)
        """
        time_step = time_step.unsqueeze(-1)
        batch_size = time_step.shape[0]
        camera_pose_se3 = self.camera_motion_model(time_step, warmup)
        camera_pose_SE3 = self._se3_to_SE3(camera_pose_se3)

        ray_origin, ray_direction = self._get_ray(camera_pose_SE3)

        rendered_pixels = self.render_ray(ray_origin, ray_direction, time_step)

        image = rearrange(
            rendered_pixels,
            "(b h w) c -> b c h w",
            b=batch_size,
            h=self.camera_intrinsics[0, 2] * 2,
            w=self.camera_intrinsics[1, 2] * 2,
            c=3,
        )
        return image
