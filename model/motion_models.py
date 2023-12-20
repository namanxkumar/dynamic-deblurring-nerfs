import torch
from torch import nn
from torch.nn import functional as F

from .utils import PositionalEmbedding

from typing import List


class MotionModel(nn.Module):
    def __init__(
        self,
        num_basis_coefficients: int,
        input_dim: int,
        output_dim: int,
        num_time_steps: int,
        device: torch.device,
        num_linear_layers: int = 8,
        skip_connections: List[int] = [4],
        hidden_dim: int = 256,
        positional_embedding_dim: int = 16,
        num_matrices: int = 1,
    ) -> None:
        """
        Initializes the MotionModel.

        Args:
            num_basis_coefficients (int): The number of basis coefficients used for the DCT trajectory parametrization.
            input_dim (int): The dimension of the input.
            output_dim (int): The dimension of the output.
            num_time_steps (int): The number of time steps in the trajectory.
            num_linear_layers (int, optional): The number of linear layers in the model. Defaults to 8.
            skip_connections (List[int], optional): The indices of the linear layers where skip connections are applied. Defaults to [4].
            hidden_dim (int, optional): The dimension of the hidden layers. Defaults to 256.
            positional_embedding_dim (int, optional): The dimension of the positional embedding. Defaults to 16.
            num_matrices (int, optional): The number of dct basis function matrices in the model. Defaults to 1.
        """
        super().__init__()
        self.device = device

        self.basis_function_matrix = nn.Parameter(
            self._initialize_dct_basis_function_matrix(
                num_time_steps, num_basis_coefficients // num_matrices
            ).repeat(1, num_matrices)
        )

        self.skip_connections = skip_connections

        self.positional_embedding = PositionalEmbedding(
            embedding_dim=positional_embedding_dim, device=self.device
        )

        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_dim + (positional_embedding_dim*input_dim), hidden_dim)]
            + [
                nn.Linear(hidden_dim, hidden_dim)
                if layer_index not in skip_connections
                else nn.Linear(hidden_dim + input_dim + (positional_embedding_dim*input_dim), hidden_dim)
                for layer_index in range(num_linear_layers - 1)
            ]
        )

        self.output_layer = nn.Linear(hidden_dim, (num_basis_coefficients // num_matrices) * output_dim)
        self.output_layer.weight.data.fill_(0.0)
        self.output_layer.bias.data.fill_(0.0)

    def _initialize_dct_basis_function_matrix(
        self, num_time_steps: int, num_basis_coefficients: int
    ) -> torch.Tensor:
        """
        Initializes the DCT basis function matrix.

        Args:
            num_basis_coefficients (int): The number of basis coefficients used for the DCT trajectory parametrization.
            num_time_steps (int): The number of time steps in the trajectory.

        Returns:
            torch.Tensor: The initialized DCT basis function matrix.
        """
        dct_basis_function_matrix = torch.zeros(
            [num_time_steps, num_basis_coefficients]
        ).to(self.device)

        for time_step_index in range(num_time_steps):
            for basis_coefficient_index in range(num_basis_coefficients):
                dct_basis_function_matrix[time_step_index, basis_coefficient_index] = (
                    (2.0 / num_time_steps) ** 0.5
                ) * torch.cos(
                    torch.tensor(
                        torch.pi
                        / (2.0 * num_time_steps)
                        * (2.0 * time_step_index + 1.0)
                        * basis_coefficient_index
                    )
                )

        return dct_basis_function_matrix


class CameraMotionModel(MotionModel):
    """
    This class represents a camera motion model that computes the camera pose at any given timestep based on a DCT trajectory parametrization.

    Args:
        num_basis_coefficients (int): The number of basis coefficients used for the DCT trajectory parametrization.
        num_time_steps (int): The number of time steps in the trajectory.
        num_linear_layers (int, optional): The number of linear layers in the model. Defaults to 16.
        skip_connections (list[int], optional): The indices of the linear layers where skip connections are applied. Defaults to [4, 8, 12].
        hidden_dim (int, optional): The dimension of the hidden layers. Defaults to 256.
        positional_embedding_dim (int, optional): The dimension of the positional embedding. Defaults to 32.

    Attributes:
        basis_function_matrix (nn.Parameter): The parameter representing the DCT basis function matrix.
        skip_connections (list[int]): The indices of the linear layers where skip connections are applied.
        positional_embedding (PositionalEmbedding): The positional embedding layer.
        linear_layers (nn.ModuleList): The list of linear layers in the model.
        output_layer (nn.Linear): The output layer of the model.

    Methods:
        _compute_camera_pose(se3_coefficients, time_step): Computes the camera pose at the given time step based on the se(3) coefficients.
        forward(time_step): Computes the se(3) camera pose at the given time step.

    Inherits from:
        MotionModel: The base class for camera and scene motion models.
    """

    def __init__(
        self,
        num_basis_coefficients: int,
        num_time_steps: int,
        device: torch.device,
        num_linear_layers: int = 16,
        skip_connections: List[int] = [4, 8, 12],
        hidden_dim: int = 256,
        positional_embedding_dim: int = 8,
    ) -> None:
        """
        Initializes the CameraMotionModel.

        Args:
            num_basis_coefficients (int): The number of basis coefficients used for the DCT trajectory parametrization.
            num_time_steps (int): The number of time steps in the trajectory.
            num_linear_layers (int, optional): The number of linear layers in the model. Defaults to 16.
            skip_connections (List[int], optional): The indices of the linear layers where skip connections are applied. Defaults to [4, 8, 12].
            hidden_dim (int, optional): The dimension of the hidden layers. Defaults to 256.
            positional_embedding_dim (int, optional): The dimension of the positional embedding. Defaults to 32.
        """
        super().__init__(
            num_basis_coefficients=num_basis_coefficients
            * 2,  # half for translation, half for rotation
            input_dim=1,
            output_dim=6,
            device=device,
            num_time_steps=num_time_steps,
            num_linear_layers=num_linear_layers,
            skip_connections=skip_connections,
            hidden_dim=hidden_dim,
            positional_embedding_dim=positional_embedding_dim,
            num_matrices=2,
        )

    def _compute_camera_pose(
        self, se3_coefficients: torch.Tensor, time_step: int, warmup: bool = False
    ) -> torch.Tensor:
        """
        Computes the camera pose at the given time step based on the se(3) coefficients.

        Args:
            se3_coefficients (torch.Tensor): The se(3) coefficients. Shape: (batch_size, num_basis_coefficients*6)
            time_step (torch.Tensor): The time step. Shape: (batch_size, 1)
            warmup (bool, optional): Whether to use the warmup mode. Defaults to False.

        Returns:
            torch.Tensor: The camera pose at the given time step. Shape: (batch_size, 6)
        """
        (
            translation_coefficient_x,
            translation_coefficient_y,
            translation_coefficient_z,
            rotation_coefficient_x,
            rotation_coefficient_y,
            rotation_coefficient_z,
        ) = torch.chunk(se3_coefficients, 6, dim=-1)
        
        if warmup:
            with torch.no_grad():
                (
                    translation_basis_coefficients,
                    rotation_basis_coefficients,
                ) = torch.chunk(
                    torch.index_select(
                        self.basis_function_matrix, 0, torch.squeeze(time_step).int()
                    ).to(self.device),
                    2,
                    dim=-1,
                )
        else:
            translation_basis_coefficients, rotation_basis_coefficients = torch.chunk(
                torch.index_select(
                    self.basis_function_matrix, 0, torch.squeeze(time_step).int()
                ).to(self.device),
                2,
                dim=-1,
            )
        
        return torch.cat(
            [
                torch.sum(
                    translation_basis_coefficients * translation_coefficient_x,
                    dim=-1,
                    keepdim=True,
                ),
                torch.sum(
                    translation_basis_coefficients * translation_coefficient_y,
                    dim=-1,
                    keepdim=True,
                ),
                torch.sum(
                    translation_basis_coefficients * translation_coefficient_z,
                    dim=-1,
                    keepdim=True,
                ),
                torch.sum(
                    rotation_basis_coefficients * rotation_coefficient_x,
                    dim=-1,
                    keepdim=True,
                ),
                torch.sum(
                    rotation_basis_coefficients * rotation_coefficient_y,
                    dim=-1,
                    keepdim=True,
                ),
                torch.sum(
                    rotation_basis_coefficients * rotation_coefficient_z,
                    dim=-1,
                    keepdim=True,
                ),
            ],
            dim=-1,
        )

    def forward(self, time_step: torch.Tensor, warmup: bool = False) -> torch.Tensor:
        """
        Computes the se(3) camera pose at the given time step.

        Args:
            time_step (torch.Tensor): The time step. Shape: (batch_size, 1)
            warmup (bool, optional): Whether to use the warmup mode. Defaults to False.

        Returns:
            torch.Tensor: The se(3) camera pose at the given time step. Shape: (batch_size, 6)
        """
        embeddings = torch.cat(
            (
                time_step,
                torch.reshape(
                    self.positional_embedding.forward(torch.squeeze(time_step)).to(
                        self.device
                    ),
                    (time_step.shape[0], -1),
                ),
            ),
            dim=-1,
        )

        hidden_state = embeddings

        for index in range(len(self.linear_layers)):
            hidden_state = self.linear_layers[index](hidden_state)
            hidden_state = F.relu(hidden_state)

            if index in self.skip_connections:
                hidden_state = torch.cat([hidden_state, embeddings], dim=-1)

        se3_coefficients = self.output_layer(hidden_state)

        return self._compute_camera_pose(se3_coefficients, time_step, warmup=warmup)


class SceneMotionModel(MotionModel):
    """
    This class represents a scene motion model that computes the warped points at any given timestep based on a DCT trajectory parametrization.

    Args:
        num_basis_coefficients (int): The number of basis coefficients used for the DCT trajectory parametrization.
        num_time_steps (int): The number of time steps in the trajectory.
        num_linear_layers (int, optional): The number of linear layers in the model. Defaults to 8.
        skip_connections (list[int], optional): The indices of the linear layers where skip connections are applied. Defaults to [4].
        hidden_dim (int, optional): The dimension of the hidden layers. Defaults to 256.
        positional_embedding_dim (int, optional): The dimension of the positional embedding. Defaults to 16.

    Attributes:
        basis_function_matrix (nn.Parameter): The parameter representing the DCT basis function matrix.
        skip_connections (list[int]): The indices of the linear layers where skip connections are applied.
        positional_embedding (PositionalEmbedding): The positional embedding layer.
        linear_layers (nn.ModuleList): The list of linear layers in the model.
        output_layer (nn.Linear): The output layer of the model.

    Methods:
        _compute_warped_points(position_coefficients, time_step): Computes the warped points at the given time step based on the position coefficients.
        forward(sample_point, time_step): Computes the warped point at the given time step.

    Inherits from:
        MotionModel: The base class for camera and scene motion models.
    """

    def __init__(
        self,
        num_basis_coefficients: int,
        num_time_steps: int,
        device: torch.device,
        num_linear_layers: int = 8,
        skip_connections: List[int] = [4],
        hidden_dim: int = 128,
        positional_embedding_dim: int = 4,
    ) -> None:
        super().__init__(
            num_basis_coefficients=num_basis_coefficients,
            input_dim=3 + 1,
            output_dim=3,
            device=device,
            num_time_steps=num_time_steps,
            num_linear_layers=num_linear_layers,
            skip_connections=skip_connections,
            hidden_dim=hidden_dim,
            positional_embedding_dim=positional_embedding_dim,
        )

    def _compute_warped_points(
        self, position_coefficients: torch.Tensor, time_step: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the warped points at the given time step based on the position coefficients.

        Args:
            position_coefficients (torch.Tensor): The position coefficients. Shape: (batch_size, num_basis_coefficients*3)
            time_step (torch.Tensor): The time step. Shape: (batch_size, 1)

        Returns:
            torch.Tensor: The warped points at the given time step. Shape: (batch_size, 3)
        """
        (
            position_coefficient_x,
            position_coefficient_y,
            position_coefficient_z,
        ) = torch.chunk(position_coefficients, 3, dim=-1)

        basis_coefficients = torch.index_select(
            self.basis_function_matrix, 0, torch.squeeze(time_step).int()
        ).to(self.device)

        return torch.cat(
            [
                torch.sum(
                    basis_coefficients * position_coefficient_x, dim=-1, keepdim=True
                ),
                torch.sum(
                    basis_coefficients * position_coefficient_y, dim=-1, keepdim=True
                ),
                torch.sum(
                    basis_coefficients * position_coefficient_z, dim=-1, keepdim=True
                ),
            ],
            dim=-1,
        )

    def forward(
        self, sample_points: torch.Tensor, time_step: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the warped point at the given time step.

        Args:
            sample_point (torch.Tensor): The sample point. Shape: (batch_size, 3)
            time_step (torch.Tensor): The time step. Shape: (batch_size, 1)

        Returns:
            torch.Tensor: The warped point at the given time step. Shape: (batch_size, 3)
        """
        embeddings = torch.cat((sample_points, time_step), dim=-1)

        embeddings = torch.cat(
            (
                embeddings,
                torch.reshape(
                    self.positional_embedding.forward(torch.flatten(embeddings)).to(
                        self.device
                    ),
                    (embeddings.shape[0], -1),
                ),
            ),
            dim=-1,
        )
        
        hidden_state = embeddings

        for index in range(len(self.linear_layers)):
            hidden_state = self.linear_layers[index](hidden_state)
            hidden_state = F.relu(hidden_state)

            if index in self.skip_connections:
                hidden_state = torch.cat([hidden_state, embeddings], dim=-1)

        position_coefficients = self.output_layer(hidden_state)

        return self._compute_warped_points(position_coefficients, time_step)
