import torch
from torch import nn, Tensor
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, theta: int = 10000):
        """
        Positional Embedding module for adding positional information to input tensors.

        Args:
            embedding_dim (int): The dimension of the positional embedding.
            theta (int, optional): A scaling factor for the positional embedding. Defaults to 10000.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta

    def forward(self, positions: Tensor) -> Tensor:
        """
        Forward pass of the PositionalEmbedding module.

        Args:
            positions (Tensor): The input tensor representing the positions. Shape: (seq_len,)

        Returns:
            Tensor: The tensor with positional embeddings concatenated with the input positions. Shape: (seq_len, embedding_dim)
        """
        embedding = math.log(self.theta) / ((self.embedding_dim // 2) - 1)
        embedding = torch.exp(torch.arange((self.embedding_dim // 2)) * -embedding)
        embedding = positions[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        
        return embedding

