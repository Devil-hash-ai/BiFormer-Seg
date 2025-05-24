import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """
    Standard MLP block with configurable hidden size and dropout.
    Optionally uses residual connection.
    """
    def __init__(self, input_dim: int, hidden_dim: int = None, output_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class ResidualMLP(nn.Module):
    """
    MLP block with residual path.
    Useful for refining token or fusion features.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, hidden_dim=dim * 4, output_dim=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))