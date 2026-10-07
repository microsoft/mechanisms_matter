"""Implementation of the variational autoencoder (VAE) architecture for the CPA model."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Normal


class VariationalEncoder(nn.Module):
    """
    Encoder for perturbation model.

    The model contains two separate neural networks for perturbation indicators
    and expression data. This is crucial to get this model to work it seems.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float | None = None,
    ) -> None:
        """
        Initialize the VariationalEncoder.

        Args:
            input_dim: Dimension of the input (number of genes).
            hidden_dim: Dimension of the hidden layers.
            latent_dim: Dimension of the latent space.
            n_layers: Number of hidden layers in the encoder.
            dropout: Dropout rate for regularization (default: None).
        """
        super().__init__()  # type: ignore
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self.mu_x = nn.Linear(hidden_dim, latent_dim)
        self.var_x = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor | Distribution | Normal]:
        """Encodes the input gene expression into a latent representation."""
        x = self.network(x)

        z_mu = self.mu_x(x)
        z_var = self.var_x(x)
        dist = Normal(z_mu, torch.exp(z_var).sqrt())
        latent = dist.rsample()

        return {"dist": dist, "latent": latent}


class VariationalDecoder(nn.Module):
    """Decoder for perturbation model. Takes in the latent representation and decodes it back to gene expression space."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float | None = None,
    ) -> None:
        """Initialize the VariationalDecoder."""
        super().__init__()  # type: ignore

        layers: list[nn.Module] = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self.mu_z = nn.Linear(hidden_dim, output_dim)
        self.var_z = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decodes the latent representation back to gene expression space."""
        z = self.network(z)

        x_mu = self.mu_z(z)
        x_log_var = self.var_z(z)
        x_var = torch.exp(x_log_var)

        return {"mu": x_mu, "var": x_var}


class Decoder(nn.Module):
    """Decoder for reconstruction of outputs from latent vectors."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Initialize the Decoder.

        Args:
            latent_dim: Dimension of the latent space.
            hidden_dim: Dimension of the hidden layers.
            output_dim: Dimension of the output.
        """
        super().__init__()  # type: ignore

        # Reverse transformation from the Encoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into non-negative reconstructed outputs."""
        z = F.relu(self.bn1(self.fc1(z)))
        # z = self.dropout(z)
        z = F.relu(self.bn2(self.fc2(z)))
        # z = self.dropout(z)
        # z = F.relu(self.fc1(z))
        # z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        # Reconstruction
        prediction = F.relu(self.out(z))

        return prediction
