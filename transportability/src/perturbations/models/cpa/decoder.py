"""
Implements the decoder of the CPA model, which reconstructs gene expression from the latent representation.

The decoder is a deep Poisson-Gamma model, which models gene expression as a negative binomial distribution with mean given by a neural network and dispersion given by a learnable parameter.
The decoder also models the library size, which can be either observed or learned. The decoder also implements the reconstruction loss, which is the negative log likelihood of the target under the predicted distribution.
"""

from typing import Literal

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import MLP
from .vae import Decoder


class DeepIsotropicGaussian(Decoder):
    """Decoder for the CPA model, which models gene expression as a Gaussian distribution with mean given by a neural network and fixed variance."""

    # zyan: todo, isotropic means the covariance matrix is diagonal. Here, it means not learning variance at all.
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        softplus_output: bool = False,
    ) -> None:
        """Initializes the decoder."""
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim, latent_dim=input_dim)
        self.network = MLP(input_dim, hidden_dim, output_dim, n_layers, dropout)
        self.softplus_output = softplus_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Decodes the latent representation to reconstruct gene expression."""
        # Forward pass through the network to get the predicted values
        if self.softplus_output:
            predictions = F.softplus(self.network(x))
        else:
            predictions = self.network(x)
        return predictions

    @staticmethod
    def reconstruction_loss(
        predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ):
        """Computes the mean squared error between the predictions and the target."""
        if reduction == "mean":
            return F.mse_loss(predictions, target, reduction="none").sum(-1).mean()
        elif reduction == "none":
            return F.mse_loss(predictions, target, reduction="none").sum(-1)
        else:
            raise ValueError("Reduction argument only accepts 'mean' or 'none'")


class DeepPoissonGamma(Decoder):
    """Decoder for the CPA model, which models gene expression as a negative binomial distribution with mean given by a neural network and dispersion given by a learnable parameter."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        library_size: Literal["observed", "learned"] | None = "observed",
        use_legacy_negative_binomial: bool = False,
    ) -> None:
        """
        Initializes the decoder.

        Args:
            input_dim: Dimension of the input latent representation.
            hidden_dim: Dimension of the hidden layers.
            output_dim: Dimension of the output gene expression.
            n_layers: Number of hidden layers.
            dropout: Dropout rate for the hidden layers.
            library_size: Whether to use observed or learned library size. If "observed", the library size must be provided as an argument to the forward method. If "learned", the library size is learned by the model. If None, the library size is not used and the mean of the negative binomial is directly given by the neural network output.
            use_legacy_negative_binomial: Whether to use the legacy parameterization of the negative binomial distribution, which uses the "concentration" and "logits" parameters. If False, the new parameterization is used, which uses the "mean" and "concentration" parameters. The new parameterization is more numerically stable, but the legacy parameterization is kept for compatibility with older versions of the model.
        """
        super().__init__(latent_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        self.library_size = library_size
        self.rho = MLP(input_dim, hidden_dim, output_dim, n_layers, dropout)
        self.log_theta = nn.Parameter(
            torch.randn(output_dim) * 1 / torch.sqrt(torch.tensor(output_dim))
        )

        if self.library_size == "learned":
            self.library_size_net = MLP(input_dim, hidden_dim, 1, n_layers, dropout)

        self.use_legacy_negative_binomial = use_legacy_negative_binomial

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, library_size: torch.Tensor | None = None
    ) -> dist.Distribution:
        """Decodes the latent representation to reconstruct gene expression."""
        rho = self.rho(x)
        rho = torch.softmax(rho, dim=-1)
        if self.library_size == "observed" and library_size is None:
            raise ValueError("Library size must be provided if library size is observed")
        elif self.library_size == "observed" and library_size is not None:
            lib = library_size.reshape(-1, 1)
        elif self.library_size == "learned":
            lib = F.relu(self.library_size_net(x)).clip(min=1e-5).reshape(-1, 1) + 1_000
        else:
            raise ValueError("Missing library_size argument")

        if self.use_legacy_negative_binomial:
            concentration = lib.expand(*rho.shape).clip(min=1e-5) * rho
            return dist.negative_binomial.NegativeBinomial(
                concentration,
                logits=self.log_theta.expand(x.shape[0], self.log_theta.shape[0]),
                validate_args=False,
            )
        else:
            mean = (lib.expand(*rho.shape) * rho).clip(min=1e-5)
            concentration = torch.exp(self.log_theta).unsqueeze(0).repeat(x.shape[0], 1)
            return dist.negative_binomial.NegativeBinomial(
                concentration,
                logits=torch.log(mean / (concentration + 1e-5)),
                validate_args=False,
            )

    @staticmethod
    def reconstruction_loss(
        predictions: dist.Distribution, target: torch.Tensor, reduction: str = "mean"
    ):
        """Computes the negative log likelihood of the target under the predicted distribution."""
        if reduction == "mean":
            return -predictions.log_prob(target).sum(-1).mean()
        elif reduction == "none":
            return -predictions.log_prob(target).sum(-1)
        else:
            raise ValueError("Reduction argument only accepts 'mean' or 'none'")
