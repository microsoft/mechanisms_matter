# Adapted from: https://github.com/altoslabs/perturbench/
"""
BSD 3-Clause License.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import logging
from typing import Literal

import anndata
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.distributions as dist
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.sparse import issparse  # type: ignore
from scipy.stats import pearsonr  # type: ignore
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torchmetrics.functional import accuracy

from ...util.anndata_util import materialize_adata
from ..test_synthentic_data import generate_synthetic_perturbation_data
from .base import PerturbationModel
from .decoder import DeepIsotropicGaussian, DeepPoissonGamma
from .modules import AnnDataLitModule
from .utils import MLP, Batch, ensure_2d_batch_tensor
from .vae import VariationalEncoder

log = logging.getLogger(__name__)
_seed = 42
torch.manual_seed(_seed)  # type: ignore
np.random.seed(_seed)  # noqa
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_seed)


class CPA(PerturbationModel):
    """CPA module using Gaussian/NegativeBinomial/Zero-InflatedNegativeBinomial Likelihood."""

    def __init__(
        self,
        n_genes: int | None = None,
        n_perts: int | None = None,
        context: dict[str, dict[str, list[str]]] | None = None,
        n_latent: int = 128,
        library_size: Literal["learned", "observed"] | None = None,
        hidden_dim: int = 256,
        n_layers_encoder: int = 3,
        n_layers_pert_emb: int = 2,
        n_layers_covar_emb: int = 1,
        adv_classifier_hidden_dim: int = 128,
        adv_classifier_n_layers: int = 2,
        variational: bool = True,
        lr: float = 1e-3,
        wd: float = 1e-8,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        kl_weight: float = 1.0,
        adv_weight: float = 1.0,
        dropout: float = 0.1,
        penalty_weight: float = 10.0,
        adv_steps: int = 7,
        n_warmup_epochs: int = 5,
        use_adversary: bool = True,
        use_covariates: bool = True,
        softplus_output: bool = False,
        elementwise_affine: bool = False,
        use_counts: bool = False,
        datamodule: pl.LightningDataModule | None = None,
    ):
        """
        The constructor for the CPA module class.

        Args:
            n_genes: Number of genes.
            n_perts: Number of perturbations.
            context: Dictionary containing context information including covariate uniques.
            n_latent: Number of latent variables.
            library_size: Library size mode, either "learned", "observed", or None.
            hidden_dim: Hidden dimension.
            n_layers_encoder: Number of encoder layers.
            n_layers_pert_emb: Number of perturbation embedding layers.
            n_layers_covar_emb: Number of covariate embedding layers.
            adv_classifier_hidden_dim: Adversarial classifier hidden dimension.
            adv_classifier_n_layers: Number of adversarial classifier layers.
            variational: Whether to use variational autoencoder.
            lr: Learning rate.
            wd: Weight decay.
            lr_scheduler_freq: Learning rate scheduler frequency.
            lr_scheduler_interval: Learning rate scheduler interval.
            lr_scheduler_patience: Learning rate scheduler patience.
            lr_scheduler_factor: Learning rate scheduler factor.
            kl_weight: KL divergence weight.
            adv_weight: Adversarial weight.
            dropout: Dropout rate.
            penalty_weight: Penalty weight.
            adv_steps: Number of adversarial steps.
            n_warmup_epochs: Number of warmup epochs for the autoencoder.
            use_adversary: Whether to use the adversarial component.
            use_covariates: Whether to use additive covariate conditioning.
            softplus_output: Whether to apply a softplus activation to the output.
            elementwise_affine: Whether to use elementwise affine in the layer norms.
            use_counts: Whether to use raw counts.
            datamodule: Data module.
        """
        if datamodule is not None:
            n_genes = getattr(datamodule, "num_genes", n_genes)
            n_perts = getattr(datamodule, "num_perturbations", n_perts)
            context = getattr(datamodule, "train_context", context)

        if n_genes is None or n_perts is None or context is None:
            raise ValueError(
                "n_genes, n_perts, and context must be provided either directly or via datamodule"
            )

        super().__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )
        self.save_hyperparameters(ignore=["datamodule"])
        self.automatic_optimization = False
        self.n_genes = n_genes
        self.n_perts = n_perts
        self.n_input_features = n_genes
        self.n_latent = n_latent
        self.variational = variational
        self.hidden_dim = hidden_dim
        self.n_layers_pert_emb = n_layers_pert_emb
        self.n_layers_covar_emb = n_layers_covar_emb
        self.n_layers_encoder = n_layers_encoder
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.penalty_weight = penalty_weight
        self.adv_classifier_hidden_dim = adv_classifier_hidden_dim
        self.adv_classifier_n_layers = adv_classifier_n_layers
        self.dropout = dropout
        self.adv_steps = adv_steps
        self.n_warmup_epochs = n_warmup_epochs
        self.softplus_output = softplus_output
        self.adv_loss_drugs = nn.CrossEntropyLoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()
        self.use_adversary = use_adversary
        self.use_covariates = use_covariates
        self.use_counts = use_counts

        self.encoder = VariationalEncoder(
            input_dim=self.n_input_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.n_latent,
            n_layers=self.n_layers_encoder,
            dropout=self.dropout,
        )

        if self.use_covariates:
            self.covars_encoder = {
                covar: uniques
                for covar, uniques in context["covariate_uniques"].items()
                if len(uniques) > 1
            }
        else:
            self.covars_encoder = {}

        if not self.use_counts:
            self.decoder = DeepIsotropicGaussian(
                input_dim=self.n_latent,
                hidden_dim=self.hidden_dim,
                output_dim=self.n_genes,
                n_layers=self.n_layers_encoder,
                dropout=self.dropout,
                softplus_output=self.softplus_output,
            )
        else:
            self.decoder = DeepPoissonGamma(
                input_dim=self.n_latent,
                hidden_dim=self.hidden_dim,
                output_dim=self.n_genes,
                n_layers=self.n_layers_encoder,
                dropout=self.dropout,
                library_size=library_size,
                use_legacy_negative_binomial=False,
            )

        self.pert_network = MLP(
            input_dim=self.n_perts,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_latent,
            n_layers=self.n_layers_pert_emb,
            dropout=self.dropout,
            elementwise_affine=elementwise_affine,
        )

        self.perturbation_adversary_classifier = MLP(
            self.n_latent,
            self.adv_classifier_hidden_dim,
            self.n_perts,
            self.adv_classifier_n_layers,
            self.dropout,
            elementwise_affine=elementwise_affine,
        )

        if self.use_covariates:
            self.covars_embeddings = nn.ModuleDict(
                {
                    key: MLP(
                        input_dim=len(unique_covars),
                        output_dim=n_latent,
                        hidden_dim=hidden_dim,
                        n_layers=self.n_layers_covar_emb,
                        dropout=dropout,
                        elementwise_affine=elementwise_affine,
                    )
                    for key, unique_covars in self.covars_encoder.items()
                    if len(unique_covars) > 1
                }
            )

            self.covars_adversary_classifiers: nn.ModuleDict = nn.ModuleDict(
                {
                    covar: MLP(
                        self.n_latent,
                        self.adv_classifier_hidden_dim,
                        len(unique_covars),
                        self.adv_classifier_n_layers,
                        self.dropout,
                        elementwise_affine=elementwise_affine,
                    )
                    for covar, unique_covars in self.covars_encoder.items()
                }
            )

        else:
            self.covars_embeddings = nn.ModuleDict()
            self.covars_adversary_classifiers = nn.ModuleDict()

        self.generative_modules = nn.ModuleList(
            [self.encoder, self.decoder, self.pert_network, self.covars_embeddings]
        )
        self.adversary_modules = nn.ModuleList(
            [self.perturbation_adversary_classifier, self.covars_adversary_classifiers]
        )

    @property
    def start_adv_training(self):
        """
        Determine whether to start adversarial training.

        Returns True after the warmup epochs have completed, or immediately if no warmup epochs are set.

        Returns:
        -------
        bool
            Whether adversarial training should begin based on the current epoch and warmup settings.
        """
        if self.n_warmup_epochs:
            return self.current_epoch > self.n_warmup_epochs
        else:
            return True

    def unpack_batch(  # type: ignore[override]
        self, batch: Batch
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], None]:
        """
        Unpack the batch into its components.

        Args:
            batch: Batch object containing gene expression, perturbations, covariates, and optionally embeddings.

        Returns:
            A tuple containing the unpacked components of the batch:
            - embeddings: Optional tensor of cell embeddings.
            - x: Tensor of gene expression data.
            - perts: Tensor of perturbation data.
            - covars_dict: Dictionary of covariate tensors.
            - None: Placeholder for compatibility with base class.
        """
        raw_embeddings: torch.Tensor | None = getattr(batch, "embeddings", None)
        embeddings = ensure_2d_batch_tensor(raw_embeddings)

        # Use controls as encoder input when available (CPA objective)
        controls = getattr(batch, "controls", None)
        if controls is not None:
            # print("DEBUG: Using control expression as encoder input")
            x = ensure_2d_batch_tensor(controls)
        else:
            # print("DEBUG: Using perturbed expression as encoder input")
            x = ensure_2d_batch_tensor(batch.gene_expression)  # type: ignore[attr-defined]
        perts = ensure_2d_batch_tensor(batch.perturbations)  # type: ignore[attr-defined]
        covars_dict: dict[str, torch.Tensor] = batch.covariates  # type: ignore[attr-defined]

        return embeddings, x, perts, covars_dict, None

    def inference(
        self,
        x: torch.Tensor,
        perts: torch.Tensor,
        covars_dict: dict[str, torch.Tensor],
        embeddings: torch.Tensor | None = None,
        n_samples: int = 1,
        covars_to_add: list[str] | None = None,
    ):
        """
        Performs the inference step of the model, encoding the input data and applying perturbation and covariate effects.

        Args:
            x: Tensor of gene expression data.
            perts: Tensor of perturbation data.
            covars_dict: Dictionary of covariate tensors.
            embeddings: Optional tensor of cell embeddings to use instead of gene expression for encoding.
            n_samples: Number of samples to draw from the latent distribution (only applicable if variational is True).
            covars_to_add: List of covariate keys to include in the inference step. If None, all covariates in covars_dict will be included.

        Returns:
            A dictionary containing the results of the inference step, including:
            - z: The combined latent representation after adding perturbation and covariate effects.
            - z_no_pert: The latent representation without perturbation effects (only basal and covariate effects).
            - z_basal: The basal latent representation from the encoder.
            - z_covs: The combined covariate effects in the latent space.
            - z_pert: The perturbation effect in the latent space.
            - library: The library size if applicable (only for certain decoder configurations).
            - qz: The latent distribution (only if variational is True).
        """
        if embeddings is not None:
            x_ = embeddings
            library = None
        else:
            x_ = x
            library = x.sum(dim=1)  ## observed library size (total counts per cell)

        if self.variational:
            enc_out = self.encoder(x_)
            qz = enc_out["dist"]
            z_basal = enc_out["latent"]
        else:
            enc_out = self.encoder(x_)
            qz = None
            z_basal = enc_out["latent"]

        if self.variational and n_samples > 1 and qz is not None:
            sampled_z = qz.sample((n_samples,))
            z_basal = sampled_z.mean(dim=0)

        z_pert_true: torch.Tensor = self.pert_network(perts)  # perturbation encoder
        z_pert = z_pert_true
        z_covs_dict: dict[str, torch.Tensor] = {}

        if covars_to_add is None:
            covars_to_add = list(self.covars_encoder.keys())
        for covar in self.covars_encoder:
            if covar in covars_to_add:
                covars_input = covars_dict[covar]
                z_cov = self.covars_embeddings[covar](covars_input)
                z_covs_dict[covar] = z_cov

        if len(z_covs_dict) > 0:
            z_covs = torch.stack(list(z_covs_dict.values()), dim=0).sum(dim=0)
        else:
            z_covs = torch.zeros_like(z_basal)
        z = z_basal + z_pert + z_covs
        z_no_pert = z_basal + z_covs

        return dict(
            z=z,
            z_no_pert=z_no_pert,
            z_basal=z_basal,
            z_covs=z_covs,
            z_pert=z_pert.sum(dim=1),
            library=library,
            qz=qz,
        )

    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dist.Distribution | Normal]:
        """
        Performs the generative step of the model, decoding the latent representation to reconstruct gene expression.

        Args:
            z: Tensor of latent representations.
            library: Optional tensor of library sizes.

        Returns:
            A dictionary containing the results of the generative step, including:
            - predictions: The reconstructed gene expression.
            - pz: The prior distribution over the latent space.
        """
        if self.use_counts:
            predictions: dist.Distribution = self.decoder(
                z, library_size=library
            )  # The decoder returns a distribution object, not a tensor
        else:
            predictions: dist.Distribution = self.decoder(
                z
            )  # The decoder returns a distribution object, not a tensor
        return {
            "predictions": predictions,
            "pz": Normal(torch.zeros_like(z), torch.ones_like(z)),
        }

    def loss(
        self,
        x: torch.Tensor,
        perturbations: torch.Tensor,
        covariates: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | dist.Distribution | Normal],
        generative_outputs: dict[str, torch.Tensor | dist.Distribution | Normal],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Computes the reconstruction loss (AE) or the ELBO (VAE)."""
        recon_loss: torch.Tensor = self.decoder.reconstruction_loss(
            generative_outputs["predictions"],  # type: ignore[arg-type]  # predictions is always a Distribution from the generative step
            x,
        )

        if self.variational:
            qz = inference_outputs["qz"]
            pz = generative_outputs["pz"]  # just a standard gaussian

            kl_divergence_z = kl(qz, pz).sum(dim=1)  # type: ignore[assignment]  # shape (batch_size,)
            kl_loss = kl_divergence_z.mean()
        else:
            kl_loss = torch.zeros_like(recon_loss)

        if self.use_adversary:
            adv_loss = self.adversarial_loss(
                perturbations,
                covariates,
                inference_outputs["z_basal"],  # type: ignore[union-attr]
                self.training,  # compute penalty only during training
            )
        else:
            adv_loss = {
                "adv_loss": torch.zeros_like(recon_loss),
                "penalty_adv": torch.zeros_like(recon_loss),
                "penalty_covars": torch.zeros_like(recon_loss),
                "penalty_perts": torch.zeros_like(recon_loss),
                "acc_perts": torch.zeros_like(recon_loss),
                "covariate_classfier_loss": torch.zeros_like(recon_loss),
                "perturbation_classifier_loss": torch.zeros_like(recon_loss),
            }

        total_loss = recon_loss + self.kl_weight * kl_loss - self.adv_weight * adv_loss["adv_loss"]

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "adv_loss": adv_loss["adv_loss"],
            "covariate_classfier_loss": adv_loss["covariate_classfier_loss"],
            "perturbation_classifier_loss": adv_loss["perturbation_classifier_loss"],
            "penalty_adv": adv_loss["penalty_adv"],
            "penalty_covars": adv_loss["penalty_covars"],
            "penalty_perts": adv_loss["penalty_perts"],
            "acc_perts": adv_loss["acc_perts"],
        }

    def adversarial_loss(
        self,
        perturbations: torch.Tensor,
        covariates: dict[str, torch.Tensor],
        z_basal: torch.Tensor,
        compute_penalty: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Computes adversarial classification losses and regularizations."""
        if compute_penalty:
            z_basal = z_basal.requires_grad_(True)

        covars_pred_logits: dict[str, torch.Tensor | None] = {}
        for covar in self.covars_encoder.keys():
            if covar in self.covars_adversary_classifiers:
                covars_pred_logits[covar] = self.covars_adversary_classifiers[covar](z_basal)
            else:
                covars_pred_logits[covar] = None

        adv_results: dict[str, torch.Tensor] = {}

        # Classification losses for different covariates
        for covar, covars in self.covars_encoder.items():
            adv_results[f"adv_{covar}"] = (
                self.adv_loss_fn(  # ∞ we've removed mixup for now
                    covars_pred_logits[covar],
                    covariates[covar],
                )
                if covars_pred_logits[covar] is not None
                else torch.as_tensor(0.0).to(self.device)
            )

            if covars_pred_logits[covar] is not None:
                preds_argmax: torch.Tensor = covars_pred_logits[covar].argmax(1)  # type: ignore[assignment]  # shape (batch_size,)
                targets_argmax: torch.Tensor = covariates[covar].argmax(1)
                adv_results[f"acc_{covar}"] = accuracy(
                    preds_argmax,
                    targets_argmax,
                    task="multiclass",
                    num_classes=len(covars),
                )
            else:
                adv_results[f"acc_{covar}"] = torch.as_tensor(0.0).to(self.device)

        if len(self.covars_encoder) > 0:
            adv_results["covariate_classfier_loss"] = sum(
                [adv_results[f"adv_{key}"] for key in self.covars_encoder.keys()],
                torch.as_tensor(0.0).to(self.device),
            )
        else:
            adv_results["covariate_classfier_loss"] = torch.as_tensor(0.0).to(self.device)

        # TODO Not using mixups for now.
        perturbations_pred_logits = self.perturbation_adversary_classifier(z_basal)

        adv_results["perturbation_classifier_loss"] = self.adv_loss_drugs(
            perturbations_pred_logits, perturbations
        )

        adv_results["acc_perts"] = accuracy(
            perturbations_pred_logits.argmax(1),
            perturbations.argmax(1),
            average="macro",
            num_classes=self.n_perts,
            task="multiclass",
        )

        adv_results["adv_loss"] = (
            adv_results["covariate_classfier_loss"] + adv_results["perturbation_classifier_loss"]
        )

        if compute_penalty:
            # Penalty losses
            for covar in self.covars_encoder.keys():
                if covars_pred_logits[covar] is not None:
                    adv_results[f"penalty_{covar}"] = (
                        torch.autograd.grad(
                            covars_pred_logits[covar].sum(),  # type: ignore[call-arg]  # sum over batch and classes
                            z_basal,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        .pow(2)
                        .mean()
                    )
                else:
                    adv_results[f"penalty_{covar}"] = torch.as_tensor(0.0).to(self.device)

            if len(self.covars_encoder) > 0:
                adv_results["penalty_covars"] = sum(
                    [adv_results[f"penalty_{covar}"] for covar in self.covars_encoder.keys()],
                    torch.as_tensor(0.0).to(self.device),
                )
            else:
                adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)

            adv_results["penalty_perts"] = (
                torch.autograd.grad(
                    perturbations_pred_logits.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                .pow(2)
                .mean()
            )

            adv_results["penalty_adv"] = (
                adv_results["penalty_perts"] + adv_results["penalty_covars"]
            )
        else:
            for covar in self.covars_encoder.keys():
                adv_results[f"penalty_{covar}"] = torch.as_tensor(0.0).to(self.device)

            adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)
            adv_results["penalty_perts"] = torch.as_tensor(0.0).to(self.device)
            adv_results["penalty_adv"] = torch.as_tensor(0.0).to(self.device)

        return adv_results

    def _get_dict_if_none(self, param: dict[str, object] | None) -> dict[str, object]:
        param = {} if not isinstance(param, dict) else param

        return param

    def forward(
        self,
        batch: Batch,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | dist.Distribution | Normal]]:
        """Defines the forward pass of the model, combining the inference and generative steps."""
        embeddings, x, perts, covars_dict, _ = self.unpack_batch(batch)
        inference_outputs = self.inference(x, perts, covars_dict, embeddings)

        generative_outputs = self.generative(inference_outputs["z"], inference_outputs["library"])
        return inference_outputs, generative_outputs

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        """Defines the training step, including loss computation and optimization logic."""
        optimizer_generative, optimizer_adversary = self.optimizers()  # type: ignore[assignment]  # get the optimizers for the generative and adversarial components

        inference_outputs, generative_outputs = self.forward(batch)
        losses = self.loss(
            batch.gene_expression,  # type: ignore[attr-defined]  # batch_size, n_genes
            batch.perturbations,  # type: ignore[attr-defined]  # batch_size, n_perts
            batch.covariates,  # type: ignore[attr-defined]  # dict of covariate tensors
            inference_outputs,  # type: ignore[dict-item]  # dict of tensors from the inference step
            generative_outputs,  # dict of tensors and distributions from the generative step
            batch_idx,
        )

        total_loss = losses["total_loss"]
        adv_loss = (
            self.adv_weight * losses["adv_loss"] + self.penalty_weight * losses["penalty_adv"]
        )

        if self.start_adv_training:
            if batch_idx % self.adv_steps == 0:
                self.toggle_optimizer(optimizer_generative)  # type: ignore[union-attr]  # toggle the generative optimizer to update its parameters
                optimizer_generative.zero_grad()  # type: ignore[union-attr]  # zero the gradients for the generative optimizer
                self.manual_backward(total_loss)
                optimizer_generative.step()  # type: ignore[union-attr]  # update the generative parameters
                self.untoggle_optimizer(optimizer_generative)  # type: ignore[union-attr]  # untoggle the generative optimizer to allow the adversarial optimizer to update its parameters in the next step

            else:
                self.toggle_optimizer(optimizer_adversary)  # type: ignore[union-attr]  # toggle the adversarial optimizer to update its parameters
                optimizer_adversary.zero_grad()  # type: ignore[union-attr]  # zero the gradients for the adversarial optimizer
                self.manual_backward(adv_loss)
                optimizer_adversary.step()  # type: ignore[union-attr]  # update the adversarial parameters
                self.untoggle_optimizer(optimizer_adversary)  # type: ignore[union-attr]  # untoggle the adversarial optimizer

        else:
            gen_loss = losses["recon_loss"] + self.kl_weight * losses["kl_loss"]
            self.toggle_optimizer(optimizer_generative)  # type: ignore[union-attr]  # toggle the generative optimizer to update its parameters during the warmup phase
            optimizer_generative.zero_grad()  # type: ignore[union-attr]  # zero the gradients for the generative optimizer
            self.manual_backward(gen_loss)
            optimizer_generative.step()  # type: ignore[union-attr]  # update the generative parameters during the warmup phase
            self.untoggle_optimizer(optimizer_generative)  # type: ignore[union-attr]  # untoggle the generative optimizer

        if self.training:
            for key, value in losses.items():
                self.log(
                    "train_" + key,
                    value,
                    prog_bar=True,
                    logger=True,
                    batch_size=len(batch),
                )

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Defines the validation step, computing the loss for the validation set."""
        inference_outputs, generative_outputs = self.forward(batch)
        losses = self.loss(
            batch.gene_expression,  # type: ignore[attr-defined]  # batch_size, n_genes
            batch.perturbations,  # type: ignore[attr-defined]  # batch_size, n_perts
            batch.covariates,  # type: ignore[attr-defined]  # dict of covariate tensors
            inference_outputs,  # type: ignore[dict-item]  # dict of tensors from the inference step
            generative_outputs,  #   # dict of tensors and distributions from the generative step
            batch_idx,
        )
        total_loss = losses["total_loss"]
        self.log("val_loss", total_loss, prog_bar=True, logger=True, batch_size=len(batch))
        return total_loss

    def predict(self, counterfactual_batch: Batch) -> torch.Tensor:
        """Defines the prediction step, returning the reconstructed gene expression."""
        _, generative_outputs = self.forward(counterfactual_batch)
        return generative_outputs["predictions"]  # type: ignore[return-value]  # return the distribution object from the generative outputs

    def configure_optimizers(self):  # type: ignore[override]  # the base class has a different return type, but this is compatible with PyTorch Lightning's expected return type for optimizers and schedulers
        """Configures the optimizers for the generative and adversarial components, as well as their learning rate schedulers."""
        optimizer_generative = torch.optim.Adam(
            self.generative_modules.parameters(), lr=self.lr, weight_decay=self.wd
        )
        optimizer_adversary = torch.optim.Adam(
            self.adversary_modules.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler_generative = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_generative,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,  # type: ignore[union-attr]  # access the learning rate scheduler parameters from the class attributes
        )
        scheduler_adversary = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_adversary,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,  # type: ignore[union-attr]  # access the learning rate scheduler parameters from the class attributes
        )
        lr_scheduler_generative = {
            "scheduler": scheduler_generative,
            "monitor": self.lr_monitor_key,
            "frequency": self.lr_scheduler_freq,
            "interval": self.lr_scheduler_interval,
        }
        lr_scheduler_adversary = {
            "scheduler": scheduler_adversary,
            "monitor": self.lr_monitor_key,
            "frequency": self.lr_scheduler_freq,
            "interval": self.lr_scheduler_interval,
        }

        return [
            {
                "optimizer": optimizer_generative,
                "lr_scheduler": lr_scheduler_generative,
            },
            {"optimizer": optimizer_adversary, "lr_scheduler": lr_scheduler_adversary},
        ]


def run_cpa(
    train_adata: sc.AnnData,
    valid_adata: sc.AnnData,
    test_adata: sc.AnnData,
    add_controls: bool,
    perturbation_column: str,
    control_label: str,
    use_counts: bool,
    epochs: int,
    model_dir: str,
    dataset_name: str,
    expression_layer: str | None = None,
    covariate_keys: list[str] | None = None,
    library_size: Literal["learned", "observed"] | None = None,
) -> dict[str, np.ndarray]:
    """Train CPA and return predictions in the same expression space used as input."""
    train_adata = materialize_adata(train_adata, expression_layer=expression_layer)
    valid_adata = materialize_adata(valid_adata, expression_layer=expression_layer)
    test_adata = materialize_adata(test_adata, expression_layer=expression_layer)
    train_adata.obs[perturbation_column] = train_adata.obs[perturbation_column].astype("category")  # type: ignore[assignment]
    valid_adata.obs[perturbation_column] = valid_adata.obs[perturbation_column].astype("category")  # type: ignore[assignment]
    test_adata.obs[perturbation_column] = test_adata.obs[perturbation_column].astype("category")  # type: ignore[assignment]
    for covariate_key in covariate_keys or []:
        train_adata.obs[covariate_key] = train_adata.obs[covariate_key].astype(str)  # type: ignore[assignment]
        valid_adata.obs[covariate_key] = valid_adata.obs[covariate_key].astype(str)  # type: ignore[assignment]
        test_adata.obs[covariate_key] = test_adata.obs[covariate_key].astype(str)  # type: ignore[assignment]

    datamodule = AnnDataLitModule(
        train_adata=train_adata,
        val_adata=valid_adata,  # or None
        test_adata=test_adata,  # or None
        add_controls=add_controls,  # True = use SingleCellPerturbationWithControls
        perturbation_key=perturbation_column,  # column in .obs with perturbation labels
        perturbation_control_value=control_label,  # label for unperturbed cells
        batch_size=128,
        perturbation_combination_delimiter="+",  # for combo perturbations like "geneA+geneB"
        covariate_keys=covariate_keys,
        use_counts=use_counts,  # True if .X has raw counts, False if log-normalized
    )

    model = CPA(datamodule=datamodule, use_counts=use_counts, library_size=library_size)
    print("Model initialized. Starting training...")
    # print("Model:", model)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=model_dir, name=dataset_name),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in datamodule.test_dataloader():  # type: ignore
            embeddings, x, perts, covars_dict, _ = model.unpack_batch(batch)
            inference_out = model.inference(x, perts, covars_dict, embeddings)
            gen_out = model.generative(inference_out["z"], inference_out["library"])

            predictions = gen_out["predictions"]  # Tensor for Gaussian, Distribution for NB
            target = batch.gene_expression  # already transformed to dense tensor

            if isinstance(predictions, torch.Tensor):
                pred_np = predictions.cpu().numpy()
            else:
                pred_np = predictions.mean.cpu().numpy()  # NB distribution → expected value

            all_predictions.append(pred_np)  # type: ignore
            all_targets.append(target.cpu().numpy())  # type: ignore

    preds = np.concatenate(all_predictions, axis=0)  # type: ignore
    targets = np.concatenate(all_targets, axis=0)  # type: ignore[arg-type]

    if add_controls:
        control_mask: np.ndarray = np.asarray(test_adata.obs[perturbation_column]) == control_label
        non_control_mask: np.ndarray = ~control_mask

        if issparse(test_adata.X):  # type: ignore[arg-type]
            test_matrix: np.ndarray = test_adata.X.toarray()  # type: ignore[union-attr]
        else:
            test_matrix = np.asarray(test_adata.X)  # type: ignore[arg-type]

        full_preds: np.ndarray = np.array(test_matrix, dtype=np.float32, copy=True)
        full_targets: np.ndarray = np.array(test_matrix, dtype=np.float32, copy=True)
        full_preds[non_control_mask] = preds
        full_targets[non_control_mask] = targets
        preds = full_preds
        targets = full_targets

    return {
        "preds": preds,
        "target": targets,
    }


if __name__ == "__main__":
    print("Training CPA model...")
    covariate_keys = ["cell_type"]
    is_synthetic = True
    adata = anndata.AnnData()
    counts_layer = "counts"
    control_label = "control"
    perturbation_column = "perturbation"
    model_dir = "/workspaces/immunorep/immunorep-scrnaseq/data/cpa/"
    add_controls = True  # NOTE for counterfactual evaluation, we want to keep controls in the dataset and use them as input to the encoder (CPA objective), so set add_controls=True to ensure controls are included in the dataloader and passed to the model
    use_counts = (
        False  # False keeps CPA in the same normalized/log1p space used by GEARS in this repo
    )
    # The reconstruction loss follows the chosen data representation:
    # Gaussian for normalized inputs, Negative Binomial for count inputs.
    if use_counts:
        library_size = "learned"  # use observed/learned library size (total counts per cell) for the Poisson-Gamma decoder when use_counts is True
    else:
        library_size = None  # not applicable when not using counts
    epochs = 100
    if is_synthetic:
        dataset_name = "synthetic"
        adata = generate_synthetic_perturbation_data(
            perturbation_column=perturbation_column,
            control_label=control_label,
            context_key="cell_type",
        )
        adata.layers["counts"] = adata.X.copy()  # type: ignore
    else:
        # Load Norman19 data
        dataset_name = "norman19"
        adata = sc.read_h5ad(
            "/workspaces/immunorep/immunorep-scrnaseq/data/norman19/norman19_processed.h5ad"
        )
        print("Original Norman adata shape:", adata.shape)
        indices = np.random.choice(adata.n_obs, size=1000, replace=False)  # noqa
        print("Subsetting to 1000 random cells for testing...")
        adata = adata[indices].copy()
        adata.obs["cell_type"] = np.array(["K562"] * adata.n_obs)

    if issparse(adata.X):  # type: ignore
        data = adata.X.data  # type: ignore
        is_all_integer = np.allclose(data, np.round(data))  # type: ignore
    else:
        is_all_integer = np.allclose(adata.X, np.round(adata.X))  # type: ignore

    if use_counts and not is_all_integer:
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"]
            print("Using 'counts' layer as raw counts.")
        else:
            raise ValueError(
                "use_counts=True but adata.X contains non-integer values and no 'counts' layer exists. Provide raw counts in adata.X or adata.layers['counts']."
            )
    elif not use_counts and is_all_integer:
        print("Data appears to be raw counts, applying normalization and log transformation...")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    elif use_counts and is_all_integer:
        print("Data contains raw counts, proceeding without modification...")
    else:
        print("Data appears to be normalized, proceeding without modification...")

    # Split adata into train (60%), valid (20%), test (20%)
    if is_synthetic:
        n = adata.n_obs
        indices = np.random.permutation(n)  # noqa
        train_end = int(0.6 * n)
        valid_end = int(0.8 * n)
        train_adata = adata[indices[:train_end]].copy()
        valid_adata = adata[indices[train_end:valid_end]].copy()
        test_adata = adata[indices[valid_end:]].copy()
    else:
        # need to split by perturbation condition to ensure all conditions are represented in train/val/test for real data (this is within-context)
        perturbations = adata.obs[perturbation_column].unique()
        train_adata_list = []
        valid_adata_list = []
        test_adata_list = []
        for pert in perturbations:
            pert_adata = adata[adata.obs[perturbation_column] == pert].copy()
            n = pert_adata.n_obs
            if n < 3:
                # If too few cells for this perturbation, put all in train
                train_adata_list.append(pert_adata)  # type: ignore
                continue
            indices = np.random.permutation(n)  # noqa
            train_end = int(0.6 * n)
            valid_end = int(0.8 * n)
            train_adata_list.append(pert_adata[indices[:train_end]])  # type: ignore
            valid_adata_list.append(pert_adata[indices[train_end:valid_end]])  # type: ignore
            test_adata_list.append(pert_adata[indices[valid_end:]])  # type: ignore
        train_adata = anndata.concat(train_adata_list)  # type: ignore
        valid_adata = anndata.concat(valid_adata_list)  # type: ignore
        test_adata = anndata.concat(test_adata_list)  # type: ignore
    print(f"Split: train={train_adata.n_obs}, valid={valid_adata.n_obs}, test={test_adata.n_obs}")
    # Ensure perturbation column is categorical in all splits
    for ad in [train_adata, valid_adata, test_adata]:
        ad.obs[perturbation_column] = ad.obs[perturbation_column].astype("category")  # type: ignore

    results = run_cpa(
        train_adata=train_adata,
        valid_adata=valid_adata,
        test_adata=test_adata,
        add_controls=add_controls,
        perturbation_column=perturbation_column,
        control_label=control_label,
        use_counts=use_counts,
        epochs=epochs,
        model_dir=model_dir,
        dataset_name=dataset_name,
        library_size=library_size,
        covariate_keys=covariate_keys,
    )

    orig = results["target"]
    recon = results["preds"]

    # Evaluate only on perturbed cells (controls are identity, not predictions)
    is_perturbed: np.ndarray = np.asarray(test_adata.obs[perturbation_column]) != control_label
    print(
        f"Evaluating on {is_perturbed.sum()} perturbed cells out of {len(test_adata)} total cells."
    )
    orig_mean = orig[is_perturbed].mean(axis=0)
    recon_mean = recon[is_perturbed].mean(axis=0)

    r, _ = pearsonr(
        orig_mean.A1 if hasattr(orig_mean, "A1") else orig_mean,
        recon_mean.A1 if hasattr(recon_mean, "A1") else recon_mean,
    )
    print(f"Pearson correlation between original and reconstructed mean expression: {r:.3f}")

    plt.scatter(orig_mean, recon_mean, alpha=0.3, s=5)  # type: ignore
    plt.xlabel("Original mean expression")  # type: ignore
    plt.ylabel("Reconstructed mean expression")  # type: ignore
    plt.title(f"Per-gene mean expression (Pearson r={r:.3f})")  # type: ignore
    plt.plot([0, orig_mean.max()], [0, orig_mean.max()], "r--")  # type: ignore
    # Diagonal line matching axis limits)
    plt.savefig(model_dir + "CPA_gene_correlation.png")  # type: ignore
    # plt.show()
    print("Done!")
