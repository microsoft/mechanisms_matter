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

from abc import ABC, abstractmethod
from typing import Any

import anndata as ad
import lightning as L
import pandas as pd
import torch

from .utils import Batch, ensure_2d_batch_tensor


class PerturbationModel(L.LightningModule, ABC):
    """
    A base model class for perturbation prediction models.

    Attributes:
        training_record: A record of the transforms and training context used
    """

    prediction_output_path: str | None = None

    def __init__(
        self,
        datamodule: L.LightningDataModule | None = None,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: float | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: float | None = None,
        lr_scheduler_factor: float | None = None,
        lr_monitor_key: str | None = None,
    ):
        """
        Initializes the perturbation model.

        Args:
            datamodule: The LightningDataModule containing the training data and context.
            lr: Learning rate for the optimizer.
            wd: Weight decay for the optimizer.
            lr_scheduler_freq: Frequency for the learning rate scheduler.
            lr_scheduler_interval: Interval for the learning rate scheduler (e.g. 'epoch' or 'step').
            lr_scheduler_patience: Patience for the learning rate scheduler.
            lr_scheduler_factor: Factor for the learning rate scheduler.
            lr_monitor_key: Metric key to monitor for the learning rate scheduler.
        """
        super().__init__()

        self.training_record: dict[str, Any] = {
            "transform": None,
            "train_context": None,
            "n_total_covs": None,
        }

        self.lr = 1e-3 if lr is None else lr
        self.wd = 1e-5 if wd is None else wd
        self.lr_scheduler_freq = 1 if lr_scheduler_freq is None else lr_scheduler_freq
        self.lr_scheduler_interval = (
            "epoch" if lr_scheduler_interval is None else lr_scheduler_interval
        )
        self.lr_scheduler_patience = 15 if lr_scheduler_patience is None else lr_scheduler_patience
        self.lr_scheduler_factor = 0.2 if lr_scheduler_factor is None else lr_scheduler_factor
        self.lr_monitor_key = "val_loss" if lr_monitor_key is None else lr_monitor_key

        if datamodule is not None:
            self.training_record["transform"] = datamodule.train_dataset.transform  # type: ignore[attr-defined]
            self.training_record["train_context"] = datamodule.train_context  # type: ignore[attr-defined]

            self.n_genes = datamodule.num_genes  # type: ignore[attr-defined]
            self.n_perts = datamodule.num_perturbations  # type: ignore[attr-defined]
            self.n_covs = datamodule.num_covariates  # type: ignore[attr-defined]

            embedding_width = datamodule.embedding_width  # type: ignore[attr-defined]
            if embedding_width is not None:
                self.n_input_features = embedding_width  # type: ignore[attr-defined]
            else:
                self.n_input_features = self.n_genes  # type: ignore[attr-defined]

    def configure_optimizers(self):  # type: ignore[override]
        """Configures the optimizer and learning rate scheduler for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,  # type: ignore[assignment]  # patience can be int or float, but torch typing is incorrect
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": self.lr_monitor_key,
            "frequency": self.lr_scheduler_freq,
            "interval": self.lr_scheduler_interval,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def unpack_batch(
        self, batch: Batch
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        """Unpacks a batch of data into its components."""
        observed_perturbed_expression = ensure_2d_batch_tensor(batch.gene_expression)  # type: ignore[attr-defined]
        control_expression = ensure_2d_batch_tensor(batch.controls)  # type: ignore[attr-defined]
        perturbation = ensure_2d_batch_tensor(batch.perturbations)  # type: ignore[attr-defined]
        covariates = (
            batch.covariates if batch.covariates is not None else None
        )  # dict of covariate tensors
        embeddings = ensure_2d_batch_tensor(batch.embeddings)
        output = (  # type: ignore[assignment]  # output is a tuple of tensors, but mypy is not able to infer the types correctly here
            observed_perturbed_expression,
            control_expression,
            perturbation,
            covariates,
            embeddings,
        )
        return output  # type: ignore[return-value]  # output is a tuple of tensors, but mypy is not able to infer the types correctly here

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Saves the training record to the checkpoint."""
        checkpoint["training_record"] = self.training_record

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Loads the training record from the checkpoint."""
        self.training_record = checkpoint["training_record"]

    def predict_step(
        self,
        data_tuple: tuple[Batch, pd.DataFrame],
        batch_idx: int,
    ) -> ad.AnnData | None:
        """
        Given a batch of data, predict the counterfactual perturbed expression as an AnnData object.

        Args:
            data_tuple: A tuple containing the counterfactual batch and a
                pandas DataFrame containing the counterfactual cell level
                metadata.
            batch_idx: The index of the current batch.

        Returns:
            ad.AnnData: The predicted counterfactual perturbed expression.
        """
        counterfactual_batch, counterfactual_obs = data_tuple
        predicted_expression = self.predict(counterfactual_batch).squeeze().cpu().detach().numpy()
        predicted_adata = ad.AnnData(
            X=predicted_expression,
            obs=counterfactual_obs,
        )
        predicted_adata.var_names = counterfactual_batch.gene_names  # type: ignore[attr-defined]  # Set gene names as variable names in the AnnData object
        if self.prediction_output_path is not None:
            predicted_adata.write_h5ad(
                self.prediction_output_path + f"/prediction_chunk_{batch_idx}.h5ad"
            )
        else:
            return predicted_adata

    @abstractmethod
    def predict(self, counterfactual_batch: Batch) -> torch.Tensor:
        """
        Given a counterfactual_batch of data, predicted the counterfactual perturbed expression.

        Example implementation:
        ```
        def predict(self, counterfactual_batch):
            control_expression = counterfactual_batch.gene_expression.squeeze()
            perturbation = counterfactual_batch.perturbations.squeeze()
            covariates = counterfactual_batch.covariates.squeeze()

            predicted_perturbed_expression = self.forward(
                control_expression,
                perturbation,
                covariates,
            )
            return predicted_perturbed_expression
        ```
        """
        pass
