"""Linear pseudobulk perturbation model based on PCA-projected gene embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.decomposition import PCA


class PseudoPCALinear:
    """
    Linear-PCA perturbation model on perturbation-level pseudobulks.

    `pseudobulk` must be shaped `(n_samples, n_genes)`.
    `target_genes` must provide one target spec per sample; each spec can be:
      - a single gene string (e.g., "KLF1"),
      - a plus-delimited string (e.g., "CEBPE+RUNX1T1"),
      - an iterable of gene strings.
    """

    def __init__(
        self,
        pseudobulk: np.ndarray,
        target_genes: Sequence[str] | Sequence[Sequence[str]],
        gene_names: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        k: int = 10,
        ridge_lambda: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize model inputs, split indices, and regularization settings."""
        self.x = np.asarray(pseudobulk, dtype=np.float64)
        if self.x.ndim != 2:
            raise ValueError(
                f"pseudobulk must be 2D with shape (n_samples, n_genes). Got shape={self.x.shape}."
            )

        self.n_samples, self.n_genes = self.x.shape
        self.y = self.x.T  # genes x samples

        self.gene_names = np.asarray(gene_names).astype(str)
        if self.gene_names.ndim != 1 or self.gene_names.size != self.n_genes:
            raise ValueError(
                "gene_names must be a 1D array with length == number of genes in pseudobulk. "
                f"Got len(gene_names)={self.gene_names.size}, n_genes={self.n_genes}."
            )
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        self.target_genes = self._normalize_targets(target_genes)
        if len(self.target_genes) != self.n_samples:
            raise ValueError(
                "target_genes must provide one target per pseudobulk sample. "
                f"Got len(target_genes)={len(self.target_genes)}, n_samples={self.n_samples}."
            )

        self.train_idx = np.asarray(train_idx, dtype=int)
        self.test_idx = np.asarray(test_idx, dtype=int)
        self._validate_split_indices()

        self.k = int(k)
        self.ridge_lambda = float(ridge_lambda)
        self.seed = int(seed)

        self.g: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None

    @staticmethod
    def _normalize_targets(
        target_genes: Sequence[str] | Sequence[Sequence[str]],
    ) -> list[tuple[str, ...]]:
        normalized: list[tuple[str, ...]] = []
        for item in target_genes:
            if isinstance(item, str):
                tokens = tuple(token for token in item.split("+") if token and token != "control")
            elif isinstance(item, Iterable):
                tokens = tuple(
                    str(token) for token in item if str(token) and str(token) != "control"
                )
            else:
                raise TypeError(f"Unsupported target_genes item type: {type(item)}")
            normalized.append(tokens)
        return normalized

    def _validate_split_indices(self) -> None:
        if self.train_idx.ndim != 1 or self.test_idx.ndim != 1:
            raise ValueError("train_idx and test_idx must be 1D integer arrays.")
        if self.train_idx.size == 0:
            raise ValueError("train_idx cannot be empty.")
        if np.any(self.train_idx < 0) or np.any(self.train_idx >= self.n_samples):
            raise IndexError("train_idx contains out-of-range sample indices.")
        if np.any(self.test_idx < 0) or np.any(self.test_idx >= self.n_samples):
            raise IndexError("test_idx contains out-of-range sample indices.")

    def _targets_to_embedding(self, indices: np.ndarray) -> np.ndarray:
        """Map each sample target spec to PCA gene embedding by averaging target rows."""
        if self.g is None:
            raise RuntimeError("Model is not trained. Call train() before mapping targets.")

        rows: list[np.ndarray] = []
        for sample_idx in np.asarray(indices, dtype=int):
            target_tokens = self.target_genes[sample_idx]
            target_positions = [self.gene_to_idx[g] for g in target_tokens if g in self.gene_to_idx]
            if len(target_positions) == 0:
                raise KeyError(
                    f"Sample index {sample_idx} has no target genes present in gene_names. "
                    f"targets={target_tokens}"
                )
            rows.append(self.g[target_positions, :].mean(axis=0))

        return np.vstack(rows)

    def train(self) -> None:
        """Fit G, W, b using training pseudobulk indices."""
        y_train = self.y[:, self.train_idx]  # genes x n_train

        k_eff = min(self.k, y_train.shape[1], max(1, y_train.shape[0] - 1))
        pca = PCA(n_components=k_eff, svd_solver="full", random_state=self.seed)
        self.g = pca.fit_transform(y_train)  # genes x k_eff

        p = self._targets_to_embedding(self.train_idx)  # n_train x k_eff
        self.b = y_train.mean(axis=1, keepdims=True)  # genes x 1
        y_centered = y_train - self.b

        eye = np.eye(k_eff, dtype=np.float64)
        left = self.g.T @ self.g + self.ridge_lambda * eye
        right = p.T @ p + self.ridge_lambda * eye
        middle = self.g.T @ y_centered @ p

        temp = np.linalg.solve(left, middle)
        self.w = np.linalg.solve(right.T, temp.T).T

    def predict(
        self,
        test_idx: np.ndarray | None = None,
        return_gene_by_sample: bool = False,
    ) -> np.ndarray:
        """Predict pseudobulk expression for test sample indices."""
        if self.g is None or self.w is None or self.b is None:
            raise RuntimeError("Call train() before predict().")

        idx = self.test_idx if test_idx is None else np.asarray(test_idx, dtype=int)
        p_tilde = self._targets_to_embedding(idx)  # n_test x k_eff
        y_hat = self.g @ self.w @ p_tilde.T + self.b  # genes x n_test
        return y_hat if return_gene_by_sample else y_hat.T

    def run(
        self,
        test_idx: np.ndarray | None = None,
        return_gene_by_sample: bool = False,
    ) -> np.ndarray:
        """Train the model and return predictions for test indices."""
        self.train()
        return self.predict(test_idx=test_idx, return_gene_by_sample=return_gene_by_sample)


def _has_known_target(
    tokens: Sequence[str],
    gene_name_set: set[str],
) -> bool:
    return any(token in gene_name_set for token in tokens)


def predict_linear_pca_baseline(
    train_means: np.ndarray,
    train_target_genes: Sequence[Sequence[str] | str],
    test_target_genes: Sequence[Sequence[str] | str],
    gene_names: np.ndarray,
    fallback_mean: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Predict test perturbation pseudobulks with the repository linearPCA baseline.

    Args:
        train_means: Training pseudobulks with shape ``(n_train, n_genes)``.
        train_target_genes: Perturbation targets aligned to ``train_means`` rows.
        test_target_genes: Perturbation targets aligned to requested predictions.
        gene_names: Gene names aligned to the pseudobulk columns.
        fallback_mean: Default prediction used when a perturbation target is unsupported.
        seed: Random seed passed through to ``PseudoPCALinear``.

    Returns:
        Array of predicted test pseudobulks with shape ``(n_test, n_genes)``.
    """
    gene_names_arr = np.asarray(gene_names).astype(str)
    fallback = np.asarray(fallback_mean, dtype=np.float32)
    if fallback.ndim != 1:
        raise ValueError(
            f"fallback_mean must be 1D with length n_genes. Got shape={fallback.shape}."
        )

    train_means_arr = np.asarray(train_means, dtype=np.float64)
    if train_means_arr.ndim != 2:
        raise ValueError(
            f"train_means must be 2D with shape (n_train, n_genes). Got shape={train_means_arr.shape}."
        )
    if train_means_arr.shape[1] != gene_names_arr.size:
        raise ValueError(
            "train_means and gene_names have inconsistent shapes. "
            f"Got n_genes={train_means_arr.shape[1]} and len(gene_names)={gene_names_arr.size}."
        )
    if fallback.size != gene_names_arr.size:
        raise ValueError(
            "fallback_mean and gene_names have inconsistent shapes. "
            f"Got len(fallback_mean)={fallback.size} and len(gene_names)={gene_names_arr.size}."
        )

    normalized_train_targets = PseudoPCALinear._normalize_targets(train_target_genes)
    normalized_test_targets = PseudoPCALinear._normalize_targets(test_target_genes)
    if len(normalized_train_targets) != train_means_arr.shape[0]:
        raise ValueError(
            "train_target_genes must align with train_means rows. "
            f"Got len(train_target_genes)={len(normalized_train_targets)} and n_train={train_means_arr.shape[0]}."
        )

    mu_pred = np.tile(fallback, (len(normalized_test_targets), 1))
    gene_name_set = set(gene_names_arr)
    train_valid_mask = np.array(
        [_has_known_target(tokens, gene_name_set) for tokens in normalized_train_targets],
        dtype=bool,
    )
    test_valid_mask = np.array(
        [_has_known_target(tokens, gene_name_set) for tokens in normalized_test_targets],
        dtype=bool,
    )

    if not (np.any(train_valid_mask) and np.any(test_valid_mask)):
        return mu_pred

    train_targets_valid = [normalized_train_targets[i] for i in np.flatnonzero(train_valid_mask)]
    test_targets_valid = [normalized_test_targets[i] for i in np.flatnonzero(test_valid_mask)]
    n_train_valid = int(np.sum(train_valid_mask))
    n_test_valid = int(np.sum(test_valid_mask))

    linear_input = np.vstack(
        [
            train_means_arr[train_valid_mask, :],
            np.zeros((n_test_valid, train_means_arr.shape[1]), dtype=np.float64),
        ]
    )
    linear_targets: list[tuple[str, ...]] = [*train_targets_valid, *test_targets_valid]
    train_idx_linear = np.arange(n_train_valid, dtype=int)
    test_idx_linear = np.arange(n_train_valid, n_train_valid + n_test_valid, dtype=int)

    linear_model = PseudoPCALinear(
        pseudobulk=linear_input,
        target_genes=linear_targets,
        gene_names=gene_names_arr,
        train_idx=train_idx_linear,
        test_idx=test_idx_linear,
        seed=seed,
    )
    mu_pred_valid = linear_model.run()
    mu_pred[test_valid_mask, :] = mu_pred_valid
    return mu_pred
