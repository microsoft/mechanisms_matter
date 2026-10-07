"""Loader for state-gene perturbation prediction model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from anndata import AnnData
from omegaconf import DictConfig
from scipy.sparse import issparse  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

from .utils import (
    get_dataset_cfg,
    get_embedding_cfg,
)

log = logging.getLogger(__file__)

# Threshold for flagging implausibly high UMIs when we undo a log1p
EXPONENTIATED_UMIS_LIMIT = 5_000_000
# If any count exceeds this, we confidently treat the tensor as raw integers
RAW_COUNT_HEURISTIC_THRESHOLD = 35


# THIS SHOULD ONLY BE USED FOR INFERENCE
def create_dataloader(
    cfg: DictConfig,
    adata: AnnData,
    adata_name: str,
    workers: int = 1,
    data_dir: str | None = None,
    sentence_collator: VCIDatasetSentenceCollator | None = None,
    protein_embeds: dict[str, torch.Tensor] | None = None,
    precision: torch.dtype | None = None,
    gene_column: str = "gene_name",
) -> DataLoader[FilteredGenesCounts]:
    """Expected to be used for inference  Either datasets and shape_dict or adata and adata_name should be provided."""
    shuffle = False

    if data_dir:
        get_dataset_cfg(cfg).data_dir = data_dir

    dataset = FilteredGenesCounts(
        cfg,
        adata=adata,
        adata_name=adata_name,
        protein_embeds=protein_embeds,
        gene_column=gene_column,
    )
    if sentence_collator is None:
        sentence_collator = VCIDatasetSentenceCollator(
            cfg,
            valid_gene_mask=dataset.valid_gene_index,
            ds_emb_mapping_inference=dataset.ds_emb_map,
            is_train=False,
            precision=precision,
        )

    # validation should not use cell augmentations
    sentence_collator.training = False

    dataloader: DataLoader[FilteredGenesCounts] = DataLoader(  # type: ignore
        dataset,
        batch_size=cfg.model.batch_size,
        shuffle=shuffle,
        collate_fn=sentence_collator,
        num_workers=workers,
        persistent_workers=True,
    )
    return dataloader


# class H5adSentenceDataset(data.Dataset):
class H5adSentenceDataset(data.Dataset[tuple[torch.Tensor, int, str | None, int]]):
    """Dataset for loading sentences from h5ad files. Can also be initialized with an AnnData object directly, in which case it will ignore the h5ad loading logic and just use the provided AnnData for inference."""

    def __init__(
        self,
        cfg: DictConfig,
        adata: AnnData,
        adata_name: str,
        test: bool = False,
    ) -> None:
        """Dataset for loading sentences from h5ad files. Can also be initialized with an AnnData object directly, in which case it will ignore the h5ad loading logic and just use the provided AnnData for inference."""
        super().__init__()

        self.adata_name = adata_name
        self.test = test

        self.adata = adata
        self.datasets = [adata_name]
        self.shapes_dict = {self.datasets[0]: adata.shape}

        self.datasets = sorted(self.datasets)
        self.cfg = cfg

        self.num_cells: dict[str, int] = {}
        self.num_genes: dict[str, int] = {}

        self.total_num_cells = 0
        for name in self.datasets:
            num_cells, num_genes = self.shapes_dict[name]
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets_to_num = {
            k: v for k, v in zip(self.datasets, range(len(self.datasets)), strict=True)
        }

    def _compute_index(self, idx: int) -> tuple[str, int]:
        for dataset in self.datasets:
            if idx < self.num_cells[dataset]:
                return dataset, idx
            else:
                idx -= self.num_cells[dataset]
        raise IndexError

    # @functools.lru_cache
    # def dataset_file(self, dataset: str) -> h5py.File:
    #     datafile = self.dataset_path_map[dataset]
    #     return h5py.File(datafile, "r")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str | None, int]:
        """Get a cell's gene expression counts as a sentence. If initialized with an AnnData, it will use the AnnData for loading instead of h5ad files."""
        # block is only used during validation
        # if .X is a numpy.ndarray
        if isinstance(self.adata.X, np.ndarray):  # type: ignore
            counts = torch.tensor(self.adata.X[idx]).reshape(1, -1)  # type: ignore
        else:
            counts = torch.tensor(self.adata.X[idx].todense())  # type: ignore

        dataset = self.adata_name
        dataset_num = 0
        return counts, idx, dataset, dataset_num

    def __len__(self) -> int:
        """Get the total number of cells across all datasets."""
        return self.total_num_cells

    def get_dim(self) -> dict[str, int]:
        """Get the number of genes for each dataset."""
        return self.num_genes


class FilteredGenesCounts(H5adSentenceDataset):
    """Dataset that filters genes based on a provided valid gene mask, and also computes the mapping from dataset-specific gene indices to global embedding indices. This is used for inference when we want to make sure we only include genes that have embeddings, and to align the dataset's gene indices with the global embedding indices."""

    def __init__(
        self,
        cfg: DictConfig,
        adata: AnnData,
        adata_name: str,
        test: bool = False,
        protein_embeds: dict[str, torch.Tensor] | None = None,
        gene_column: str | None = "gene_name",
    ) -> None:
        """FilteredGenesCounts dataset initializer. In addition to the standard H5adSentenceDataset initialization, it also computes a valid gene mask for each dataset based on the provided protein embeddings, and creates a mapping from the dataset's gene indices to the global embedding indices."""
        super().__init__(
            cfg=cfg,
            test=test,
            adata=adata,
            adata_name=adata_name,
        )
        self.valid_gene_index: dict[str, np.ndarray] = {}
        self.protein_embeds = protein_embeds
        self.gene_column = gene_column

        # make sure we get training datasets
        self.datasets: list[str] = []
        self.shapes_dict: dict[str, tuple[int, int]] = {}
        self.ds_emb_map: dict[str, np.ndarray] = {}

        emb_cfg = get_embedding_cfg(self.cfg)
        # for inference, let's make sure this dataset's valid mask is available
        # append it to self.datasets
        self.datasets.append(adata_name)
        self.shapes_dict[adata_name] = adata.shape

        # compute its embedding index vector
        esm_data = self.protein_embeds or torch.load(emb_cfg["all_embeddings"], weights_only=False)
        valid_genes_list: list[str] = list(esm_data.keys())
        # make a gene→global index lookup
        global_pos = {g: i for i, g in enumerate(valid_genes_list)}

        # grab var_names from the AnnData
        gene_names = np.array(adata.var_names)

        # for each gene in this dataset, find its global idx or -1 if missing
        new_mapping = np.array([global_pos.get(g, -1) for g in gene_names])
        if (new_mapping == -1).all():
            # probably it contains ensembl id's instead
            assert self.gene_column in adata.var.keys(), (
                f"Column '{self.gene_column}' not found in adata.var. Available columns: {list(adata.var.keys())}"
            )
            gene_names = adata.var[self.gene_column].values
            new_mapping = np.array([global_pos.get(g, -1) for g in gene_names])

            log.info(
                f"{(new_mapping != -1).sum()} genes mapped to embedding file (out of {len(new_mapping)})"
            )

        self.ds_emb_map[adata_name] = new_mapping

        print(
            f"!!! {(self.ds_emb_map[adata_name] != -1).sum()} genes mapped to embedding file (out of {len(self.ds_emb_map[adata_name])})"
        )

        esm_data = self.protein_embeds or torch.load(emb_cfg["all_embeddings"], weights_only=False)
        valid_genes_list = list(esm_data.keys())
        for name in self.datasets:
            gene_names = np.array(adata.var_names)
            valid_mask = np.isin(gene_names, valid_genes_list)

            if not valid_mask.any():
                # none of the genes were valid, probably ensembl id's
                gene_names = np.array(adata.var[self.gene_column].values)
                valid_mask = np.isin(gene_names, valid_genes_list)

            self.valid_gene_index[name] = valid_mask

    def __getitem__(self, idx: int):
        """Get a cell's gene expression counts as a sentence, after applying the valid gene mask to filter out genes that don't have embeddings. Also returns the mapping from dataset-specific gene indices to global embedding indices."""
        counts, idx, dataset, dataset_num = super().__getitem__(idx)
        return counts, idx, dataset, dataset_num


class VCIDatasetSentenceCollator:
    """Collator for the state-gene perturbation prediction model. It takes a batch of cells (with their counts, dataset names, etc.) and processes them into the input format expected by the model, including creating the cell sentences, task sentences, masks, and any other necessary tensors. It also handles any dataset-specific logic such as mapping gene indices to global embedding indices and applying valid gene masks."""

    def __init__(
        self,
        cfg: DictConfig,
        valid_gene_mask: dict[str, np.ndarray] | None = None,
        ds_emb_mapping_inference: dict[str, np.ndarray] | None = None,
        is_train: bool = True,
        precision: torch.dtype | None = None,
    ):
        """Initializer for the collator. It sets up the necessary configurations, loads the dataset mappings, and prepares any other necessary information for processing batches."""
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.S = cfg.dataset.S
        self.cfg = cfg
        self.training = is_train
        self.precision = precision

        # Load the dataset mappings
        self.use_dataset_info = getattr(cfg.model, "dataset_correction", False)
        self.batch_tabular_loss = getattr(cfg.model, "batch_tabular_loss", False)

        if valid_gene_mask is not None:
            # this branch is for inference
            self.valid_gene_mask = valid_gene_mask
            self.dataset_to_protein_embeddings = ds_emb_mapping_inference
        else:
            # otherwise for training, load from config
            gene_mask_file = get_embedding_cfg(self.cfg).valid_genes_masks
            if gene_mask_file is not None:
                # we have a config for training
                self.valid_gene_mask = torch.load(gene_mask_file, weights_only=False)
            else:
                # we don't have a config for training
                self.valid_gene_mask = None

            self.dataset_to_protein_embeddings = torch.load(
                get_embedding_cfg(self.cfg).ds_emb_mapping.format(get_embedding_cfg(self.cfg).size),
                weights_only=False,
            )

        self.global_size = get_embedding_cfg(self.cfg).num
        self.global_to_local: dict[str, torch.Tensor] = {}
        assert self.dataset_to_protein_embeddings is not None
        for dataset_name, ds_emb_idxs in self.dataset_to_protein_embeddings.items():
            # make sure tensor with long data type
            ds_emb_idxs = torch.tensor(ds_emb_idxs, dtype=torch.long)
            # assert ds_emb_idxs.unique().numel() == ds_emb_idxs.numel(), f"duplicate global IDs in dataset {dataset_name}!"

            # Create a tensor filled with -1 (indicating not present in this dataset)
            reverse_mapping = torch.full((self.global_size,), -1, dtype=torch.int64)

            local_indices = torch.arange(ds_emb_idxs.size(0), dtype=torch.int64)
            mask = (ds_emb_idxs >= 0) & (ds_emb_idxs < self.global_size)
            reverse_mapping[ds_emb_idxs[mask]] = local_indices[mask]
            self.global_to_local[dataset_name] = reverse_mapping

    def __call__(self, batch: list[tuple[torch.Tensor, int, str, int]]):
        """Collate a batch of cells into the input format expected by the model. This includes creating the cell sentences, task sentences, masks, and any other necessary tensors. It also handles any dataset-specific logic such as mapping gene indices to global embedding indices and applying valid gene masks."""
        num_aug = getattr(self.cfg.model, "num_downsample", 1)
        if num_aug > 1 and self.training:
            # for each original sample, duplicate it num_aug times
            batch = [item for item in batch for _ in range(num_aug)]

        batch_size = len(batch)

        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.int32)
        batch_sentences_counts = torch.zeros((batch_size, self.pad_length))
        masks = torch.zeros((batch_size, self.pad_length), dtype=torch.bool)

        idxs = torch.zeros(batch_size, dtype=torch.int32)
        if self.cfg.loss.name == "tabular":
            task_num = self.P + self.N + self.S
        else:
            task_num = self.P + self.N
        task_input_indices = torch.zeros((batch_size, (task_num)), dtype=torch.int32)
        task_output_indices = torch.zeros((batch_size, (task_num)))

        largest_cnt = max([x[0].shape[1] for x in batch])
        batch_weights = torch.zeros((batch_size, largest_cnt))

        total_counts_all = None
        if self.cfg.model.rda:
            total_counts_all = torch.zeros(batch_size)

        datasets: list[str] = []
        for (
            _,
            _,
            ds_name,
            _,
        ) in batch:
            datasets.append(ds_name)

        if self.cfg.loss.name == "tabular":
            if "global_size" not in self.__dict__:
                self.global_size = get_embedding_cfg(self.cfg).num
            shared_genes = torch.randint(
                low=0, high=self.global_size, size=(self.S,), device=masks.device, dtype=torch.long
            )
        else:
            shared_genes = None

        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)

        i = 0
        max_len = 0
        for counts, idx, dataset, dataset_num in batch:
            if self.valid_gene_mask is not None:
                if dataset in self.valid_gene_mask:
                    valid_mask = self.valid_gene_mask[dataset]
                else:
                    valid_mask = None
            else:
                valid_mask = None

            # compute downsample fraction. this is the first sample of the augmentation then
            # use no downsampling
            downsample_fraction = (
                1.0 if (num_aug > 1 and i % num_aug == 0 and self.training) else None
            )
            (bs, xx, yy, batch_weight, mask, cell_total_counts, cell_sentence_counts) = (
                self.sample_cell_sentences(
                    counts, dataset, shared_genes, valid_mask, downsample_fraction
                )
            )

            batch_sentences[i, :] = bs
            masks[i, :] = mask
            batch_weight = batch_weight.squeeze()
            batch_weights[i, : len(batch_weight)] = batch_weight

            max_len = max(max_len, self.cfg.dataset.pad_length)
            idxs[i] = idx

            task_input_indices[i] = xx  # [pn_idx]
            task_output_indices[i] = yy.squeeze()  # [pn_idx]
            dataset_nums[i] = dataset_num

            if (
                self.cfg.model.rda
                and cell_total_counts is not None
                and total_counts_all is not None
            ):
                total_counts_all[i] = cell_total_counts[0]
            if self.cfg.model.counts and cell_sentence_counts is not None:
                batch_sentences_counts[i, :] = cell_sentence_counts
            i += 1

        # Cast tensors to specified precision if provided
        if self.precision is not None:
            # batch_sentences = batch_sentences.to(dtype=self.precision)
            # Xs = Xs.to(dtype=self.precision)
            task_output_indices = task_output_indices.to(dtype=self.precision)
            batch_weights = batch_weights.to(dtype=self.precision)
            if total_counts_all is not None:
                total_counts_all = total_counts_all.to(dtype=self.precision)

            batch_sentences_counts = batch_sentences_counts.to(dtype=self.precision)

        return (
            batch_sentences[:, :max_len],
            task_input_indices,
            task_output_indices,
            idxs,
            batch_weights,
            masks,
            total_counts_all if self.cfg.model.rda else None,
            batch_sentences_counts if self.cfg.model.counts else None,
            dataset_nums if self.use_dataset_info else None,
        )

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax in a numerically stable way."""
        e_x = torch.exp(x - torch.max(x))
        return e_x / e_x.sum()

    def is_raw_integer_counts(self, counts: torch.Tensor):
        """
        Check whether counts are raw integer UMI counts versus log1p-transformed.

        1. If any entry > RAW_COUNT_HEURISTIC_THRESHOLD, assume raw ints.
        2. Otherwise, invert log1p (via expm1) and sum:
        - If the total UMIs exceeds EXPONENTIATED_UMIS_LIMIT, it means
            the data were actually raw ints that we mistakenly log-transformed.
        - Otherwise, assume the data were correctly log1p counts.
        """
        max_val = torch.max(counts).item()

        # Primary heuristic: very large individual counts => raw counts
        if max_val > RAW_COUNT_HEURISTIC_THRESHOLD:
            return True

        # Ambiguous case: try undoing log1p
        total_umis = int(torch.expm1(counts).sum().item())
        if total_umis > EXPONENTIATED_UMIS_LIMIT:
            return True

        return False

    # sampling a single cell sentence
    # counts_raw is a view of a cell
    def sample_cell_sentences(
        self,
        counts_raw: torch.Tensor,
        dataset: str,
        shared_genes: torch.Tensor | None = None,
        valid_gene_mask: np.ndarray | None = None,
        downsample_frac: float | None = None,
    ):
        """Given the raw counts for a single cell, create the cell sentence, task sentence, and associated tensors for that cell. This includes applying any necessary transformations to the counts (e.g., log1p), creating the cell sentence based on the most highly expressed genes, creating the task sentence based on the P most expressed and N least expressed genes, and applying any dataset-specific logic such as mapping gene indices to global embedding indices and applying valid gene masks."""
        if torch.isnan(counts_raw).any():
            log.error(f"NaN values in counts for dataset {dataset}")

        if torch.any(counts_raw < 0):
            counts_raw = F.relu(counts_raw)

        if self.is_raw_integer_counts(counts_raw):  # CAN WE CHANGE THIS TO INT VS REAL
            # total_umis = int(counts_raw.sum(dim=1).item())
            # count_expr_dist = counts_raw / counts_raw.sum(dim=1, keepdim=True)
            counts_raw = torch.log1p(counts_raw)
        # else:  # counts are already log1p
        #     exp_log_counts = torch.expm1(counts_raw)
        #     # total_umis = int(exp_log_counts.sum(dim=1).item())
        #     # count_expr_dist = exp_log_counts / exp_log_counts.sum(dim=1, keepdim=True)

        ### At this point, counts_raw is assumed to be log counts ###

        # store the raw counts here, we need them as targets
        original_counts_raw = counts_raw.clone()

        # logic to sample a single cell sentence and task sentence here
        assert self.dataset_to_protein_embeddings is not None
        ds_emb_idxs = torch.tensor(self.dataset_to_protein_embeddings[dataset], dtype=torch.long)

        original_counts = original_counts_raw
        counts = counts_raw
        if valid_gene_mask is not None:
            if ds_emb_idxs.shape[0] == valid_gene_mask.shape[0]:
                # Filter the dataset embedding indices based on the valid gene mask
                ds_emb_idxs = ds_emb_idxs[valid_gene_mask]
            else:
                # Our preprocessing is such that sometimes the ds emb idxs are already filtered
                # in this case we do nothing to (no subsetting) but assert that the mask matches
                assert valid_gene_mask.sum() == ds_emb_idxs.shape[0], (
                    f"Something wrong with filtering or mask for dataset {dataset}"
                )

            # Counts are never filtered in our preprocessing step, so we always need to apply the valid genes mask
            if counts_raw.shape[1] == valid_gene_mask.shape[0]:
                counts = counts_raw[:, valid_gene_mask]
                original_counts = original_counts_raw[:, valid_gene_mask]

        # so counts are filtered. wtf is happening with the tabular loss then? and why do we error out?
        if counts.sum() == 0:
            expression_weights = F.softmax(counts, dim=1)
        else:
            expression_weights = counts / torch.sum(counts, dim=1, keepdim=True)

        cell_sentences = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        cell_sentence_counts = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        mask = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length), dtype=torch.bool)

        if self.cfg.loss.name == "tabular":
            # include capacity for shared genes
            task_num = self.cfg.dataset.P + self.cfg.dataset.N + self.cfg.dataset.S
        else:
            task_num = self.cfg.dataset.P + self.cfg.dataset.N

        task_counts = torch.zeros((counts.shape[0], task_num))
        task_sentence = torch.zeros((counts.shape[0], task_num))

        if self.cfg.model.rda:
            cell_total_counts = torch.zeros((counts.shape[0],))
        else:
            cell_total_counts = None

        # len(counts) = 1, e.g., we are looping over [cell]
        for c, cell in enumerate(counts):
            num_pos_genes = torch.sum(cell > 0)
            # this is either the number of positive genes, or the first pad_length / 2 most expressed genes
            # the first is only used if you have more expressed genes than pad_length / 2
            assert self.cfg.model.counts
            # shuffle before argsort - randomly break ties so we select random unexpressed genes each time, if pad_length > num_non_zero genes
            indices = torch.randperm(cell.shape[-1])
            shuffled_cell = cell[indices]
            shuffled_genes_ranked_exp = torch.argsort(shuffled_cell, descending=True)
            genes_ranked_exp = indices[shuffled_genes_ranked_exp]
            cell_sentences[c, 0] = self.cfg.dataset.cls_token_idx
            if len(genes_ranked_exp) >= self.cfg.dataset.pad_length - 1:
                cell_sentences[c, 1:] = genes_ranked_exp[: self.cfg.dataset.pad_length - 1]
            else:
                # take the nonzero genes first
                num_nonzero = min(num_pos_genes, self.cfg.dataset.pad_length - 1)
                cell_sentences[c, 1 : num_nonzero + 1] = genes_ranked_exp[:num_nonzero]

                # sample the unexpressed genes with replacement
                remaining_slots = self.cfg.dataset.pad_length - 1 - num_nonzero
                unexpressed_genes = genes_ranked_exp[num_nonzero:]
                cell_sentences[c, num_nonzero + 1 :] = unexpressed_genes[
                    torch.randint(len(unexpressed_genes), (remaining_slots,))
                ]

            cell_sentence_counts[c, :] = (
                100 * expression_weights[c, cell_sentences[c, :].to(torch.long)]
            )

            # Convert tokens to Embeddings - local to global
            # this also includes the cls token, but we will override it later with a learnable torch vector
            cell_sentences[c, :] = ds_emb_idxs[cell_sentences[c, :].to(torch.int32)]

            # pick P expressed genes to mask for MLM
            exp_genes = torch.where(cell > 0)[0]
            if len(exp_genes) > self.cfg.dataset.P:
                task_sentence[c, : self.cfg.dataset.P] = exp_genes[
                    torch.randperm(len(exp_genes))[0 : self.cfg.dataset.P]
                ]
            elif len(exp_genes) > 0:
                task_sentence[c, : self.cfg.dataset.P] = exp_genes[
                    torch.randint(len(exp_genes), (self.cfg.dataset.P,))
                ]

            # get the total number of genes unique to this cell; everything
            # past this are shared genes across all cells in a batch, used for tabular loss
            unshared_num = self.cfg.dataset.P + self.cfg.dataset.N

            unexp_genes = torch.where(cell < 1)[0]
            if len(unexp_genes) > self.cfg.dataset.N:
                task_sentence[c, self.cfg.dataset.P : unshared_num] = unexp_genes[
                    torch.randperm(len(unexp_genes))[0 : self.cfg.dataset.N]
                ]
            else:
                task_sentence[c, self.cfg.dataset.P : unshared_num] = unexp_genes[
                    torch.randint(len(unexp_genes), (self.cfg.dataset.N,))
                ]

            # set counts for unshared genes
            task_idxs = task_sentence[c, :unshared_num].to(torch.int32)
            task_counts[c, :unshared_num] = original_counts[c, task_idxs]

            # convert from dataset specific gene indices to global gene indices
            # only do this for everything up to shared genes, which are already global indices
            task_sentence[c, :unshared_num] = ds_emb_idxs[
                task_sentence[c, :unshared_num].to(torch.int32)
            ]

            # now take care of shared genes across all cells in the batch
            if shared_genes is not None:
                # Overwrite the final positions of task_sentence

                task_sentence[c, unshared_num:] = (
                    shared_genes  # in the old impl these are global gene indices
                )
                # task_sentence[c, unshared_num:] = ds_emb_idxs[shared_genes.to(torch.int32)] # in the new impl these are local gene indices

                # convert the shared_genes, which are global indices, to the dataset specific indices
                local_indices = self.global_to_local[dataset][shared_genes].to(
                    cell.device
                )  # in the old impl these are global gene indices
                # local_indices = shared_genes # in the new impl these are local gene indices

                shared_counts = torch.zeros(
                    local_indices.shape, dtype=cell.dtype, device=cell.device
                )
                valid_mask = local_indices != -1
                if valid_mask.any():
                    shared_counts[valid_mask] = original_counts_raw[c, local_indices[valid_mask]]

                # for indices which are -1, count is 0, else index into cell
                task_counts[c, unshared_num:] = shared_counts

            assert self.cfg.model.rda
            # sum the counts of the task sentence
            if cell_total_counts is not None:
                cell_total_counts[c] = torch.sum(task_counts[c])

            if self.cfg.loss.name == "cross_entropy":
                # binarize the counts to 0/1
                task_counts[c] = (task_counts[c] > 0).float()

            # make sure that the CLS token is never masked out.
            mask[c, 0] = False

            assert not task_counts.isnan().any()
            assert not counts.isnan().any()

        return (
            cell_sentences,
            task_sentence,
            task_counts,
            counts,
            mask,
            cell_total_counts if self.cfg.model.rda else None,
            cell_sentence_counts if self.cfg.model.counts else None,
        )


class PerturbationSetDataset(data.Dataset[dict[str, torch.Tensor | list[str]]]):
    """
    Dataset that groups cells by perturbation condition for the ST model.

    Each sample pairs ``cell_sentence_len`` control cells with
    ``cell_sentence_len`` perturbed cells under one perturbation condition.
    """

    def __init__(
        self,
        adata: AnnData,
        perturbation_column: str,
        control_label: str,
        cell_sentence_len: int,
        pert_categories: list[str],
        context_key: str | None = None,
        basal_embedding_key: str | None = None,
    ) -> None:
        """Initialize dataset by extracting control and perturbed expression matrices."""
        # When ``basal_embedding_key`` is provided, the basal (control) state is
        # drawn from ``adata.obsm[basal_embedding_key]`` (e.g. a foundation-model
        # cell embedding) while the perturbed target remains gene expression from
        # ``adata.X``. This lets the model consume a pretrained encoder embedding
        # as the basal state and still predict in gene space.
        self.cell_sentence_len = cell_sentence_len
        self.pert_categories = pert_categories
        n_perts = len(pert_categories)
        self.pert_to_idx = {p: i for i, p in enumerate(pert_categories)}
        self.n_perts = n_perts

        # Build context (cell type / cell line) integer mapping
        if context_key is not None and context_key in adata.obs.columns:
            all_contexts = sorted(adata.obs[context_key].unique().tolist())
            self.context_to_idx: dict[str, int] = {c: i for i, c in enumerate(all_contexts)}
            self.context_labels: np.ndarray | None = np.asarray(adata.obs[context_key])
        else:
            self.context_to_idx = {}
            self.context_labels = None

        # Basal (control) state source: optional precomputed embedding in obsm,
        # otherwise gene expression from .X. The perturbed target always uses .X.
        if basal_embedding_key is not None:
            if basal_embedding_key not in adata.obsm:
                raise KeyError(
                    f"basal_embedding_key='{basal_embedding_key}' not found in adata.obsm "
                    f"(available: {list(adata.obsm.keys())})."
                )
            basal_raw: Any = adata.obsm[basal_embedding_key]
            basal_source = basal_raw.toarray() if issparse(basal_raw) else np.asarray(basal_raw)
            basal_source = np.asarray(basal_source, dtype=np.float32)
        else:
            basal_source = adata.X

        ctrl_mask = np.asarray(adata.obs[perturbation_column]) == control_label
        ctrl_X = basal_source[ctrl_mask]  # type: ignore[index]
        self.ctrl_X: np.ndarray = np.asarray(
            ctrl_X.todense() if issparse(ctrl_X) else ctrl_X,  # type: ignore[union-attr]
            dtype=np.float32,
        )
        # When the basal state is an embedding, also keep the control cells'
        # gene expression so the training-time sparsity penalty (delta-to-basal)
        # can still be computed in gene space against the true control profile.
        self.ctrl_X_genes: np.ndarray | None = None
        if basal_embedding_key is not None:
            ctrl_genes = adata.X[ctrl_mask]  # type: ignore[index]
            self.ctrl_X_genes = np.asarray(
                ctrl_genes.todense() if issparse(ctrl_genes) else ctrl_genes,  # type: ignore[union-attr]
                dtype=np.float32,
            )
        # Store context indices for control cells so we can sample matching controls
        self.ctrl_context: np.ndarray | None = (
            np.asarray(
                [self.context_to_idx[c] for c in self.context_labels[ctrl_mask]], dtype=np.int64
            )
            if self.context_labels is not None
            else None
        )

        # Store per-(perturbation, context) cell pools for dynamic re-sampling
        self._pert_pools: list[tuple[str, np.ndarray, int]] = []
        self.samples: list[tuple[str, int, int]] = []  # (pert_name, pool_index, ctx_idx)
        labels = np.asarray(adata.obs[perturbation_column])
        for pert in pert_categories:
            if pert == control_label:
                continue
            pert_mask = labels == pert

            # Group cells by context so each sentence has a single, correct context
            if self.context_labels is not None:
                pert_contexts = self.context_labels[pert_mask]
                unique_contexts = np.unique(pert_contexts)
                context_groups: list[tuple[np.ndarray, int]] = []
                for ctx in unique_contexts:
                    ctx_within_pert = pert_contexts == ctx
                    pert_X_ctx = adata.X[pert_mask][ctx_within_pert]  # type: ignore[index]
                    pert_X_ctx = np.asarray(
                        pert_X_ctx.todense() if issparse(pert_X_ctx) else pert_X_ctx,  # type: ignore[union-attr]
                        dtype=np.float32,
                    )
                    ctx_idx: int = self.context_to_idx[ctx]
                    context_groups.append((pert_X_ctx, ctx_idx))
            else:
                pert_X_all = adata.X[pert_mask]  # type: ignore[index]
                pert_X_all = np.asarray(
                    pert_X_all.todense() if issparse(pert_X_all) else pert_X_all,  # type: ignore[union-attr]
                    dtype=np.float32,
                )
                context_groups = [(pert_X_all, 0)]

            for pert_X, ctx_idx in context_groups:
                if pert_X.shape[0] < cell_sentence_len or self.ctrl_X.shape[0] < cell_sentence_len:
                    raise ValueError(
                        f" Perturbation '{pert}' (context={ctx_idx}) with {pert_X.shape[0]} cells (< cell_sentence_len={cell_sentence_len})"
                    )
                pool_idx = len(self._pert_pools)
                self._pert_pools.append((pert, pert_X, ctx_idx))
                # Create one sample entry per sentence worth of cells
                n_samples = max(1, pert_X.shape[0] // cell_sentence_len)
                for _ in range(n_samples):
                    self.samples.append((pert, pool_idx, ctx_idx))

    def __len__(self) -> int:
        """Return the number of perturbation set samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[str]]:
        """Return a dict with control, perturbed, perturbation embedding, and batch tensors."""
        pert, pool_idx, ctx_idx = self.samples[idx]
        # Re-sample fresh perturbed cells each time for data augmentation
        _, pert_pool, _ = self._pert_pools[pool_idx]
        pert_ids = np.random.choice(  # noqa
            pert_pool.shape[0], size=self.cell_sentence_len, replace=True
        )
        pert_chunk = pert_pool[pert_ids]
        # Sample context-matched controls when context information is available
        if self.ctrl_context is not None:
            ctx_mask = self.ctrl_context == ctx_idx
            pool = np.flatnonzero(ctx_mask)
            if pool.size > 0:
                ctrl_ids = np.random.choice(pool, size=self.cell_sentence_len, replace=True)  # noqa
            else:
                ctrl_ids = np.random.choice(  # noqa
                    self.ctrl_X.shape[0], size=self.cell_sentence_len, replace=True
                )
        else:
            ctrl_ids = np.random.choice(  # noqa
                self.ctrl_X.shape[0], size=self.cell_sentence_len, replace=True
            )
        one_hot = np.zeros(self.n_perts, dtype=np.float32)
        one_hot[self.pert_to_idx[pert]] = 1.0
        pert_emb = np.tile(one_hot, (self.cell_sentence_len, 1))
        item: dict[str, torch.Tensor | list[str]] = {
            "ctrl_cell_emb": torch.from_numpy(self.ctrl_X[ctrl_ids]),  # type: ignore[reportUnknownMemberType]
            "pert_cell_emb": torch.from_numpy(pert_chunk),  # type: ignore[reportUnknownMemberType]
            "pert_emb": torch.from_numpy(pert_emb),  # type: ignore[reportUnknownMemberType]
            "batch": torch.full((self.cell_sentence_len,), ctx_idx, dtype=torch.long),
            "pert_name": [pert] * self.cell_sentence_len,
        }
        # Provide gene-space control expression for the delta-to-basal penalty
        # when the basal input is an embedding.
        if self.ctrl_X_genes is not None:
            item["ctrl_cell_gene"] = torch.from_numpy(self.ctrl_X_genes[ctrl_ids])  # type: ignore[reportUnknownMemberType]
        return item


def _perturbation_collate(
    batch_list: list[dict[str, torch.Tensor | list[str]]],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate perturbation set samples by concatenating tensors and extending string lists."""
    result: dict[str, torch.Tensor | list[str]] = {}
    for k in batch_list[0]:
        vals = [b[k] for b in batch_list]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.cat(vals, dim=0)  # type: ignore[arg-type]
        else:
            flat: list[str] = []
            for v in vals:
                flat.extend(v)  # type: ignore[arg-type]
            result[k] = flat
    return result


def create_perturbation_dataloader(
    adata: AnnData,
    perturbation_column: str,
    control_label: str,
    cell_sentence_len: int,
    pert_categories: list[str],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    context_key: str | None = None,
    basal_embedding_key: str | None = None,
) -> DataLoader[dict[str, torch.Tensor | list[str]]]:
    """
    Build a DataLoader yielding dict batches for StateTransitionPerturbationModel.

    Args:
        adata: AnnData with expression in .X and perturbation labels in .obs.
        perturbation_column: Column name in adata.obs with perturbation labels.
        control_label: Label for control/unperturbed cells.
        cell_sentence_len: Number of cells per set (must match model's cell_sentence_len).
        pert_categories: All perturbation categories (including control).
        batch_size: Number of perturbation sets per batch.
        shuffle: Whether to shuffle samples.
        num_workers: DataLoader workers.
        context_key: Optional column in adata.obs for cell type/context conditioning.
        basal_embedding_key: Optional key in adata.obsm holding a precomputed
            basal cell embedding (e.g. a foundation-model encoder output). When
            set, the basal state is read from that embedding while the perturbed
            target stays gene expression from .X.

    Returns:
        DataLoader yielding dicts with keys: ctrl_cell_emb, pert_cell_emb, pert_emb, batch.
    """
    dataset = PerturbationSetDataset(
        adata=adata,
        perturbation_column=perturbation_column,
        control_label=control_label,
        cell_sentence_len=cell_sentence_len,
        pert_categories=pert_categories,
        context_key=context_key,
        basal_embedding_key=basal_embedding_key,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_perturbation_collate,
        num_workers=num_workers,
    )
