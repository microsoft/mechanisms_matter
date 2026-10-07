"""
GEARS model implementation for perturbation effect prediction.

This module implements the GEARS (Gene Expression Augmented by Regulatory Structure) model,
adapted from https://github.com/snap-stanford/GEARS/blob/master

The model combines gene graph context, perturbation graph context, and gene-specific decoders
to predict how perturbations shift gene expression in single-cell RNA-seq data.

Classes:
    GEARS_Model: Main model combining GNN layers and gene-specific decoders
    MLP: Multi-layer perceptron with configurable architecture
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn import Linear, Module, ReLU
from torch_geometric.nn import SGConv


class GEARS_Model(torch.nn.Module):
    """
    GEARS model.

    Combines gene graph context, perturbation graph context, and gene-specific decoders to model how perturbations shift gene expression.
    """

    def __init__(self, args: dict[str, Any]) -> None:
        """:param args: arguments dictionary"""
        super().__init__()  # type: ignore
        self.args = args
        self.num_genes: int = args["num_genes"]
        self.num_perts: int = args["num_perts"]
        hidden_size: int = args["hidden_size"]
        self.uncertainty: bool = args["uncertainty"]
        self.num_layers: int = args["num_go_gnn_layers"]
        self.indv_out_hidden_size: int = args["decoder_hidden_size"]
        self.num_layers_gene_pos: int = args["num_gene_gnn_layers"]
        self.no_perturb: bool = args["no_perturb"]
        self.pert_emb_lambda: float = 0.2
        self.control_label_pert_idx: int = args["control_label_pert_idx"]

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w: Linear = Linear(1, hidden_size)

        # gene/globel perturbation embedding dictionary lookup
        self.gene_emb: Module = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb: Module = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans: ReLU = ReLU()
        self.pert_base_trans: ReLU = ReLU()
        self.transform: ReLU = ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act="ReLU")
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act="ReLU")

        # gene co-expression GNN
        self.G_coexpress = args["G_coexpress"].to(args["device"])
        self.G_coexpress_weight = args["G_coexpress_weight"].to(args["device"])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)

        # GNN layers for gene positional embedding
        self.layers_emb_pos = torch.nn.ModuleList()
        for _ in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))

        ### perturbation gene ontology GNN
        self.G_sim = args["G_sim"].to(args["device"])
        self.G_sim_weight = args["G_sim_weight"].to(args["device"])

        # GNN layers for perturbation embedding
        self.sim_layers = torch.nn.ModuleList()
        for _ in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size * 2, hidden_size], last_layer_act="linear")

        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes, hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size, hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes, hidden_size + 1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP(
                [hidden_size, hidden_size * 2, hidden_size, 1], last_layer_act="linear"
            )

    def forward(self, data: Any) -> Any:
        """Forward pass of the model."""
        # 0) Unpack graph batch inputs:
        #    - x: baseline gene expression
        #    - pert_idx: perturbation IDs per sample (with -1 as padding/no-perturb marker)
        x, pert_idx = data.x, data.pert_idx
        # pert_idx: is a list of lists, where each inner list corresponds to a sample in the batch and contains the perturbation IDs for that sample. The perturbation IDs are integers, and -1 is used as a padding value to indicate no perturbation for that position in the list.
        # x:
        # 1) Fast path: if perturbations are disabled, return baseline reshaped by graph/sample.
        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)
        else:
            # Number of graphs/samples in this mini-batch.
            num_graphs = len(data.batch.unique())  # num of unique graphs/samples in the batch
            #  # 2) Build base gene embeddings for every (sample, gene).
            ## get base gene embeddings
            emb = self.gene_emb(
                torch.LongTensor(list(range(self.num_genes)))
                .repeat(
                    num_graphs,
                )
                .to(self.args["device"])
            )  # get gene embedding for each gene, repeated for each sample in the batch
            emb = self.bn_emb(emb)  # batchnorm on gene embedding before GNN
            base_emb = self.emb_trans(emb)  # activation transformation before GNN

            ## 3) Augment gene embeddings with positional embedding and gene co-expression GNN.
            pos_emb = self.emb_pos(
                torch.LongTensor(list(range(self.num_genes)))
                .repeat(
                    num_graphs,
                )
                .to(self.args["device"])
            )  # get positional embedding for each gene, repeated for each sample in the batch
            for idx, layer in enumerate(self.layers_emb_pos):
                # apply GNN layers to augment positional embedding with gene co-expression information
                # pos_embd: (num_graphs * num_genes, hidden_size) -> (num_graphs * num_genes, hidden_size) Node features are updated by aggregating features of neighboring nodes in the gene co-expression graph, repeated for each sample in the batch
                # G_coexpress: (2, num_edges) adjacency list of the gene co-expression graph, shared across all samples in the batch
                # G_coexpress_weight: (num_edges,) edge weights for the gene co-expression graph, shared across all samples in the batch
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = (
                base_emb + 0.2 * pos_emb
            )  # augment base gene embedding with positional embedding that has been enriched with gene co-expression information through GNN
            base_emb = self.emb_trans_v2(
                base_emb
            )  # further transform the augmented gene embedding with an MLP

            ## get perturbation index and embeddings for each sample in the batch
            sample_to_targets: dict[int, list[int]] = {}
            for sample_idx, sample_pert_idx in enumerate(pert_idx):
                targets = [
                    int(gene_idx)
                    for gene_idx in sample_pert_idx
                    if gene_idx != self.control_label_pert_idx
                ]
                if targets:
                    sample_to_targets[sample_idx] = targets

            ## 4) Build global perturbation embedding for each sample and augment to perturbed genes.
            pert_global_emb = self.pert_emb(
                torch.LongTensor(list(range(self.num_perts))).to(self.args["device"])
            )

            ## augment global perturbation embedding with GNN
            # G_sim: (2, num_edges) adjacency list of the perturbation similarity graph, shared across all samples in the batch
            # G_sim_weight: (num_edges,) edge weights for the perturbation similarity graph, shared across all samples in the batch
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if sample_to_targets:
                sample_ids = list(sample_to_targets.keys())
                sample_emb_inputs = []
                for sample_idx in sample_ids:
                    sample_emb_inputs.append(
                        torch.stack(
                            [
                                pert_global_emb[gene_idx]
                                for gene_idx in sample_to_targets[sample_idx]
                            ]
                        ).sum(dim=0)
                    )

                pert_inputs = torch.stack(sample_emb_inputs)
                if pert_inputs.shape[0] == 1:
                    fused_emb = self.pert_fuse(torch.cat([pert_inputs, pert_inputs], dim=0))[:1]
                else:
                    fused_emb = self.pert_fuse(pert_inputs)

                base_emb = base_emb.clone()
                for emb_idx, sample_idx in enumerate(sample_ids):
                    for gene_idx in sample_to_targets[sample_idx]:
                        base_emb[sample_idx, gene_idx] = (
                            base_emb[sample_idx, gene_idx] + fused_emb[emb_idx]
                        )

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)  # batchnorm after adding perturbation embedding

            ## apply the first MLP
            base_emb = self.transform(base_emb)
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, dim=2)
            out = w + self.indv_b1

            # Cross gene
            # concatenate the output of the gene-specific decoder with the original input (baseline expression) for each gene, and feed into a cross-gene MLP to get a cross-gene embedding that captures interactions between genes. Then combine the cross-gene embedding with the output of the gene-specific decoder to get the final output for each gene.
            cross_gene_embed = self.cross_gene_state(  # MLP that takes in the output of the gene-specific decoder for all genes and outputs a cross-gene embedding for each gene
                out.reshape(num_graphs, self.num_genes, -1).squeeze(2)
            )
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])
            cross_gene_out = torch.cat(
                [out, cross_gene_embed], 2
            )  # concatenate the output of the gene-specific decoder with the cross-gene embedding for each gene

            cross_gene_out = (
                cross_gene_out * self.indv_w2
            )  # element-wise multiplication with gene-specific weights to get the final output for each gene
            cross_gene_out = torch.sum(cross_gene_out, dim=2)
            out = (
                cross_gene_out + self.indv_b2
            )  # add gene-specific bias to get the final output for each gene
            out = (
                out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
            )  # add the original input (baseline expression) back to the output to get the final predicted expression for each gene, which models how the perturbations shift the gene expression from the baseline
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            # If uncertainty is enabled, also compute the log variance for each gene using a separate MLP head that takes the same base embedding as input. The log variance output has the same shape as the mean output and can be used to model heteroscedastic uncertainty in the predictions.
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)

            return torch.stack(out)


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron module.

    A flexible feed-forward neural network with configurable layer sizes,
    batch normalization, and activation functions.

    Attributes:
    ----------
    network : torch.nn.Sequential
        The sequential container of linear and activation layers.

    Methods:
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass through the MLP.
    """

    def __init__(self, sizes: list[int], batch_norm: bool = True, last_layer_act: str = "linear"):
        """
        Multi-layer perceptron.

        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer.

        """
        super().__init__()  # type: ignore
        layers: list[torch.nn.Module] = []
        n_layers = len(sizes) - 1

        for s in range(n_layers):
            is_last = s == n_layers - 1
            layers.append(torch.nn.Linear(sizes[s], sizes[s + 1]))

            # Apply BN/ReLU on hidden layers only
            if not is_last:
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(sizes[s + 1]))
                layers.append(torch.nn.ReLU())
            else:
                if last_layer_act.lower() == "relu":
                    layers.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        :param x: input tensor
        :return: output tensor after passing through the network
        """
        return self.network(x)
