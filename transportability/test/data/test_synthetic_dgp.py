from __future__ import annotations

import unittest

import anndata as ad
import numpy as np

from perturbations.data.dgp import causalDGP, directDGP


def _make_gene_names(n: int) -> np.ndarray:
    return np.asarray([f"gene_{i}" for i in range(n)], dtype=str)


class SyntheticDGPTests(unittest.TestCase):
    def test_direct_dgp_returns_in_memory_anndata(self) -> None:
        G = 16
        N0 = 8
        Nk = 4
        P = 3
        control_mu = np.linspace(0.1, 1.6, 32, dtype=np.float64)
        pert_mu = control_mu + 0.2
        all_theta = np.full(32, 2.0, dtype=np.float64)

        adata, affected_masks = directDGP(
            G=G,
            N0=N0,
            Nk=Nk,
            P=P,
            p_effect=0.2,
            effect_factor=2.0,
            B=0.5,
            mu_l=1.0,
            all_theta=all_theta,
            control_mu=control_mu,
            pert_mu=pert_mu,
            gene_names=_make_gene_names(32),
            seed=0,
        )

        self.assertIsInstance(adata, ad.AnnData)
        self.assertEqual(adata.shape, (N0 + P * Nk, G))
        self.assertIn("normalized_log1p", adata.layers)
        self.assertNotIn("counts", adata.layers)
        self.assertListEqual(
            ["perturbation", "perturbation_id", "cell_line"],
            list(adata.obs.columns),
        )
        self.assertEqual(int(np.sum(adata.obs["perturbation"] == "control")), N0)
        self.assertTrue(np.all(adata.obs["cell_line"].to_numpy() == 0))
        self.assertEqual(len(affected_masks), P)
        self.assertTrue(all(mask.shape == (G,) for mask in affected_masks))

    def test_causal_dgp_returns_in_memory_anndata(self) -> None:
        G = 18
        N0 = 6
        Nk = 4
        P = 3
        all_theta = np.full(36, 2.0, dtype=np.float64)

        adata, affected_masks_pair = causalDGP(
            G=G,
            N0=N0,
            Nk=Nk,
            P=P,
            mu_l=1.0,
            all_theta=all_theta,
            gene_names=_make_gene_names(36),
            seed=1,
            diversity_type="both",
            mask_method="power-law",
            visualize=False,
        )

        self.assertIsInstance(adata, ad.AnnData)
        self.assertEqual(adata.shape, (N0 + P * Nk, G))
        self.assertIn("normalized_log1p", adata.layers)
        self.assertNotIn("counts", adata.layers)
        self.assertListEqual(
            ["perturbation", "perturbation_id", "cell_line"],
            list(adata.obs.columns),
        )
        self.assertSetEqual(set(np.unique(adata.obs["cell_line"])), {0, 1})
        self.assertEqual(len(affected_masks_pair), 2)
        self.assertTrue(all(len(mask_group) == P for mask_group in affected_masks_pair))
        self.assertTrue(
            all(mask.shape == (G,) for mask_group in affected_masks_pair for mask in mask_group)
        )

    def test_causal_dgp_handles_condition_sizes_larger_than_sampler_chain_cap(self) -> None:
        G = 12
        N0 = 96
        Nk = 80
        P = 2
        all_theta = np.full(24, 2.0, dtype=np.float64)

        adata, _ = causalDGP(
            G=G,
            N0=N0,
            Nk=Nk,
            P=P,
            mu_l=1.0,
            all_theta=all_theta,
            gene_names=_make_gene_names(24),
            seed=7,
            diversity_type="A",
            mask_method="power-law",
            visualize=False,
        )

        self.assertEqual(adata.shape, (N0 + P * Nk, G))
        counts = adata.obs["perturbation"].value_counts().to_dict()
        self.assertEqual(counts["control"], N0)
        non_control_counts = [count for label, count in counts.items() if label != "control"]
        self.assertEqual(len(non_control_counts), P)
        self.assertTrue(all(count == Nk for count in non_control_counts))


if __name__ == "__main__":
    unittest.main()
