# Preprocessing steps for CD4+ T cell data

## Prerequisites

Download the raw data by the following command:

```bash
uv pip install vcp-cli
uv pip install 'vcp-cli[data]'
vcp data search "Primary Human CD4+ T Cell Perturb-seq" --exact

for id in \
  6946b5261d32b0e84ba87057 \
  6946b5261d32b0e84ba8705d \
  6946b5261d32b0e84ba87061 \
  6946b5261d32b0e84ba87063 \
  6946b5261d32b0e84ba87067 \
  6946b5261d32b0e84ba87069 \
  6946b5261d32b0e84ba8705b \
  6946b5261d32b0e84ba8706b \
  6946b5261d32b0e84ba87059 \
  6946b5261d32b0e84ba8705f \
  6946b5261d32b0e84ba87065 \
  6946b5261d32b0e84ba8706d
do
  vcp data download --id "$id" -o .
done
```

## Files
The raw data is stored in `data/cd4+/`. All the raw data files are postfixed with '.assigned_guide.h5ad'.

Each file corresponds to a different donar and a different culture condition.

For example, file 'D1_Rest.assigned_guide.h5ad' contains the expression data for donor 1 in the resting condition. The 'assigned_guide' in the filename indicates that the data has already been processed to assign guides to cells. Another example, file 'D2_Stim8hr.assigned_guide.h5ad' contains the expression data for donor 2 in the 8 hour stimulation condition.

The structure of the annData object in each file is as follows:
Observation Metadata (.obs)
Annotations for each single cell:

lane_id: 10X lane identifier (corresponds to one cellranger output)
n_genes_by_counts: Number of genes with non-zero counts detected in the cell
total_counts: Total UMI counts in the cell
pct_counts_mt: Percentage of counts mapping to mitochondrial genes
top_guide_UMI_counts: UMI counts for the most abundant guide RNA in the cell
guide_id: Unique identifier for the guide RNA detected in the cell (if more than one guide was detected, we annotate as "multi-guide")
perturbed_gene_name: Name of the gene perturbed by the detected guide (before target curation)
perturbed_gene_id: Ensembl gene ID of the perturbed gene (before target curation)
guide_type: Type of guide (e.g., targeting, non-targeting)
PuroR: Puromycin resistance marker expression level
guide_group: Group classification for the guide
low_quality: Boolean flag indicating low-quality cells to be filtered
Variable Metadata (.var)
Annotations for each measured gene:

gene_ids: Ensembl gene identifiers
feature_types: Type of feature (e.g., Gene Expression)
genome: Reference genome used for alignment
gene_name: Gene symbols
mt: Boolean flag indicating mitochondrial genes
Expression Matrix (.X)
Single-cell gene expression data:

Content: UMI counts for each gene in each cell
Data type: Sparse matrix (likely CSR format)
Pseudobulk-level data
Filename: GWCD4i.pseudobulk_merged.h5ad

How to access:

S3 bucket via AWS Command Line
This AnnData object contains pseudobulk expression profiles. Each observation represents a pseudobulk (aggregated by guide, donor and culture condition). Each variable is a measured gene in the transcriptome (n_vars = 18,129).

Observation Metadata (.obs)
Annotations for each pseudobulk sample:

10xrun_id: processing batch identifier (R1 or R2)
donor_id: Donor identifier
culture_condition: Culture condition (Rest, Stim8hr, Stim48hr)
guide_id: Unique guide identifier
perturbed_gene_name: Name of the gene perturbed by the guide (note that the annotated gene in the guide identifier doesn't always match because we did some post-hoc curation of the target gene)
perturbed_gene_id: Ensembl gene ID of the perturbed gene
guide_type: Type of guide (e.g., targeting, non-targeting)
n_cells: Number of cells aggregated in this pseudobulk sample
total_counts: Total UMI counts across all cells in this pseudobulk
log10_n_cells: Log10-transformed number of cells
keep_min_cells: Boolean flag indicating sample passes minimum cell count threshold to be used for DE analysis
keep_effective_guides: Boolean flag indicating guide was considered effective (t-test significant) to be used for DE analysis
keep_total_counts: Boolean flag indicating sample passes total counts threshold to be used for DE analysis
keep_for_DE: Boolean flag indicating sample is suitable for differential expression analysis
keep_test_genes: Boolean flag indicating whether the perturbed gene passes criteria for differential expression analysis
Variable Metadata (.var)
Annotations for each measured gene:

gene_ids: Ensembl gene identifiers
gene_name: Gene symbols
Expression Matrix (.X)
Sum of UMI counts across cells for each gene in each pseudobulk sample

## Step 1

Ensure we have 12 raw data files corresponding to the 12 contexts.
The 12 raw data files are very large so we need to find a memory-efficient way to load and process the data.
Before QC, the pipeline intersects gene IDs across all 12 contexts and reorders each file into that shared canonical gene space.

## Step 2 — Cell + gene quality control and HVG selection

Cell QC is done **separately for each donor-timepoint combination**.
Gene QC is then done **globally across all QC-passed cells** in the shared gene space.

### 2.1 Cell-level filters (per donor–timepoint)
Keep cells that satisfy all of:

1) **Single guide assignment only**. 
2) **≥ 100 detected genes** (cells expressing fewer than 100 genes removed). 
3) **Total UMI outlier filter** using **MAD** on total UMI counts, with:
   - **MAD multiplier = 9**
   - **minimum lower bound = 1400 UMIs**
   - Keep if:
     \[
     \max(1400,\; \text{median} - 9 \times \text{MAD}) \le \text{counts} \le \text{median} + 9 \times \text{MAD}
     \]


4) **No mitochondrial-percent filtering**:
   - They **did not filter** cells by mitochondrial read proportion
   - They **kept mitochondrial expression** (motivated by stress-related responses) 

**Scanpy sketch**
- `adata = adata[adata.obs["guide_group"] == "targeting single sgRNA"]` (or equivalent single-guide rule)
- `sc.pp.filter_cells(adata, min_genes=100)`
- Compute `total_counts = adata.X.sum(axis=1)` (or `adata.obs["total_counts"]`)
- Compute `median`, `MAD = median(|counts - median|)` and apply the bounds above.

### 2.2 Gene-level filters (global after cell QC)
This is applied after aggregating gene counts over all QC-passed cells from all contexts, restricted to the shared gene set.

Remove genes that satisfy **either**:

- Expressed in **< 100 cells**, OR
- **Total count < 100**

**Scanpy sketch**
- `n_cells = (adata.X > 0).sum(axis=0)`
- `tot = adata.X.sum(axis=0)`
- Keep genes where `n_cells >= 100 AND tot >= 100`.

### 2.3 HVG selection (per timepoint; donors combined)
- HVG selection is done **independently per timepoint** using **QC-passed cells from all donors combined**
- By default, each timepoint is capped at **200,000 cells** for HVG selection via proportional subsampling across donor contexts (`--hvg-max-cells-per-timepoint`; use `0` to disable the cap)
- For each timepoint:
  - raw counts are stored in `layers["counts"]`
  - `X` is library-size normalized to **10,000 counts** and **log-transformed**
  - `sc.pp.highly_variable_genes(..., flavor="seurat_v3", layer="counts")` is run on the raw-count layer
- Keep **2,000 HVGs per timepoint**
- Take the **union across timepoints**; the final size depends on overlap and preprocessing settings

**Scanpy sketch**
- For each `timepoint`:
  - concatenate donors at that timepoint (or subset full anndata)
  - `sc.pp.normalize_total(..., target_sum=1e4)`
  - `sc.pp.log1p(...)`
  - `sc.pp.highly_variable_genes(..., flavor="seurat_v3", layer="counts", n_top_genes=2000)`
- Union HVGs across timepoints and subset to the union gene list.

---

## Step 3 — Perturbation + cell filtering by knockdown quality (Appendix C.3)

Compute two knockdown metrics **within each donor–timepoint**:

1) **Perturbation knockdown ratio**  
   Mean(target gene expression across perturbed cells) / mean(target gene expression in controls)

2) **Cell knockdown ratio**  
   Expression(target gene in each perturbed cell) / mean(target gene expression in controls)

### 3.1 Perturbation-level filter
This filter is implemented at the **guide (`guide_id`) level**.
With the current defaults, keep a guide if its **perturbation knockdown ratio < 0.5** (≥ 50% reduction) in **at least two donors within each timepoint** (`Rest`, `Stim8hr`, `Stim48hr`).

### 3.2 Cell-level filter (within kept perturbations)
Keep all **non-targeting controls**.
Within the kept guides, keep only **perturbed cells** with **cell knockdown ratio < 0.5**.

### 3.3 Minimum cell count per perturbation
This filter is also implemented at the **guide (`guide_id`) level**, after the cell-level knockdown filter.
Keep a guide only if it has **at least 256 remaining perturbed cells in every donor-timepoint context**.
Equivalently, remove a guide if it has **fewer than 256 remaining cells in any donor-timepoint context**.
This step enforces a shared guide set across all donor-timepoint contexts.


**Implementation notes**
- “Controls” here refers to **non-targeting controls** within the same `(donor, timepoint)` context.
- In the current implementation, knockdown ratios are computed on the **raw counts** from the source `.assigned_guide.h5ad` files.
- Before the knockdown filters are applied, rows whose perturbed target gene is not in the final retained gene set are excluded.
- Filtering in Step 3 is done on `guide_id`, while saved `obs["perturbation"]` / `obs["condition"]` values are `perturbed_gene_id` for targeting cells and `control` for non-targeting cells.


## Step 4 — Save results 

Save the processed data as chunked `.h5ad` files to keep the workflow memory efficient.
For each chunk, `X` contains normalized/log-transformed values and `layers["counts"]` stores raw counts.
Also save:

- gene metadata as `cd4_genes.csv.gz`
- per-context cell QC summary as `cell_qc_summary.csv`
- post-Step-3 filter summary as `filter_summary.csv`
- perturbation knockdown metrics as `perturbation_knockdown_metrics.csv.gz`
- a top-level `processed_manifest.json` describing the run and all chunk paths
