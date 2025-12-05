## Dataset

The processed datasets used in this work are available in the following Google Drive folder:

- [Google Drive – DB-GCN datasets](https://drive.google.com/drive/folders/16tBjsK88flqBP2xOEAH44swrCF5Z2jmZ?usp=drive_link)

These files were derived from TCGA Pan-Cancer Atlas data (RNA-seq and CNV) and gene networks (STRING, BioGRID, Coexpression) using the preprocessing steps described below.

---

## Data and preprocessing

This repository uses pan-cancer TCGA data (RNA + CNV) together with gene-level networks (STRING / BioGRID / Coexpression).  
The steps below describe how the CSV input files were created from the raw sources.

---

### 1. Download TCGA pan-cancer data (UCSC Xena)

1. Open the **UCSC Xena** portal.
2. From the TCGA Pan-Cancer / PanCanAtlas hub, download:
   - Gene expression matrix (genes × samples, e.g. log2-normalized RNA-seq such as RSEM / TPM).
   - Gene-level CNV matrix (e.g. GISTIC calls or log2 CNV).
   - Clinical table with sample barcodes and cancer type / subtype labels.
3. Save these raw files under, for example:
   - `data/raw/expression_tcga.tsv`
   - `data/raw/cnv_tcga.tsv`
   - `data/raw/clinical_tcga.tsv`

---

### 2. Select samples and cancer types

4. Use the clinical table to:
   - Keep **tumor** samples only (drop normal and metastatic samples).
   - Keep only the **cancer types / subtypes** used in the paper (e.g. the 28 TCGA subtypes reported in the manuscript).
5. Extract the TCGA barcode for each sample (e.g. first 15 characters).
6. Compute the intersection of barcodes that appear in:
   - the expression matrix,
   - the CNV matrix,
   - the clinical table.
7. Reorder all three tables so that samples appear in the **same order**.  
   Save the final list of sample IDs as:
   - `data/processed/sample_ids_used.txt`.

---

### 3. Harmonize genes across omics

8. Ensure all genes use **HGNC gene symbols** (convert from Ensembl IDs if needed).
9. Remove non-gene rows (e.g. `NA`, `?`, chromosome coordinates, summary rows).
10. Take the intersection of genes present in **both**:
    - the expression matrix,
    - the CNV matrix.
11. Reorder rows so that genes appear in the **same order** in both omics.
12. Drop genes with **zero variance** across samples in either expression or CNV.

This produces two matched matrices:

- `expression_allgenes.tsv` — samples × genes (RNA).
- `cnv_allgenes.tsv` — samples × genes (CNV).

---

### 4. Feature selection for target gene budgets

13. For each target gene budget (e.g. 1000, 2000, …), run a **feature selection** method on the expression data  
    (in this work: per-gene variance; other selectors such as ANOVA or random forest are also supported):

    - Within each **training fold** of cross-validation, rank genes using the chosen selector.
    - Select the top-N genes in that fold.
    - Apply the same gene subset to the validation and test samples in that fold.

14. For reproducibility, compute the *global* union of all genes that were ever selected across folds for a given budget N, and save it as:

    - `data/processed/gene_list_1000.txt`
    - `data/processed/gene_list_2000.txt`
    - etc.

    Each file contains one HGNC symbol per line, in the exact order used when constructing matrices.

> **Note:** Scaling (z-score per gene) is performed **inside the training code**, using training folds only, and then applied to validation and test data.  
> The released CSV matrices are **not** z-scored in advance, so there is no leakage from test into train.

---

### 5. Build N × N adjacency matrices (STRING / BioGRID / Coexpression)

For each network type and each gene list (e.g. N = 1000, 2000):

15. Obtain a gene–gene network for human:
    - **STRING**: protein–protein interactions with confidence scores.
    - **BioGRID**: gene–gene / protein–protein interaction list.
    - **Coexpression**: gene–gene correlations (constructed from expression if not pre-supplied).

16. Filter edges to keep only interactions where **both genes** are present in `gene_list_N.txt`.
17. Create an **N × N** weighted adjacency matrix `A`:
    - Rows and columns follow the exact order of `gene_list_N.txt`.
    - If an edge between genes `g_i` and `g_j` exists, set  
      `A[i, j] = A[j, i] =` the corresponding network weight (confidence / correlation).
    - If no edge exists, set `A[i, j] = A[j, i] = 0`.
    - Set diagonal entries `A[i, i] = 0`.
18. Replace any missing values with `0` so the matrix contains **no NaN / null** entries.
19. Save each adjacency matrix as a dense CSV, for example:
    - `graphs/STRING_adj_1000.csv`
    - `graphs/BioGRID_adj_1000.csv`
    - `graphs/COEXP_adj_1000.csv`

    Recommended format:
    - First row: header with an empty top-left cell, then gene symbols.
    - First column: gene symbols.
    - Inner cells: numeric edge weights.

---

### 6. Final files expected by the code

For each gene budget N, the training code expects the following files:

- `data/processed/expression_Ngenes.csv`  
  → samples × N RNA matrix  
  (columns follow the order in `gene_list_N.txt`).

- `data/processed/cnv_Ngenes.csv`  
  → samples × N CNV matrix  
  (same gene order as the expression file).

- `data/processed/labels.csv`  
  → table with `sample_id` and cancer type / subtype label;  
  the sample order matches the rows in the omics matrices.

- `data/processed/gene_list_N.txt`  
  → ordered list of N HGNC symbols.

- `graphs/<NETWORK>_adj_N.csv`  
  → N × N adjacency matrix with no null values.

All matrices use:

- the **same sample order** across RNA, CNV, and labels,  
- the **same gene order** across all omics and adjacency matrices.
