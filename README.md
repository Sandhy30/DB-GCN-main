## Dataset

The processed datasets used in this work are available at the following Google Drive folder:

- [Google Drive – DB-GCN datasets](https://drive.google.com/drive/folders/16tBjsK88flqBP2xOEAH44swrCF5Z2jmZ?usp=drive_link)

## Data and preprocessing
 repository  uses pan-cancer TCGA data (RNA + CNV) and gene networks (STRING / BioGRID / Coexpression).  
Below are the exact preprocessing steps used to create the CSV input files.

---

### 1. Download TCGA pan-cancer data (UCSC Xena)

1. Open the **UCSC Xena** portal.
2. From the TCGA Pan-Cancer / PanCanAtlas hub download:
   - Gene expression matrix (genes × samples, e.g. log2 TPM / RSEM).
   - Gene-level CNV matrix (GISTIC calls or log2 CNV).
   - Clinical table with sample barcodes and cancer type / subtype labels.
3. Save these raw files under, for example:
   - `data/raw/expression_tcga.tsv`
   - `data/raw/cnv_tcga.tsv`
   - `data/raw/clinical_tcga.tsv`

---

### 2. Select samples and cancer types

4. Use the clinical table to:
   - Keep **tumor** samples only (drop normals / metastatic).
   - Keep only the **cancer types / subtypes** used in the paper.
5. Extract the TCGA barcode for each sample (e.g. first 15 characters).
6. Compute the intersection of barcodes that appear in:
   - the expression matrix,
   - the CNV matrix,
   - the clinical table.
7. Reorder all three tables so that samples appear in the **same order**.  
   Save the list as `data/processed/sample_ids_used.txt`.

---

### 3. Harmonize genes across omics

8. Ensure all genes use **HGNC gene symbols** (convert if needed).
9. Remove non-gene rows (e.g. `NA`, `?`, chromosome locations).
10. Take the intersection of genes present in **both**:
    - the expression matrix,
    - the CNV matrix.
11. Reorder rows so that genes appear in the **same order** in both omics.
12. Drop genes with **zero variance** across samples in expression or CNV.

The resulting matrices are:
- `expression_allgenes.tsv` (samples × genes),
- `cnv_allgenes.tsv` (samples × genes).

---

### 4. Feature selection for target gene budgets

13. For each target gene budget (e.g. 1000, 2000, …), run a feature selection method  
    (variance / ANOVA / random forest, as used in the paper) **inside CV folds**:
    - Within each training fold, rank genes using the chosen selector.
    - Select the top-N genes.
    - Apply the same gene subset to validation and test samples in that fold.
14. Save the *global* union of selected genes for each budget as:
    - `data/processed/gene_list_1000.txt`
    - `data/processed/gene_list_2000.txt`
    - etc.  
    Each file contains one HGNC symbol per line in the order used for matrices.

> Note: Scaling (z-score per gene) is performed **inside the training code** using  
> training folds only, and then applied to validation/test. No leak from test into train.

---

### 5. Build N × N adjacency matrices (STRING / BioGRID / Coexpression)

For each network type and each gene list (e.g. N = 1000, 2000):

15. Download the network **edge list** for human:
    - STRING: PPI interactions with confidence scores.
    - BioGRID: gene–gene interactions.
    - Coexpression: gene–gene correlations (constructed from expression if not pre-given).
16. Filter edges to keep only interactions where **both genes** are in `gene_list_N.txt`.
17. Create an **N × N weighted adjacency matrix** `A`:
    - Rows and columns are in the same order as `gene_list_N.txt`.
    - If an edge between genes \(g_i\) and \(g_j\) exists, set
      - `A[i, j] = A[j, i] =` network weight (confidence / correlation).
    - If no edge exists, set `A[i, j] = A[j, i] = 0`.
    - Set diagonal entries `A[i, i] = 0`.
18. Replace any missing values with `0` so the matrix contains **no NaN / null** values.
19. Save each adjacency matrix as a dense CSV, for example:
    - `graphs/STRING_adj_1000.csv`
    - `graphs/BioGRID_adj_1000.csv`
    - `graphs/COEXP_adj_1000.csv`

    Recommended format:
    - First row: header with an empty cell, then gene symbols.
    - First column: gene symbols.
    - Inner cells: numeric edge weights.

---

### 6. Final files used by the code

The training code expects the following (per gene budget N):

- `data/processed/expression_Ngenes.csv`  
  → samples × N RNA matrix (columns follow `gene_list_N.txt` order).

- `data/processed/cnv_Ngenes.csv`  
  → samples × N CNV matrix (same gene order).

- `data/processed/labels.csv`  
  → sample_id, cancer_type / subtype label; sample order matches the rows in the omics matrices.

- `data/processed/gene_list_N.txt`  
  → ordered list of N HGNC symbols.

- `graphs/<NETWORK>_adj_N.csv`  
  → N × N adjacency matrix with no null values.

All matrices use:
- the **same sample order** across RNA, CNV, and labels,
- the **same gene order** across omics and adjacency matrices.

