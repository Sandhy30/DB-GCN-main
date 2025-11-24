#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import numpy as np

def normalize_by_train(
    X: np.ndarray,
    train_index: np.ndarray,
    method: str = "zscore",
    eps: float = 1e-8,
):
    """
    Normalize features using *train-only* statistics.

    X shape:
      - single-omics: (N, G, 1)
      - multi-omics : (N, G, C)  e.g., C=2 for [CNV, Expr]

    Returns:
      X_norm : same shape as X
      stats  : dict with per-gene/per-channel stats for reproducibility
    """
    if X.ndim == 2:
        # allow (N, G) by promoting to (N, G, 1)
        X = X[:, :, None]

    N, G, C = X.shape
    tr = np.asarray(train_index, dtype=int)
    X_norm = X.copy()

    if method == "zscore":
        mu = X[tr].mean(axis=0, keepdims=True)          # (1, G, C)
        sd = X[tr].std(axis=0, keepdims=True) + eps     # (1, G, C)
        X_norm = (X - mu) / sd
        stats = {"method": "zscore", "mu": mu, "sd": sd}

    elif method == "minmax":
        mn = X[tr].min(axis=0, keepdims=True)
        mx = X[tr].max(axis=0, keepdims=True)
        rng = (mx - mn)
        rng[rng < eps] = 1.0
        X_norm = (X - mn) / rng
        stats = {"method": "minmax", "min": mn, "max": mx}

    elif method == "robust":
        q1 = np.percentile(X[tr], 25, axis=0, keepdims=True)
        q3 = np.percentile(X[tr], 75, axis=0, keepdims=True)
        iqr = (q3 - q1)
        iqr[iqr < eps] = 1.0
        X_norm = (X - q1) / iqr
        stats = {"method": "robust", "q1": q1, "q3": q3}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, stats

def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy for logits (N, C) vs int64 labels (N,)."""
    labels = labels.to(torch.long)                 # ensure correct dtype
    preds = output.argmax(dim=1)
    correct = (preds == labels).to(torch.float64).sum()
    return correct / labels.numel()

def encode_onehot(labels):
    """
    Deterministic one-hot encoder.
    NOTE: sorts classes so output is stable across runs/platforms.
    """
    labels = list(labels)
    classes = sorted(set(labels))
    class_to_vec = {c: np.eye(len(classes), dtype=np.float32)[i] for i, c in enumerate(classes)}
    return np.stack([class_to_vec[c] for c in labels], axis=0).astype(np.float32)

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a torch.sparse_coo_tensor (float32)."""
    if not sp.issparse(sparse_mx):
        raise TypeError("Expected a SciPy sparse matrix.")
    mx = sparse_mx.tocoo().astype(np.float32)
    idx = np.vstack((mx.row, mx.col)).astype(np.int64)
    indices = torch.from_numpy(idx)
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def load_singleomic_data(expression_data_path: str) -> pd.DataFrame:
    df = pd.read_csv(expression_data_path, sep="\t", index_col=0, header=0)
    if "icluster_cluster_assignment" not in df.columns:
        raise KeyError("Column 'icluster_cluster_assignment' missing in expression table.")
    # ensure integer labels for PyTorch (long)
    df["icluster_cluster_assignment"] = df["icluster_cluster_assignment"].astype("int64")
    return df

def load_multiomics_data(expression_data_path: str, cnv_data_path: str):
    expr = load_singleomic_data(expression_data_path)
    cnv  = pd.read_csv(cnv_data_path, sep="\t", index_col=0, header=0)
    # Ensure sample alignment
    if not expr.index.equals(cnv.index):
        common = expr.index.intersection(cnv.index)
        if len(common) == 0:
            raise ValueError("Expression and CNV have disjoint samples; cannot align.")
        expr = expr.loc[common]
        cnv  = cnv.loc[common]
    assert expr.index.equals(cnv.index), "Post-alignment indices still differ."
    return expr, cnv


def _numeric_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature columns (exclude the label column even if numeric)."""
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "icluster_cluster_assignment" in cols:
        cols.remove("icluster_cluster_assignment")
    feat = df[cols]
    # fail early on NaN/Inf
    if not np.isfinite(feat.to_numpy(dtype="float32", copy=False)).all():
        raise ValueError("Non-finite values found in feature table; clean or impute before training.")
    return feat

def _read_non_null_list(non_null_index_path: str):
    """Read allowed genes (optional). Accepts columns: 'gene' (required), plus optional index columns."""
    try:
        nn = pd.read_csv(non_null_index_path)
        if "gene" in nn.columns:
            allowed = nn["gene"].astype(str).tolist()
            for idx_col in ("index", "id", "idx"):
                if idx_col in nn.columns:
                    return allowed, nn[idx_col].astype(int).tolist()
            return allowed, None
    except Exception:
        pass
    return None, None

def select_top_genes_from_train_fold(expression_df: pd.DataFrame,
                                     non_null_index_path: str,
                                     train_index: np.ndarray,
                                     num_gene: int):
    """
    Compute per-gene variance ONLY on TRAIN rows, then select top-N genes.
    Returns:
        - gene_list: ordered names of selected genes
        - gene_idx: positions in the *full expression* numeric-column order
    """
    feat_df = _numeric_gene_columns(expression_df)
    expr_train = feat_df.iloc[train_index]
    if expr_train.empty:
        raise ValueError("Training subset for variance selection is empty.")

    variances = expr_train.var(axis=0, ddof=0)  # population variance (stable)

    allowed, _ = _read_non_null_list(non_null_index_path) if non_null_index_path else (None, None)
    if allowed is not None:
        mask = variances.index.astype(str).isin(set(allowed))
        variances = variances[mask]
        if variances.empty:
            raise ValueError(
                "After applying non-null gene list, no genes remain. "
                "Check 'non_null_index_path' consistency."
            )

    n = min(int(num_gene), variances.shape[0])
    if n <= 0:
        raise ValueError(f"Requested num_gene={num_gene}, but {variances.shape[0]} candidates are available.")
    top_genes = variances.nlargest(n).index.tolist()

    full_gene_order = list(feat_df.columns)
    missing = [g for g in top_genes if g not in full_gene_order]
    if missing:
        raise KeyError(f"Selected genes not found in expression table: {missing[:10]}...")

    gene_idx = [full_gene_order.index(g) for g in top_genes]
    return top_genes, gene_idx

def _load_gene_index_map(non_null_index_path: str, expression_df: pd.DataFrame):
    """
    Map {gene_name -> adjacency_row_index}.
    Priority:
      1) If non_null_index_path has columns ('gene', one of ['index','id','idx']),
         use that mapping (most reliable).
      2) Else fallback to the expression_df numeric-column order,
         assuming adjacency was built in the same order.
    """
    allowed, idxs = _read_non_null_list(non_null_index_path) if non_null_index_path else (None, None)
    if allowed is not None and idxs is not None and len(allowed) == len(idxs):
        return dict(zip([str(g) for g in allowed], [int(i) for i in idxs]))
    gene_order = list(_numeric_gene_columns(expression_df).columns)
    return {g: i for i, g in enumerate(gene_order)}

def _slice_adjacency(adj_path: str, adj_idx: list, add_self_loops: bool) -> sp.csr_matrix:
    """Load sparse adjacency and slice rows/cols by adj_idx without densifying."""
    A = sp.load_npz(adj_path)
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    max_req = max(adj_idx) if len(adj_idx) else -1
    if max_req >= A.shape[0]:
        raise IndexError(
            f"Adjacency index {max_req} out of bounds for matrix with shape {A.shape}. "
            "Check your non_null mapping / gene order."
        )
    A_sel = A[adj_idx, :][:, adj_idx].astype(np.float32).tocsr()
    if add_self_loops:
        A_sel = (A_sel + sp.eye(A_sel.shape[0], format="csr", dtype=np.float32)).tocsr()
        A_sel.eliminate_zeros()  # keep structure compact
    A_sel.sort_indices()         # cheap + helpful
    return A_sel

def build_fold_data_singleomics(expression_df: pd.DataFrame,
                                adj_path: str,
                                gene_list: list,
                                gene_idx: list,
                                singleton: bool = False,
                                non_null_index_path: str = None):
    """
    Leak-free single-omics builder:
      - Slices expression by gene_list for ALL samples -> (N, G, 1)
      - Slices sparse adjacency by gene indices matching adjacency row order (no densify)
    """
    if not gene_list:
        raise ValueError("gene_list is empty.")
    labels = expression_df["icluster_cluster_assignment"].to_numpy(dtype=np.int64) - 1

    feat_df = _numeric_gene_columns(expression_df)
    missing = [g for g in gene_list if g not in feat_df.columns]
    if missing:
        raise KeyError(f"Missing genes in expression table: {missing[:10]}...")
    X = feat_df.reindex(columns=gene_list).to_numpy(dtype=np.float32).reshape(-1, len(gene_list), 1)

    gene_to_adj = _load_gene_index_map(non_null_index_path, expression_df) if non_null_index_path else \
                  {g: i for i, g in enumerate(feat_df.columns)}
    try:
        adj_idx = [int(gene_to_adj[g]) for g in gene_list]
    except KeyError as e:
        raise KeyError(f"Gene {e} missing in adjacency mapping. Check 'non_null_index_path'.")
    A_sel = _slice_adjacency(adj_path, adj_idx, add_self_loops=singleton)
    return A_sel, X, labels, gene_list

def build_fold_data_multiomics(expression_df: pd.DataFrame,
                               cnv_df: pd.DataFrame,
                               adj_path: str,
                               gene_list: list,
                               gene_idx: list,
                               singleton: bool = False,
                               non_null_index_path: str = None):

    if not gene_list:
        raise ValueError("gene_list is empty.")
    labels = expression_df["icluster_cluster_assignment"].to_numpy(dtype=np.int64) - 1

    expr_feat = _numeric_gene_columns(expression_df)
    missing_expr = [g for g in gene_list if g not in expr_feat.columns]
    missing_cnv  = [g for g in gene_list if g not in cnv_df.columns]
    if missing_expr:
        raise KeyError(f"Missing genes in EXPRESSION table: {missing_expr[:10]}...")
    if missing_cnv:
        raise KeyError(f"Missing genes in CNV table: {missing_cnv[:10]}...")

    X_expr = expr_feat.reindex(columns=gene_list).to_numpy(dtype=np.float32).reshape(-1, len(gene_list), 1)
    X_cnv  = cnv_df.reindex(columns=gene_list).to_numpy(dtype=np.float32).reshape(-1, len(gene_list), 1)
    X = np.concatenate([X_cnv, X_expr], axis=2)  # (N, G, 2) [CNV, Expr]

    gene_to_adj = _load_gene_index_map(non_null_index_path, expression_df) if non_null_index_path else \
                  {g: i for i, g in enumerate(expr_feat.columns)}
    try:
        adj_idx = [int(gene_to_adj[g]) for g in gene_list]
    except KeyError as e:
        raise KeyError(f"Gene {e} missing in adjacency mapping. Check 'non_null_index_path'.")

    A_sel = _slice_adjacency(adj_path, adj_idx, add_self_loops=singleton)
    return A_sel, X, labels, gene_list
