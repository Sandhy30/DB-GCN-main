#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh

_EPS = 1e-12


def _to_csr(W, dtype=np.float32):
    """Return a CSR matrix with desired dtype (if dtype is not None)."""
    if sp.isspmatrix(W):
        W = W.tocsr()
    else:
        W = sp.csr_matrix(W)
    if dtype is not None and W.dtype != dtype:
        W = W.astype(dtype)
    return W


def graph_laplacian(W, normalized=True, symmetrize=True, zero_diag=False):
    
    W = _to_csr(W, dtype=np.float32)
    if zero_diag:
        W.setdiag(0)
        W.eliminate_zeros()
    if symmetrize:
        W = ((W + W.T) * 0.5).tocsr()
    L = csgraph_laplacian(W, normed=normalized)
    return L.tocsr()

def _lmax_power(L, max_iter=200, tol=1e-6, seed=0, use_rayleigh=True):
    """
    Power-method estimate of the largest eigenvalue of symmetric PSD L.
    Stable for large sparse Laplacians where eigsh is expensive.
    """
    L = _to_csr(L, dtype=np.float64)  # numeric stability
    n = L.shape[0]
    if n == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    nv = np.linalg.norm(v)
    if nv < _EPS:
        return 0.0
    v /= nv

    prev_val = 0.0
    for _ in range(max_iter):
        w = L @ v
        normw = np.linalg.norm(w)
        if normw < 1e-30:
            return 0.0
        v = w / normw
        # Rayleigh quotient is more accurate than ||Lv||
        val = float(v @ (L @ v)) if use_rayleigh else normw
        if not np.isfinite(val):
            break
        if abs(val - prev_val) <= tol * max(1.0, abs(prev_val)):
            return max(val, 0.0)
        prev_val = val
    return max(prev_val, 0.0)


def lmax_L(L, method="power", assume_normalized=False, **kwargs):
    """
    Largest eigenvalue (float64). method ∈ {"power","eigsh"}.

    kwargs forwarded to chosen method:
      power: max_iter, tol, seed, use_rayleigh
      eigsh: k=1, which='LA'

    If assume_normalized=True, cap λ_max at 2.0 (normalized Laplacian).
    """
    val = None
    if method == "power":
        try:
            val = _lmax_power(L, **kwargs)
            if np.isfinite(val) and val >= 0.0:
                if assume_normalized:
                    val = min(val, 2.0)
                return float(val)
        except Exception:
            val = None  # fall through to eigsh

    # Fallback: eigsh
    L = _to_csr(L, dtype=np.float64)
    try:
        val = float(eigsh(L, k=1, which='LA', return_eigenvectors=False)[0])
    except Exception:
        val = float(eigsh(L, k=1, which='LM', return_eigenvectors=False)[0])

    if assume_normalized:
        val = min(val, 2.0)
    return float(val)


def rescale_to_unit_interval(L, lmax=None, method="power", **kwargs):
    """Backward-compat shim for rescale_to_unit(...)."""
    return rescale_to_unit(L, lmax=lmax, method=method, **kwargs)

def rescale_L(L, lmax=None, method="power", **kwargs):
    """
    Chebyshev scaling to [-1, 1]:
        L_tilde = (2/λ_max) * L - I
    If λ_max ≈ 0, returns -I (correct limit when L → 0).
    """
    L = _to_csr(L, dtype=np.float32)
    if lmax is None:
        lmax = lmax_L(L, method=method, **kwargs)
    lmax = float(lmax) if np.isfinite(lmax) else 0.0

    n = L.shape[0]
    I = sp.identity(n, format='csr', dtype=L.dtype)

    if lmax <= _EPS:
        return -I
    return (2.0 / lmax) * L - I


def rescale_to_unit(L, lmax=None, method="power", **kwargs):
    """
    Scale to [0, 1]:
        L_hat = L / λ_max
    Useful for Bernstein filtering on [0,1].
    If λ_max ≈ 0, returns zero matrix.
    """
    L = _to_csr(L, dtype=np.float32)
    if lmax is None:
        lmax = lmax_L(L, method=method, **kwargs)
    lmax = float(lmax) if np.isfinite(lmax) else 0.0
    if lmax <= _EPS:
        return L.multiply(0.0)
    return (1.0 / lmax) * L



# Adjacency normalization (for APPNP/GCN/SAGE)

def to_dense_adj(W):
    """Convert sparse/array-like to a float32 dense numpy array."""
    if sp.isspmatrix(W):
        return W.toarray().astype(np.float32, copy=False)
    return np.asarray(W, dtype=np.float32)


def add_self_loops_dense(A):
    """A := A + I on dense arrays."""
    A = np.array(A, dtype=np.float32, copy=True)
    n = A.shape[0]
    A[np.arange(n), np.arange(n)] += 1.0
    return A


def sym_norm_dense(A):
    """
    Symmetric normalization on dense adjacency:
        S = D^{-1/2} A D^{-1/2}
    """
    if sp.isspmatrix(A):
        A = A.toarray()
    A = np.asarray(A, dtype=np.float32)
    d = np.clip(A.sum(axis=1), _EPS, None)
    Dm12 = 1.0 / np.sqrt(d)
    S = (Dm12[:, None] * A) * Dm12[None, :]
    return S.astype(np.float32, copy=False)


def row_norm_dense_with_selfloops(A):
    """
    Row-normalize (A + I):
        T = D̃^{-1} (A + I)
    """
    if sp.isspmatrix(A):
        A = A.toarray()
    A = np.asarray(A, dtype=np.float32)
    n = A.shape[0]
    A = A.copy()
    A[np.arange(n), np.arange(n)] += 1.0  # add self-loops
    rowsum = np.clip(A.sum(axis=1, keepdims=True), _EPS, None)
    return (A / rowsum).astype(np.float32, copy=False)


def appnp_transition(W):
    """APPNP transition: T = D̃^{-1}(A + I)."""
    A = to_dense_adj(W)
    return row_norm_dense_with_selfloops(A)


def gcn_sym_norm(W):
    """GCN normalization: S = D̃^{-1/2}(A + I)D̃^{-1/2}."""
    A = to_dense_adj(W)
    A = add_self_loops_dense(A)
    return sym_norm_dense(A)


def torch_dense(t, device=None, dtype=None):
    """
    Convert scipy.spmatrix/ndarray to a Torch dense tensor.
    Zero-copy on CPU when possible.
    """
    import torch

    arr = t.toarray() if sp.isspmatrix(t) else np.asarray(t)
    if (not arr.flags.get("C_CONTIGUOUS", False)) or (not arr.flags.get("WRITEABLE", True)):
        arr = arr.copy()
    ten = torch.from_numpy(arr).to(dtype=dtype or torch.float32)
    if device is not None:
        ten = ten.to(device)
    return ten


def torch_sparse_csr(S, device=None, dtype=None):
    """
    Convert scipy.sparse.csr_matrix to torch.sparse_csr_tensor.
    """
    import torch

    if not sp.isspmatrix_csr(S):
        S = sp.csr_matrix(S)

    indptr = np.asarray(S.indptr, dtype=np.int64)
    indices = np.asarray(S.indices, dtype=np.int64)

    tdtype = dtype or torch.float32
    if tdtype == torch.float64:
        v_np = np.asarray(S.data, dtype=np.float64)
    elif tdtype == torch.float16:
        v_np = np.asarray(S.data, dtype=np.float16)
    else:
        v_np = np.asarray(S.data, dtype=np.float32)

    indptr_t  = torch.from_numpy(indptr)
    indices_t = torch.from_numpy(indices)
    values_t  = torch.from_numpy(v_np)

    M, N = S.shape
    ten = torch.sparse_csr_tensor(indptr_t, indices_t, values_t, size=(M, N), dtype=tdtype)
    if device is not None:
        ten = ten.to(device)
    return ten

__all__ = [
    "_to_csr",
    "graph_laplacian",
    "lmax_L",
    "rescale_L",
    "rescale_to_unit",
    "rescale_to_unit_interval",
    "to_dense_adj",
    "add_self_loops_dense",
    "sym_norm_dense",
    "row_norm_dense_with_selfloops",
    "appnp_transition",
    "gcn_sym_norm",
    "torch_dense",
    "torch_sparse_csr",
]
