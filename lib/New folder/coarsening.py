#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh

# =========================================================
#                  CSR + symmetry helpers
# =========================================================
def _to_csr(W, dtype=np.float32, copy=False):
    """
    Ensure CSR, desired dtype, and sorted indices.
    """
    if sp.isspmatrix(W):
        W = W.tocsr(copy=copy)
    else:
        W = sp.csr_matrix(W, copy=copy)
    if W.dtype != dtype:
        W = W.astype(dtype, copy=False)
    W.sort_indices()
    return W

def _symmetrize(W):
    """
    Symmetrize without densifying: 0.5 * (W + W.T).
    """
    return ((W + W.T) * 0.5).tocsr()

# =========================================================
#                       Laplacians
# =========================================================
def graph_laplacian(W, normalized=True, assume_symmetric=False):
    """
    Return CSR Laplacian for adjacency W.
      - Unnormalized:      L = D - W              (normalized=False)
      - Sym. normalized:   L = I - D^{-1/2} W D^{-1/2}  (normalized=True)

    Parameters
    ----------
    W : array-like or sparse
        Adjacency (will be converted to CSR float32).
    normalized : bool
        If True, build the symmetric normalized Laplacian.
    assume_symmetric : bool
        If False, W is symmetrized as 0.5*(W+W.T).

    Returns
    -------
    L : csr_matrix (float32)
    """
    W = _to_csr(W, dtype=np.float32)
    if not assume_symmetric:
        W = _symmetrize(W)
    L = csgraph_laplacian(W, normed=normalized)
    return _to_csr(L, dtype=np.float32)

# =========================================================
#                 Largest eigenvalue (λ_max)
# =========================================================
def lmax_L(L, normalized_hint=False, max_iter=60, tol=1e-5, random_state=0):
    """
    Estimate λ_max(L).
    If L is the symmetric normalized Laplacian, set normalized_hint=True
    (spectrum ⊆ [0,2]) to immediately return 2.0.

    Returns
    -------
    float
    """
    L = _to_csr(L, dtype=np.float64)

    if normalized_hint:
        # For L_sym, λ_max ≤ 2; using 2.0 is a safe + cheap upper bound
        return 2.0

    # Preferred: eigsh (PSD, symmetric)
    try:
        vals = eigsh(L, k=1, which='LA', return_eigenvectors=False)
        return float(vals[0])
    except Exception:
        pass
    try:
        vals = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        return float(vals[0])
    except Exception:
        pass

    # Fallback: power iteration (deterministic)
    rs = np.random.RandomState(int(random_state))
    n = L.shape[0]
    x = rs.randn(n).astype(np.float64)
    x /= np.linalg.norm(x) + 1e-12
    lam_prev = 0.0
    for _ in range(max_iter):
        y = L @ x
        ny = np.linalg.norm(y) + 1e-12
        x = y / ny
        lam = float(x @ (L @ x))
        if abs(lam - lam_prev) <= tol * max(1.0, abs(lam_prev)):
            break
        lam_prev = lam
    return float(lam)

# =========================================================
#         Rescale Laplacian spectrum to [0, 1]
# =========================================================
def rescale_to_unit_interval(L, lmax=None, normalized_hint=False):
    """
    Spectral scaling for Bernstein: L_tilde = L / λ_max.
    If L is the normalized Laplacian, pass lmax=2.0 and/or normalized_hint=True
    to avoid any eigensolve.

    Returns
    -------
    L_tilde : csr_matrix (float32)
    """
    L = _to_csr(L, dtype=np.float32)
    if lmax is None:
        lmax = lmax_L(L, normalized_hint=normalized_hint)
    lm = float(lmax)
    if not np.isfinite(lm) or lm <= 0:
        lm = 1.0  # ultra-conservative fallback
    return (L * (1.0 / lm)).tocsr()

# =========================================================
#        One-liner: A  ->  L_sym  ->  L_tilde in [0,1]
# =========================================================
def build_rescaled_normalized_laplacian(A, assume_symmetric=False):
    """
    Convenience wrapper used by DB-GCN:

    A (adjacency) → L_sym = I - D^{-1/2} A D^{-1/2} →  ẼL = L_sym / 2

    This guarantees the **spectrum** of ẼL lies in [0,1] (safely using λ_max=2).

    Returns
    -------
    L_tilde : csr_matrix (float32)
    """
    L_sym = graph_laplacian(A, normalized=True, assume_symmetric=assume_symmetric)
    # Cheap, guaranteed scaling for normalized Laplacian:
    return rescale_to_unit_interval(L_sym, lmax=2.0, normalized_hint=True)

# =========================================================
#         (Optional) quick spectral sanity check
# =========================================================
def approx_lambda_bounds(L, k=2):
    """
    VERY light diagnostic: returns approx (λ_min, λ_max) using eigsh on k extremes.
    Use only for debugging small/medium graphs.
    """
    L = _to_csr(L, dtype=np.float64)
    try:
        lmax = eigsh(L, k=1, which='LA', return_eigenvectors=False)[0]
        lmin = eigsh(L, k=1, which='SA', return_eigenvectors=False)[0]
        return float(lmin), float(lmax)
    except Exception:
        return np.nan, np.nan
