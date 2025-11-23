#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coarsening / Laplacian utilities (CPU + GPU).

CPU (used by your main script):
- graph_laplacian(A, normalized=True, assume_symmetric=False) -> scipy.sparse.csr_matrix
- rescale_to_unit_interval(L, normalized_hint=True) -> scipy.sparse.csr_matrix
  (For normalized Laplacian, returns L/2 so spectrum ∈ [0,1])

Optional GPU helpers (if you want an all-CUDA path):
- graph_laplacian_gpu(A_torch, normalized=True, assume_symmetric=False) -> torch.sparse_coo_tensor (CUDA)
- rescale_to_unit_interval_gpu(L_torch, normalized_hint=True) -> torch.sparse_coo_tensor (CUDA)
- build_rescaled_normalized_laplacian_gpu(A_torch, assume_symmetric=False) -> torch.sparse_coo_tensor (CUDA)
- approx_lambda_bounds_gpu_dense(L_torch, max_n=2048) -> (λ_min, λ_max)  # debug (small N only)
"""

from typing import Union
import numpy as np
import torch

# ----------------------------- CPU (scipy) -----------------------------
try:
    import scipy.sparse as sp
except Exception:
    sp = None

def _require_scipy():
    if sp is None:
        raise ImportError(
            "scipy.sparse is required for CPU Laplacian functions. "
            "Please install SciPy or use the GPU helpers."
        )

def _ensure_csr_float(A) -> "sp.csr_matrix":
    _require_scipy()
    if sp.issparse(A):
        A = A.tocsr()
        if A.dtype not in (np.float32, np.float64):
            A = A.astype(np.float32)
        return A
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency must be a square matrix.")
    return sp.csr_matrix(A.astype(np.float32, copy=False))

def _symmetrize_csr(A: "sp.csr_matrix") -> "sp.csr_matrix":
    # 0.5 * (A + A^T) without densifying
    return ((A + A.T) * 0.5).tocsr()

def graph_laplacian(
    A: Union["sp.spmatrix", np.ndarray],
    normalized: bool = True,
    assume_symmetric: bool = False
) -> "sp.csr_matrix":
    """
    CPU Laplacian (scipy.sparse CSR).

    Unnormalized:    L = D - A
    Sym. normalized: L = I_deg>0 - D^{-1/2} A D^{-1/2}
      (diagonal is 1 only for nodes with deg>0; 0 for isolated nodes)
    """
    _require_scipy()
    A = _ensure_csr_float(A)
    if not assume_symmetric:
        A = _symmetrize_csr(A)

    N = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()

    if not normalized:
        D = sp.diags(deg, format="csr")
        return (D - A).tocsr()

    # normalized:
    nz = deg > 0
    inv_sqrt = np.zeros_like(deg, dtype=A.dtype)
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])

    # A_norm = D^{-1/2} A D^{-1/2}
    D_left  = sp.diags(inv_sqrt, format="csr")
    A_tmp   = D_left @ A
    D_right = sp.diags(inv_sqrt, format="csr")
    A_norm  = (A_tmp @ D_right).tocsr()

    # L = I_deg>0 - A_norm
    I_mask = sp.diags(nz.astype(A.dtype), format="csr")
    L = (I_mask - A_norm).tocsr()
    return L

def rescale_to_unit_interval(
    L: "sp.csr_matrix",
    normalized_hint: bool = True,
) -> "sp.csr_matrix":
    """
    Rescale Laplacian spectrum to [0, 1] for Bernstein filters (CPU).

    If normalized_hint=True (default) and L is a symmetrical normalized Laplacian
    (eigs in [0,2]), return L/2 to place spectrum in [0,1].

    If you pass a non-normalized L and normalized_hint=False, this still divides by 2
    (safe but conservative).
    """
    _require_scipy()
    if not sp.issparse(L):
        L = _ensure_csr_float(L)
    else:
        L = L.tocsr()
        if L.dtype not in (np.float32, np.float64):
            L = L.astype(np.float32)

    # For normalized Laplacian, spectrum ∈ [0,2] → L/2 ∈ [0,1]
    return (L * 0.5).tocsr()


# ----------------------------- GPU (torch.sparse COO) -----------------------------

def _ensure_coalesced_coo_gpu(A: torch.Tensor) -> torch.Tensor:
    """
    Ensure A is a CUDA sparse COO, coalesced, float32.
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")
    if A.is_sparse:
        A = A.coalesce()
        if A.dtype != torch.float32:
            A = A.to(dtype=torch.float32)
        if A.device.type != "cuda":
            A = A.to("cuda")
        return A
    # Dense -> sparse COO on GPU
    if A.device.type != "cuda":
        A = A.to("cuda")
    if A.dtype != torch.float32:
        A = A.to(dtype=torch.float32)
    return A.to_sparse_coo().coalesce()

def _symmetrize_coo_gpu(A: torch.Tensor) -> torch.Tensor:
    """
    0.5 * (A + A^T) in sparse COO (CUDA) without densifying.
    """
    A = _ensure_coalesced_coo_gpu(A)
    AT = torch.sparse_coo_tensor(
        indices=torch.stack([A.indices()[1], A.indices()[0]], dim=0),
        values=A.values(),
        size=A.size(),
        device=A.device,
        dtype=A.dtype,
    ).coalesce()
    S_idx = torch.cat([A.indices(), AT.indices()], dim=1)
    S_val = torch.cat([A.values(),  AT.values()],  dim=0)
    S = torch.sparse_coo_tensor(S_idx, S_val, size=A.size(),
                                device=A.device, dtype=A.dtype).coalesce()
    return torch.sparse_coo_tensor(S.indices(), 0.5 * S.values(), size=S.size(),
                                   device=S.device, dtype=S.dtype).coalesce()

def _degree_from_coo_gpu(A: torch.Tensor) -> torch.Tensor:
    d = torch.sparse.sum(A, dim=1).to_dense().reshape(-1)
    return d.to(torch.float32) if d.dtype != torch.float32 else d

def graph_laplacian_gpu(
    A: torch.Tensor,
    normalized: bool = True,
    assume_symmetric: bool = False
) -> torch.Tensor:
    """
    GPU Laplacian (torch.sparse COO on CUDA).

    Unnormalized:    L = D - A
    Sym. normalized: L = I_deg>0 - D^{-1/2} A D^{-1/2}
    """
    A = _ensure_coalesced_coo_gpu(A)
    if not assume_symmetric:
        A = _symmetrize_coo_gpu(A)

    N = A.size(0)
    d = _degree_from_coo_gpu(A)
    if normalized:
        inv_sqrt_d = torch.zeros_like(d)
        nz = d > 0
        inv_sqrt_d[nz] = d[nz].rsqrt()

        rows, cols = A.indices()
        vals = A.values() * inv_sqrt_d[rows] * inv_sqrt_d[cols]

        # off-diagonal part: -A_norm
        L_off = torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            -vals,
            size=(N, N),
            device=A.device,
            dtype=A.dtype
        ).coalesce()

        # diagonal: 1 for deg>0, else 0
        diag_idx = torch.arange(N, device=A.device, dtype=torch.long)
        L_diag = torch.sparse_coo_tensor(
            torch.stack([diag_idx, diag_idx], dim=0),
            nz.to(A.dtype),
            size=(N, N),
            device=A.device,
            dtype=A.dtype
        ).coalesce()

        L = torch.sparse_coo_tensor(
            torch.cat([L_diag.indices(), L_off.indices()], dim=1),
            torch.cat([L_diag.values(),  L_off.values()],  dim=0),
            size=(N, N),
            device=A.device,
            dtype=A.dtype
        ).coalesce()
        return L

    # Unnormalized
    diag_idx = torch.arange(N, device=A.device, dtype=torch.long)
    D = torch.sparse_coo_tensor(
        torch.stack([diag_idx, diag_idx], dim=0),
        d,
        size=(N, N),
        device=A.device,
        dtype=A.dtype
    ).coalesce()

    L = torch.sparse_coo_tensor(
        torch.cat([D.indices(), A.indices()], dim=1),
        torch.cat([D.values(),  -A.values()], dim=0),
        size=(N, N),
        device=A.device,
        dtype=A.dtype
    ).coalesce()
    return L

def rescale_to_unit_interval_gpu(
    L: torch.Tensor,
    normalized_hint: bool = True
) -> torch.Tensor:
    """
    Spectral scaling for Bernstein on GPU.
    For normalized Laplacian, returns L/2 so spectrum ∈ [0,1].
    """
    L = _ensure_coalesced_coo_gpu(L)
    scale = 2.0  # normalized Laplacian has eigs in [0,2]
    return torch.sparse_coo_tensor(
        L.indices(), (L.values() / scale), size=L.size(),
        device=L.device, dtype=L.dtype
    ).coalesce()

def build_rescaled_normalized_laplacian_gpu(
    A: torch.Tensor,
    assume_symmetric: bool = False
) -> torch.Tensor:
    """
    A (CUDA) -> L_sym (CUDA sparse COO) -> L_tilde = L_sym/2
    """
    L_sym = graph_laplacian_gpu(A, normalized=True, assume_symmetric=assume_symmetric)
    return rescale_to_unit_interval_gpu(L_sym, normalized_hint=True)

@torch.no_grad()
def approx_lambda_bounds_gpu_dense(L: torch.Tensor, max_n: int = 2048):
    """
    Debug helper (small N only): convert to dense on GPU and return (λ_min, λ_max).
    Do NOT use for large graphs.
    """
    L = _ensure_coalesced_coo_gpu(L)
    N = L.size(0)
    if N > max_n:
        return float("nan"), float("nan")
    vals = torch.linalg.eigvalsh(L.to_dense())
    return float(vals.min().item()), float(vals.max().item())
