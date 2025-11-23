#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
layermodel.py

Models:
  • Graph_GCN      — DB-GCN (Bernstein polynomial spectral) + MLP, streams: {"fusion","gcn","mlp"}
  • ChebNet        — Chebyshev spectral GCN (expects L rescaled to [-1,1])
  • BernNetHead    — Bernstein spectral head (expects L rescaled to [0,1])
  • APPNPHead      — APPNP (needs symmetric-normalized adjacency S = D^{-1/2} A D^{-1/2})
  • GraphSAGEHead  — Mean-aggregator SAGE (adds self-loops & row-normalizes internally)

All forward(...) return a 4-tuple:
  (x_decode_gae_or_None, x_hidden_gae, logits_or_probs, 0)
Logits are raw scores; use return_probs=True on Graph_GCN or .predict_proba() helpers.
"""

from typing import Union, Optional, Sequence, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, Sequence[float], np.ndarray]

# (Kept for compatibility if you use it elsewhere)
try:
    from utilsdata import sparse_mx_to_torch_sparse_tensor  # noqa: F401
except Exception:
    pass


# ============================================================
#                   DB-GCN (Bernstein) + MLP
# ============================================================
class Graph_GCN(nn.Module):
    """
    DB-GCN: Bernstein-polynomial spectral GCN + MLP dual-stream with ablations.
    stream_mode in {"fusion","gcn","mlp"}.

    net_parameters = [F_0, D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
    """

    def __init__(self, net_parameters, stream_mode: str = "fusion"):
        super().__init__()
        print(f'Graph_GCN initialized (stream_mode="{stream_mode}")')

        # Unpack
        F_0, D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim = net_parameters
        self.F_0, self.D_g     = int(F_0), int(D_g)
        self.CL1_F, self.CL1_K = int(CL1_F), int(CL1_K)
        self.FC2_F             = int(FC2_F)
        self.stream_mode       = str(stream_mode).lower()
        self.out_dim           = int(out_dim)

        # Keep more nodes before FC head (helps rare classes; deterministic pooling unchanged)
        self.pool_out_nodes = int(min(self.D_g, max(160, self.D_g // 2)))

        if (self.D_g % self.pool_out_nodes) == 0:
            k = self.D_g // self.pool_out_nodes
            self._pool_kind = "avg"
            self.avg_pool = nn.AvgPool1d(kernel_size=k, stride=k, ceil_mode=False, count_include_pad=False)
        else:
            self._pool_kind = "adaptive_det"
            self.avg_pool = None
        self.FC1Fin = self.CL1_F * self.pool_out_nodes  # flattened pooled graph features

        # ----- Graph stream -----
        self.cl1 = nn.Linear(self.F_0, self.CL1_F)
        self.fc1 = nn.Linear(self.FC1Fin, FC1_F)
        self.fc2 = nn.Linear(FC1_F, self.FC2_F) if self.FC2_F > 0 else None
        self.fc3 = nn.Linear(self.FC2_F, self.D_g * self.F_0) if self.FC2_F > 0 else None  # optional decoder

        # ----- MLP stream (omics dense path) -----
        self.nn_fc1 = nn.Linear(self.D_g * self.F_0, NN_FC1)
        self.nn_bn1 = nn.BatchNorm1d(NN_FC1)
        self.nn_fc2 = nn.Linear(NN_FC1, NN_FC2)
        self.nn_bn2 = nn.BatchNorm1d(NN_FC2)

        # Fusion gate to prevent MLP from dominating early (sigmoid(-1)≈0.27)
        self.mlp_gate = nn.Parameter(torch.tensor(-1.0))

        # Light channel-dropout on raw features (affects MLP more than GCN)
        self.input_feat_dropout_p = 0.05  # small; safe for omics

        # ----- Heads -----
        self.fc_head_gcn = nn.Linear(FC1_F, out_dim)          # for "gcn"
        self.fc_head_mlp = nn.Linear(NN_FC2, out_dim)         # for "mlp"
        self.FC_sum2     = nn.Linear(FC1_F + NN_FC2, out_dim) # for "fusion"

        # Cache for Bernstein coefficients a_j(θ, K)
        self._bern_cache = {}

        self.apply(self._weight_init)

    # --------------------------- init ---------------------------
    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # --------------------------- utils ---------------------------
    def set_stream_mode(self, mode: str):
        mode = mode.lower()
        assert mode in ("gcn", "mlp", "fusion")
        self.stream_mode = mode
        print(f"[Graph_GCN] stream_mode -> {self.stream_mode}")

    @torch.no_grad()
    def _bern_coeffs(self, K: int, theta_t: torch.Tensor) -> torch.Tensor:
        """
        Precompute coefficients a_j so that:
          sum_{k=0}^K θ_k B_k(L) X  ==  sum_{j=0}^K a_j L^j X
        where B_k(L) = (2I - L)^{K-k} L^k
        """
        key = (int(K), tuple(map(float, theta_t.detach().cpu().tolist())))
        cached = self._bern_cache.get(key)
        if cached is not None:
            return cached  # CPU tensor
        K = int(K)
        theta = list(map(float, key[1]))
        a = [0.0] * (K + 1)
        pow2 = [2.0 ** (K - j) for j in range(K + 1)]
        for j in range(K + 1):
            s = 0.0
            for k in range(j + 1):
                s += theta[k] * math.comb(K - k, j - k) * pow2[j] * ((-1.0) ** (j - k))
            a[j] = s
        a_t_cpu = torch.tensor(a, dtype=torch.float32)  # keep cache on CPU
        self._bern_cache[key] = a_t_cpu
        return a_t_cpu

    # -------- deterministic pooling helper --------
    def _pool1d_det(self, x_ch_last: torch.Tensor) -> torch.Tensor:
        """
        Deterministic 1D downsampling by average over contiguous segments.
        Works under torch.use_deterministic_algorithms(True) by computing cumsum on CPU
        when necessary. x_ch_last: (B, C, G)
        """
        B, C, G = x_ch_last.shape
        T = int(self.pool_out_nodes)
        if G == T:
            return x_ch_last

        # If divisible, use AvgPool1d (already deterministic)
        if self._pool_kind == "avg" and self.avg_pool is not None:
            return self.avg_pool(x_ch_last)

        device = x_ch_last.device
        need_cpu_cumsum = (x_ch_last.is_cuda and torch.are_deterministic_algorithms_enabled())

        # prefix sums
        if need_cpu_cumsum:
            x_work = x_ch_last.to("cpu")
            prefix = F.pad(x_work, (1, 0)).cumsum(dim=2)   # (B, C, G+1) on CPU
            idx_device = "cpu"
        else:
            x_work = x_ch_last
            prefix = F.pad(x_work, (1, 0)).cumsum(dim=2)   # (B, C, G+1) on current device
            idx_device = device

        # integer bin edges 0..G
        edges = torch.linspace(0, G, steps=T + 1, dtype=torch.int64, device=idx_device)
        edges = torch.clamp(edges, 0, G)
        left_idx, right_idx = edges[:-1], edges[1:]

        seg_sum = prefix.index_select(2, right_idx) - prefix.index_select(2, left_idx)  # (B, C, T)
        seg_len = (right_idx - left_idx).view(1, 1, T).clamp_min(1)
        out = seg_sum / seg_len

        if need_cpu_cumsum:
            out = out.to(device)
        return out

    # --------------------------- core ----------------------------
    def forward(
        self,
        x_in: torch.Tensor,
        dropout_p: float,
        L: Union[torch.Tensor, Sequence[torch.Tensor]],
        theta: TensorLike,
        *,
        return_probs: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, int]:
        """
        Returns:
          x_decode_gae : optional reconstruction (None if decoder disabled)
          x_hidden_gae : graph hidden representation after fc1
          out          : logits (training) or probabilities if return_probs=True
          0            : placeholder (API compatibility)
        """
        # ---------- shape normalize ----------
        if x_in.ndim == 2:
            if x_in.shape[1] != self.D_g * self.F_0:
                raise ValueError(f"Expected features {self.D_g * self.F_0}, got {x_in.shape[1]}")
            x = x_in.view(x_in.shape[0], self.D_g, self.F_0)
        elif x_in.ndim == 3:
            x = x_in
            _, V, F0 = x.shape
            if V != self.D_g:
                raise ValueError(f"Expected V={self.D_g}, got {V}")
            if F0 != self.F_0:
                raise ValueError(f"Expected F0={self.F_0}, got {F0}")
        else:
            raise ValueError(f"Bad input shape {tuple(x_in.shape)}; expected (B,V,F0) or (B,V*F0)")

        # --- very light feature dropout at input (per sample, per gene-channel)
        if self.training and self.input_feat_dropout_p > 0:
            mask = (torch.rand((x.shape[0], 1, x.shape[2]), device=x.device) >
                    self.input_feat_dropout_p)
            x = x * mask

        # ---------- MLP-only ----------
        if self.stream_mode == "mlp":
            x_nn = x.reshape(x.size(0), -1)
            x_nn = self.nn_fc1(x_nn); x_nn = self.nn_bn1(x_nn); x_nn = F.relu(x_nn)
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            x_nn = self.nn_fc2(x_nn); x_nn = self.nn_bn2(x_nn); x_nn = F.relu(x_nn)
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            logits = self.fc_head_mlp(x_nn)
            out = F.softmax(logits, dim=1) if return_probs else logits
            return None, x_nn, out, 0

        # ---------- GCN (for "gcn" and "fusion") ----------
        if L is None or (isinstance(L, (list, tuple)) and len(L) == 0):
            raise ValueError("Laplacian L is required for stream_mode 'gcn' or 'fusion'.")
        if not isinstance(theta, (list, tuple, torch.Tensor)):
            raise ValueError("theta must be list/1D tensor when using GCN stream.")

        theta_t = torch.as_tensor(theta, dtype=torch.float32, device=x.device)
        if theta_t.numel() != self.CL1_K + 1:
            raise ValueError(f"Theta length mismatch: expected {self.CL1_K+1}, got {theta_t.numel()}")

        # Optional numeric guard: expect rescaled normalized Laplacian in [0,1]
        if __debug__:
            L0 = L[0] if isinstance(L, (list, tuple)) else L
            if isinstance(L0, torch.Tensor) and (not L0.is_sparse):
                try:
                    Lmin = float(L0.min()); Lmax = float(L0.max())
                    if not (Lmin >= -1e-6 and Lmax <= 1.0001):
                        raise ValueError("Expected rescaled normalized Laplacian with entries in [0,1].")
                except Exception:
                    pass

        x_g = self.graph_conv_bernstein(x, self.cl1, L[0] if isinstance(L, (list, tuple)) else L,
                                        self.CL1_F, self.CL1_K, theta_t)
        x_g = F.relu(x_g)

        # ---- Deterministic pooling over nodes -> fixed-size vector per sample ----
        # (B, V, CL1_F) -> (B, CL1_F, V) -> pool -> (B, CL1_F, P) -> (B, P, CL1_F)
        x_g = x_g.permute(0, 2, 1)
        x_g = self._pool1d_det(x_g)
        x_g = x_g.permute(0, 2, 1)

        x_g = x_g.reshape(x_g.size(0), -1)
        x_hidden_gae = self.fc1(x_g); x_hidden_gae = F.relu(x_hidden_gae)
        x_hidden_gae = F.dropout(x_hidden_gae, p=dropout_p, training=self.training)

        # Optional decoder branch (reconstruction)
        x_decode_gae = None
        if self.fc2 is not None and self.fc3 is not None:
            x_decode_gae = self.fc2(x_hidden_gae); x_decode_gae = F.relu(x_decode_gae)
            x_decode_gae = F.dropout(x_decode_gae, p=dropout_p, training=self.training)
            x_decode_gae = self.fc3(x_decode_gae)

        if self.stream_mode == "gcn":
            logits = self.fc_head_gcn(x_hidden_gae)
            out = F.softmax(logits, dim=1) if return_probs else logits
            return x_decode_gae, x_hidden_gae, out, 0

        # ---------- Fusion ----------
        x_nn = x.reshape(x.size(0), -1)
        x_nn = self.nn_fc1(x_nn); x_nn = self.nn_bn1(x_nn); x_nn = F.relu(x_nn)
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
        x_nn = self.nn_fc2(x_nn); x_nn = self.nn_bn2(x_nn); x_nn = F.relu(x_nn)
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)

        # learnable gate in (0,1); starts low so GCN leads, grows if MLP helps
        gate = torch.sigmoid(self.mlp_gate)
        x_nn = gate * x_nn

        x_cat = torch.cat([x_hidden_gae, x_nn], dim=1)
        logits = self.FC_sum2(x_cat)
        out = F.softmax(logits, dim=1) if return_probs else logits
        return x_decode_gae, x_hidden_gae, out, 0

    def graph_conv_bernstein(
        self,
        x: torch.Tensor,
        cl: nn.Linear,
        L: torch.Tensor,
        Fout: int,
        K: int,
        theta_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply sum_j a_j * L^j to x, where a_j are expanded from θ_k and Bernstein B_k,
        then normalize and apply a per-node linear map cl: R^{F0} -> R^{Fout}.
        """
        device = x.device
        a_t = self._bern_coeffs(K, theta_t).to(device)

        def L_apply_dense(y, Ld):
            # y: (B, V, F0); Ld: (V, V) -> (B, V, F0)
            return torch.einsum("vw,bwf->bvf", Ld, y)

        def L_apply_sparse(y, Ls):
            # y: (B, V, F0); Ls sparse (V,V)
            B, V, F0 = y.shape
            y2 = y.permute(0, 2, 1).reshape(B * F0, V)   # (B*F0, V)
            z2 = torch.sparse.mm(Ls, y2.t()).t()         # (B*F0, V)
            return z2.reshape(B, F0, V).permute(0, 2, 1) # (B, V, F0)

        # Choose dense/sparse path (expects L already on same device)
        if isinstance(L, torch.Tensor):
            if getattr(L, "layout", None) == torch.sparse_csr:
                Ls = L.to_sparse_coo().coalesce()
                L_apply = lambda y: L_apply_sparse(y, Ls)
            elif L.is_sparse:
                Ls = L.coalesce()
                L_apply = lambda y: L_apply_sparse(y, Ls)
            else:
                Ld = L
                L_apply = lambda y: L_apply_dense(y, Ld)
        else:
            Ld = torch.tensor(L, dtype=torch.float32, device=device)
            L_apply = lambda y: L_apply_dense(y, Ld)

        # Accumulate a_0*X + a_1*L X + ... + a_K*L^K X
        if K == 0:
            out = a_t[0] * x
        else:
            out = a_t[0] * x
            w = x
            for j in range(1, K + 1):
                w = L_apply(w)
                out = out + a_t[j] * w

        # Normalize across nodes, then per-node linear projection
        out = F.normalize(out, p=2, dim=1, eps=1e-12)
        out = cl(out)  # (B, V, Fout)
        return out

    # --------------------- convenience api -----------------------
    @torch.no_grad()
    def predict_proba(
        self,
        x_in: torch.Tensor,
        L: Union[torch.Tensor, Sequence[torch.Tensor]],
        theta: TensorLike,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Returns softmax probabilities P(c | X) for eval/plots."""
        self.eval()
        _, _, probs, _ = self.forward(x_in, dropout_p, L, theta, return_probs=True)
        return probs

    def loss(self, y_recon, y_recon_target, y_logits, y_true, l2_regularization, lambda_l2=0.2):
        loss_recon = 0.0
        if y_recon is not None and y_recon_target is not None:
            loss_recon = nn.MSELoss()(y_recon, y_recon_target.reshape(y_recon.shape))
        loss_cls = nn.CrossEntropyLoss()(y_logits, y_true)
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return loss_recon + loss_cls + lambda_l2 * l2_regularization * l2_loss


# ============================================================
#                        Chebyshev GCN
# ============================================================
class ChebNet(nn.Module):
    """
    Chebyshev spectral GCN:
      H = sum_{k=0..K} T_k(L_tilde) X W_k, with L_tilde in [-1, 1].
    Reuses your MLP/fusion heads for fair comparison.

    NOTE: Make sure you rescale L to the Chebyshev domain [-1,1]
          before passing it in (do this in coarsening.py).
    """
    def __init__(self, net_parameters, stream_mode: str = "fusion"):
        super().__init__()
        F_0, D_g, CL1_F, CL1_K, FC1_F, _FC2_F_unused, NN_FC1, NN_FC2, out_dim = net_parameters
        self.F_0, self.D_g, self.K = int(F_0), int(D_g), int(CL1_K)
        self.poolsize = 8
        self.stream_mode = stream_mode.lower()

        # Chebyshev weights (per order)
        self.W = nn.ParameterList([nn.Parameter(torch.empty(self.F_0, CL1_F)) for _ in range(self.K + 1)])
        for w in self.W:
            nn.init.xavier_uniform_(w)

        # readout + MLP heads reuse your design
        self.fc1 = nn.Linear(CL1_F * max(1, (self.D_g // self.poolsize)), FC1_F)
        self.nn_fc1 = nn.Linear(self.D_g * self.F_0, NN_FC1)
        self.nn_fc2 = nn.Linear(NN_FC1, NN_FC2)
        self.fc_head_gcn = nn.Linear(FC1_F, out_dim)
        self.fc_head_mlp = nn.Linear(NN_FC2, out_dim)
        self.FC_sum2     = nn.Linear(FC1_F + NN_FC2, out_dim)

    def _cheb_stack(self, X, L_tilde):
        # X: (B,V,F0), L_tilde: (V,V) dense
        def L_apply(Y):  # (B,V,F)
            return torch.einsum('vw,bwf->bvf', L_tilde, Y)

        T0 = X
        outs = [T0 @ self.W[0]]
        if self.K == 0:
            return outs[0]
        T1 = L_apply(X)
        outs.append(T1 @ self.W[1])
        Tk_1, Tk = T0, T1
        for k in range(2, self.K + 1):
            Tk1 = 2.0 * L_apply(Tk) - Tk_1
            outs.append(Tk1 @ self.W[k])
            Tk_1, Tk = Tk, Tk1
        H = torch.stack(outs, dim=0).sum(dim=0)  # (B,V,CL1_F)
        return H

    @staticmethod
    def _max_pool(x, p):
        if p > 1:
            x = x.permute(0, 2, 1)
            x = nn.MaxPool1d(p)(x)
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x_in, dropout_p, L, _theta_unused):
        # shape normalize (same as Graph_GCN)
        if x_in.ndim == 2:
            B = x_in.shape[0]
            x = x_in.view(B, self.D_g, self.F_0)
        elif x_in.ndim == 3:
            x = x_in
        else:
            raise ValueError("x_in must be (B,V,F0) or (B,V*F0)")

        if self.stream_mode == "mlp":
            x_nn = torch.relu(self.nn_fc1(x.reshape(x.size(0), -1)))
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            x_nn = torch.relu(self.nn_fc2(x_nn))
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            return None, x_nn, self.fc_head_mlp(x_nn), 0

        # Chebyshev on L_rescaled (expect L already scaled to [-1,1])
        if not isinstance(L, (list, tuple)) or len(L) == 0:
            raise ValueError("Laplacian required for ChebNet.")
        Ld = L[0]
        if isinstance(Ld, torch.Tensor) and Ld.is_sparse:
            Ld = Ld.to_dense()
        elif not isinstance(Ld, torch.Tensor):
            Ld = torch.tensor(Ld, dtype=torch.float32, device=x.device)
        else:
            Ld = Ld.to(x.device)

        H = torch.relu(self._cheb_stack(x, Ld))
        H = self._max_pool(H, self.poolsize).reshape(x.size(0), -1)
        H = F.dropout(torch.relu(self.fc1(H)), p=dropout_p, training=self.training)

        if self.stream_mode == "gcn":
            return None, H, self.fc_head_gcn(H), 0

        # fusion
        x_nn = torch.relu(self.nn_fc1(x.reshape(x.size(0), -1)))
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
        x_nn = torch.relu(self.nn_fc2(x_nn))
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
        logits = self.FC_sum2(torch.cat([H, x_nn], dim=1))
        return None, H, logits, 0

    @torch.no_grad()
    def predict_proba(self, x_in, L, theta=None, dropout_p=0.0):
        self.eval()
        _, _, logits, _ = self.forward(x_in, dropout_p, L, theta)
        return logits.softmax(dim=1)


# ============================================================
#                     BernNet (spectral head)
# ============================================================
class BernNetHead(nn.Module):
    """
    Bernstein spectral filter head (graph-level classification).

    g(L)X = sum_{i=0}^K theta_i * C(K,i) * L^i * (I - L)^{K-i} X
    Assumes L has been rescaled to [0,1] upstream.
    """
    def __init__(self, D_g, F_0, hidden, out_dim, K, stream_mode="fusion", dropout=0.0):
        super().__init__()
        self.D_g, self.F_0 = int(D_g), int(F_0)
        self.stream_mode = str(stream_mode).lower()
        self.K = int(K)
        self.dropout = float(dropout)

        # per-node MLP before propagation
        self.fc1 = nn.Linear(self.F_0, hidden)
        # readout -> graph-level logits
        self.fc2 = nn.Linear(hidden, out_dim)

        # learnable Bernstein coefficients
        self.theta = nn.Parameter(torch.zeros(self.K + 1))
        nn.init.normal_(self.theta, mean=0.0, std=0.02)

        # binomial coefficients
        combs = [math.comb(self.K, i) for i in range(self.K + 1)]
        self.register_buffer("binom", torch.tensor(combs, dtype=torch.float32))

    @staticmethod
    def _mat_apply(L, Y):
        # L: (N,N), Y: (B,N,F) -> (B,N,F)
        return torch.einsum('vw,bwf->bvf', L, Y)

    def _bern_filter(self, X, L):
        """
        X: (B, N, F) | L: (N, N)
        Returns: (B, N, F)
        """
        device = X.device
        if not isinstance(L, torch.Tensor):
            L = torch.tensor(L, dtype=torch.float32, device=device)
        else:
            L = L.to(device)
            if L.is_sparse:
                L = L.to_dense()

        B, N, F = X.shape
        I = torch.eye(N, device=device, dtype=X.dtype)
        M = I - L  # also in [0,1]

        # precompute powers up to K
        L_pows = [I]
        for _ in range(1, self.K + 1):
            L_pows.append(L_pows[-1] @ L)

        M_pows = [I]
        for _ in range(1, self.K + 1):
            M_pows.append(M_pows[-1] @ M)

        out = torch.zeros_like(X)
        # correct weighted sum: sum_i theta[i] * C(K,i) * (L^i) * (I-L)^(K-i) * X
        for i in range(self.K + 1):
            Bi = self.binom[i] * (L_pows[i] @ M_pows[self.K - i])
            out = out + self.theta[i] * self._mat_apply(Bi, X)
        return out

    def forward(self, X, dp, L_list=None, _theta_unused=None, A_dense=None):
        # normalize to (B,N,F)
        if X.dim() == 2:
            X = X.view(X.size(0), self.D_g, self.F_0)

        # per-node feature lift
        H = F.relu(self.fc1(F.dropout(X, p=dp, training=self.training)))

        # spectral propagation if L is provided
        if L_list is not None and len(L_list) > 0 and L_list[0] is not None:
            L = L_list[0]
            H = self._bern_filter(H, L)

        # node -> graph readout (mean pool) then classify
        H = F.dropout(H, p=dp, training=self.training)
        Hg = H.mean(dim=1)                  # (B, hidden)
        logits = self.fc2(Hg)               # (B, out_dim)
        return None, Hg, logits, 0

    @torch.no_grad()
    def predict_proba(self, X, L_list=None, theta=None, dropout_p=0.0):
        self.eval()
        _, _, logits, _ = self.forward(X, dropout_p, L_list, theta)
        return logits.softmax(dim=1)


# ============================================================
#                          APPNP
# ============================================================
class APPNPHead(nn.Module):
    """
    Predict-then-Propagate (K steps with teleport α) on feature space
    for per-sample signals on a common gene graph.

    Expectations:
      • A_dense is ALREADY symmetric-normalized (S = D^{-1/2} A D^{-1/2})
      • We keep the logits equal to the MLP head (Z0) for parity with DB-GCN MLP stream.
    """
    def __init__(self, D_g, F_0, hidden=64, out_dim=2, alpha=0.1, K=10, stream_mode="fusion"):
        super().__init__()
        self.alpha = float(alpha)
        self.K = int(K)
        self.stream_mode = str(stream_mode).lower()
        self.D_g, self.F_0 = int(D_g), int(F_0)

        # Simple MLP head (same spirit as your MLP stream)
        self.lin1 = nn.Linear(self.D_g * self.F_0, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)

    def forward(self, x_in, dropout_p, _L_unused=None, _theta_unused=None, A_dense=None):
        """
        x_in:    (B, V, F0) or (B, V*F0)
        A_dense: (V, V) symmetric-normalized adjacency (torch.FloatTensor) on same device as x_in
        """
        # ---- shape normalize ----
        if x_in.ndim == 2:
            B = x_in.shape[0]
            x = x_in.view(B, self.D_g, self.F_0)
        elif x_in.ndim == 3:
            x = x_in
            if x.size(1) != self.D_g or x.size(2) != self.F_0:
                raise ValueError(f"Expected x_in (B,{self.D_g},{self.F_0}); got {tuple(x.shape)}")
        else:
            raise ValueError("x_in must be (B,V,F0) or (B,V*F0)")

        # ---- MLP path (base logits) ----
        H0 = F.dropout(F.relu(self.lin1(x.reshape(x.size(0), -1))),
                       p=dropout_p, training=self.training)
        Z0 = self.lin2(H0)  # [B, C]

        if self.stream_mode == "mlp":
            return None, H0, Z0, 0  # identical semantics

        # ---- Graph propagation on shared, pre-normalized S ----
        if A_dense is None:
            raise ValueError("APPNPHead requires A_dense (pre-normalized dense adjacency).")
        if not isinstance(A_dense, torch.Tensor):
            A_dense = torch.tensor(A_dense, dtype=torch.float32, device=x.device)
        else:
            A_dense = A_dense.to(x.device, dtype=torch.float32)

        S = A_dense  # [V, V]  (symmetric-normalized)

        # Personalized PageRank style propagation on feature space
        X0 = x.mean(dim=0)  # [V, F0] (batch-mean feature as restart)
        X  = X0
        for _ in range(self.K):
            X = (1.0 - self.alpha) * (S @ X) + self.alpha * X0

        # (Available for future fusion, if desired)
        _Xg = X.mean(dim=0, keepdim=True).repeat(x.size(0), 1)  # [B, F0]

        # Keep current protocol: classification uses the MLP logits
        return None, H0, Z0, 0

    @torch.no_grad()
    def predict_proba(self, x_in, A_dense, dropout_p=0.0):
        self.eval()
        _, _, logits, _ = self.forward(x_in, dropout_p, None, None, A_dense=A_dense)
        return logits.softmax(dim=1)


# ============================================================
#                         GraphSAGE (mean)
# ============================================================
class GraphSAGEHead(nn.Module):
    """
    Two-layer GraphSAGE (mean aggregator), PyG-free.

    H^1 = σ( mean_aggr(Ã, X) @ W1 )
    H^2 = σ( mean_aggr(Ã, H^1) @ W2 )
    where Ã has self-loops and is row-normalized.

    Works with per-sample signals on the same topology.
    Provide dense adjacency A_dense (V,V). Self-loops are added inside.
    """
    def __init__(self, D_g, F_0, hidden=64, out_dim=2, stream_mode="fusion"):
        super().__init__()
        self.D_g, self.F_0 = int(D_g), int(F_0)
        self.hidden, self.out_dim = int(hidden), int(out_dim)
        self.stream_mode = stream_mode.lower()

        self.W1 = nn.Linear(self.F_0, self.hidden, bias=True)
        self.W2 = nn.Linear(self.hidden, self.hidden, bias=True)

        self.poolsize = 8
        # Readout + MLP heads (parity with others)
        self.fc1 = nn.Linear(self.hidden * max(1, (self.D_g // self.poolsize)), self.hidden)
        self.fc_head_gcn = nn.Linear(self.hidden, self.out_dim)

        self.nn_fc1 = nn.Linear(self.D_g * self.F_0, self.hidden)
        self.nn_fc2 = nn.Linear(self.hidden, self.hidden)
        self.FC_sum2 = nn.Linear(self.hidden + self.hidden, self.out_dim)

    @staticmethod
    def _row_norm_with_self_loops(A_dense: torch.Tensor) -> torch.Tensor:
        V = A_dense.size(0)
        A = A_dense + torch.eye(V, device=A_dense.device, dtype=A_dense.dtype)
        deg = A.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return A / deg  # row-normalized mean aggregator

    @staticmethod
    def _max_pool(x, p):
        if p > 1:
            x = x.permute(0, 2, 1)
            x = nn.MaxPool1d(p)(x)
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x_in, dropout_p, _L_unused, _theta_unused, A_dense=None):
        """
        x_in:    (B,V,F0) or (B, V*F0)
        A_dense: (V,V) dense adjacency (no self-loops required; added inside)
        """
        if x_in.ndim == 2:
            B = x_in.shape[0]
            x = x_in.view(B, self.D_g, self.F_0)
        elif x_in.ndim == 3:
            x = x_in
        else:
            raise ValueError("x_in must be (B,V,F0) or (B,V*F0)")

        if self.stream_mode == "mlp":
            # mirror MLP path for completeness
            x_nn = F.relu(self.nn_fc1(x.reshape(x.size(0), -1)))
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            x_nn = F.relu(self.nn_fc2(x_nn))
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            logits = self.FC_sum2(torch.cat([x_nn, x_nn], dim=1))  # trivial fusion
            return None, x_nn, logits, 0

        if A_dense is None:
            raise ValueError("GraphSAGEHead requires A_dense (dense adjacency).")
        if not isinstance(A_dense, torch.Tensor):
            A_dense = torch.tensor(A_dense, dtype=torch.float32, device=x.device)
        else:
            A_dense = A_dense.to(x.device)

        S = self._row_norm_with_self_loops(A_dense)  # (V,V)

        def prop(Y):  # Y: (B,V,F)
            return torch.einsum('vw,bwf->bvf', S, Y)

        # Layer 1
        H1 = F.relu(self.W1(prop(x)))
        # Layer 2
        H2 = F.relu(self.W2(prop(H1)))

        # Graph branch readout
        Hg = self._max_pool(H2, self.poolsize).reshape(x.size(0), -1)
        Hg = F.dropout(F.relu(self.fc1(Hg)), p=dropout_p, training=self.training)

        if self.stream_mode == "gcn":
            return None, Hg, self.fc_head_gcn(Hg), 0

        # Fusion with an MLP stream (same spirit as others)
        x_nn = F.relu(self.nn_fc1(x.reshape(x.size(0), -1)))
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
        x_nn = F.relu(self.nn_fc2(x_nn))
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)

        logits = self.FC_sum2(torch.cat([Hg, x_nn], dim=1))
        return None, Hg, logits, 0

    @torch.no_grad()
    def predict_proba(self, x_in, A_dense, dropout_p=0.0):
        self.eval()
        _, _, logits, _ = self.forward(x_in, dropout_p, None, None, A_dense=A_dense)
        return logits.softmax(dim=1)


# ============================================================
#                      Model factory (optional)
# ============================================================
def build_model(name: str, net_parameters, **kwargs) -> nn.Module:
    """
    Convenience factory:
      name ∈ {"dbgcn", "chebnet", "bernnet", "appnp", "sage"}
    """
    name = str(name).lower()
    if name == "dbgcn":
        return Graph_GCN(net_parameters, stream_mode=kwargs.get("stream_mode", "fusion"))
    if name == "chebnet":
        return ChebNet(net_parameters, stream_mode=kwargs.get("stream_mode", "fusion"))
    if name == "bernnet":
        F_0, D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim = net_parameters
        return BernNetHead(D_g=D_g, F_0=F_0, hidden=CL1_F, out_dim=out_dim, K=CL1_K,
                           stream_mode=kwargs.get("stream_mode", "fusion"),
                           dropout=kwargs.get("dropout", 0.0))
    if name == "appnp":
        F_0, D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim = net_parameters
        return APPNPHead(D_g=D_g, F_0=F_0, hidden=NN_FC1, out_dim=out_dim,
                         alpha=kwargs.get("alpha", 0.1),
                         K=kwargs.get("K", 10),
                         stream_mode=kwargs.get("stream_mode", "fusion"))
    if name == "sage":
        F_0, D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim = net_parameters
        return GraphSAGEHead(D_g=D_g, F_0=F_0, hidden=CL1_F, out_dim=out_dim,
                             stream_mode=kwargs.get("stream_mode", "fusion"))
    raise ValueError(f"Unknown model name: {name!r}")


__all__ = [
    "Graph_GCN",
    "ChebNet",
    "BernNetHead",
    "APPNPHead",
    "GraphSAGEHead",
    "build_model",
]
