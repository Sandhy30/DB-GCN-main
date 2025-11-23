#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import comb


class Graph_GCN(nn.Module):
    """
    Bernstein-polynomial spectral GCN + MLP dual-stream with ablations.

    stream_mode:
        - "fusion": concatenate GCN and MLP embeddings (default)
        - "gcn":    use GCN stream only
        - "mlp":    use MLP stream only (does not touch L/theta)

    Forward inputs:
        x_in : (B, V, F0) OR (B, V*F0) tensor
        dropout_p : float in [0,1]
        L : list/tuple with the Laplacian at L[0] (torch tensor, dense or sparse)
        theta : list/1D tensor of length (K+1)
    """

    def __init__(self, net_parameters, stream_mode: str = "fusion"):
        super().__init__()
        print(f'Graph_GCN initialized (stream_mode="{stream_mode}")')

        # Unpack params
        F_0, D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim = net_parameters
        self.F_0, self.D_g     = int(F_0), int(D_g)
        self.CL1_F, self.CL1_K = int(CL1_F), int(CL1_K)
        self.FC2_F             = int(FC2_F)
        self.poolsize          = 8
        self.FC1Fin            = self.CL1_F * max(1, (self.D_g // self.poolsize))
        self.stream_mode       = stream_mode.lower()

        # ----- Graph stream -----
        self.cl1 = nn.Linear(self.F_0, self.CL1_F)
        self.fc1 = nn.Linear(self.FC1Fin, FC1_F)
        self.fc2 = nn.Linear(FC1_F, FC2_F) if self.FC2_F > 0 else None
        self.fc3 = nn.Linear(FC2_F, self.D_g * self.F_0) if self.FC2_F > 0 else None  # optional decoder

        # ----- MLP stream (stabilized with BatchNorm) -----
        self.nn_fc1 = nn.Linear(self.D_g * self.F_0, NN_FC1)
        self.nn_bn1 = nn.BatchNorm1d(NN_FC1)
        self.nn_fc2 = nn.Linear(NN_FC1, NN_FC2)
        self.nn_bn2 = nn.BatchNorm1d(NN_FC2)

        # ----- Heads -----
        self.fc_head_gcn = nn.Linear(FC1_F, out_dim)          # for "gcn"
        self.fc_head_mlp = nn.Linear(NN_FC2, out_dim)         # for "mlp"
        self.FC_sum2     = nn.Linear(FC1_F + NN_FC2, out_dim) # for "fusion"

        # Cache for Bernstein coefficients: key=(K, tuple(theta)) -> a_t (CPU tensor)
        self._bern_cache = {}

    def set_stream_mode(self, mode: str):
        mode = mode.lower()
        assert mode in ("gcn", "mlp", "fusion")
        self.stream_mode = mode
        print(f"[Graph_GCN] stream_mode -> {self.stream_mode}")

    def forward(self, x_in, dropout_p, L, theta):
        """
        x_in:  (B, V, F0) or (B, V*F0)
        L:     list/tuple with L[0] Laplacian (torch.Tensor; sparse or dense)
        theta: list/1D tensor with length K+1 (only used if stream != "mlp")
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

        # ---------- MLP-only ----------
        if self.stream_mode == "mlp":
            x_nn = x.reshape(x.size(0), -1)
            x_nn = F.relu(self.nn_fc1(x_nn))
            x_nn = self.nn_bn1(x_nn)
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            x_nn = F.relu(self.nn_fc2(x_nn))
            x_nn = self.nn_bn2(x_nn)
            x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
            logits = self.fc_head_mlp(x_nn)
            return None, x_nn, logits, 0

        # ---------- GCN (required for "gcn" and "fusion") ----------
        if L is None or (isinstance(L, (list, tuple)) and len(L) == 0):
            raise ValueError("Laplacian L is required for stream_mode 'gcn' or 'fusion'.")
        if not isinstance(theta, (list, tuple, torch.Tensor)):
            raise ValueError("theta must be list/1D tensor when using GCN stream.")

        theta_t = torch.as_tensor(theta, dtype=torch.float32, device=x.device)
        if theta_t.numel() != self.CL1_K + 1:
            raise ValueError(f"Theta length mismatch: expected {self.CL1_K+1}, got {theta_t.numel()}")

        # Optional numeric guard (debug builds only): expect spectrum ⊆ [0,1]
        if __debug__:
            L0 = L[0]
            if isinstance(L0, torch.Tensor) and (not L0.is_sparse):
                try:
                    Lmin = float(L0.min())
                    Lmax = float(L0.max())
                    if not (Lmin >= -1e-6 and Lmax <= 1.0001):
                        raise ValueError("Expected rescaled normalized Laplacian with entries in [0,1].")
                except Exception:
                    pass

        x_g = self.graph_conv_bernstein(x, self.cl1, L[0], self.CL1_F, self.CL1_K, theta_t)
        x_g = F.relu(x_g)
        x_g = self.graph_max_pool(x_g, self.poolsize)         # (B, V/p, CL1_F)
        x_g = x_g.reshape(x_g.size(0), -1)
        x_hidden_gae = F.relu(self.fc1(x_g))
        x_hidden_gae = F.dropout(x_hidden_gae, p=dropout_p, training=self.training)

        # Optional decoder path
        x_decode_gae = None
        if self.fc2 is not None and self.fc3 is not None:
            x_decode_gae = self.fc2(x_hidden_gae)
            x_decode_gae = F.relu(x_decode_gae)
            x_decode_gae = F.dropout(x_decode_gae, p=dropout_p, training=self.training)
            x_decode_gae = self.fc3(x_decode_gae)            # (B, V*F0)

        if self.stream_mode == "gcn":
            logits = self.fc_head_gcn(x_hidden_gae)
            return x_decode_gae, x_hidden_gae, logits, 0

        # ---------- Fusion ----------
        x_nn = x.reshape(x.size(0), -1)
        x_nn = F.relu(self.nn_fc1(x_nn))
        x_nn = self.nn_bn1(x_nn)
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)
        x_nn = F.relu(self.nn_fc2(x_nn))
        x_nn = self.nn_bn2(x_nn)
        x_nn = F.dropout(x_nn, p=dropout_p, training=self.training)

        x_cat = torch.cat([x_hidden_gae, x_nn], dim=1)
        logits = self.FC_sum2(x_cat)
        return x_decode_gae, x_hidden_gae, logits, 0

    @torch.no_grad()
    def _bern_coeffs(self, K: int, theta_t: torch.Tensor) -> torch.Tensor:
        """
        Compute Bernstein coefficients a_j so that
          sum_{k=0..K} θ_k (2I - L)^{K-k} L^k x  ==  sum_{j=0..K} a_j L^j x
        Cached per (K, theta). Cache lives on CPU; moved to device on use.
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
                s += theta[k] * comb(K - k, j - k) * pow2[j] * ((-1.0) ** (j - k))
            a[j] = s
        a_t_cpu = torch.tensor(a, dtype=torch.float32)  # keep cache on CPU
        self._bern_cache[key] = a_t_cpu
        return a_t_cpu

    def graph_conv_bernstein(self, x, cl, L, Fout, K, theta_t):
        """
        Efficient evaluation of ∑_{j=0..K} a_j L^j x, with sparse- or dense-L support.
        x: (B, V, F0)
        L: (V, V) torch tensor; can be sparse_coo/sparse_csr or dense
        """
        device = x.device

        # Bernstein coefficients (cached on CPU; move here)
        a_t = self._bern_coeffs(K, theta_t).to(device)  # (K+1,)

        # matmul helper: (V,V) @ (B,V,F)  -> (B,V,F)
        def L_apply_dense(y, Ld):
            return torch.einsum("vw,bwf->bvf", Ld, y)

        def L_apply_sparse(y, Ls):
            B, V, F0 = y.shape
            y2 = y.permute(0, 2, 1).reshape(B * F0, V)   # (B*F, V)
            z2 = torch.sparse.mm(Ls, y2.t()).t()         # (B*F, V)
            return z2.reshape(B, F0, V).permute(0, 2, 1)  # (B, V, F)

        # robust sparse/dense selection + device move
        if isinstance(L, torch.Tensor):
            if getattr(L, "layout", None) == torch.sparse_csr:
                Ls = L.to_sparse_coo().coalesce().to(device)
                L_apply = lambda y: L_apply_sparse(y, Ls)
            elif L.is_sparse:
                Ls = L.coalesce().to(device)
                L_apply = lambda y: L_apply_sparse(y, Ls)
            else:
                Ld = L.to(device)
                L_apply = lambda y: L_apply_dense(y, Ld)
        else:
            Ld = torch.tensor(L, dtype=torch.float32, device=device)
            L_apply = lambda y: L_apply_dense(y, Ld)

        # Polynomial via iterative L applications (with K==0 fast path)
        if K == 0:
            out = a_t[0] * x
        else:
            out = a_t[0] * x
            w = x
            for j in range(1, K + 1):
                w = L_apply(w)
                out = out + a_t[j] * w

        # Normalize over vertices to stabilize (keeps parity with prior code)
        out = F.normalize(out, p=2, dim=1)
        out = cl(out)  # (B, V, Fout)
        return out

    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1)   # (B,Fin,V)
            x = nn.MaxPool1d(kernel_size=p, stride=p, ceil_mode=False)(x)  # floor(V/p)
            x = x.permute(0, 2, 1)   # (B,V/p,Fin)
        return x

    def loss(self, y_recon, y_recon_target, y_logits, y_true,
             l2_regularization: float = 0.0, lambda_l2: float = 0.2):
        """
        Cross-entropy + optional reconstruction + optional L2 weight decay.
        Set l2_regularization>0 to include L2 on all parameters.
        """
        loss_recon = 0.0
        if y_recon is not None and y_recon_target is not None:
            loss_recon = nn.MSELoss()(y_recon, y_recon_target.reshape(y_recon.shape))
        loss_cls = nn.CrossEntropyLoss()(y_logits, y_true)

        loss = loss_recon + loss_cls
        if l2_regularization and l2_regularization > 0.0:
            l2 = torch.zeros((), device=y_logits.device)
            for p in self.parameters():
                if p.requires_grad:
                    l2 = l2 + p.pow(2).sum()
            loss = loss + (lambda_l2 * float(l2_regularization)) * l2
        return loss
