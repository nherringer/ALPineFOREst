import torch
import numpy as np
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from alpfore.utils.kernel_utils import compute_kernel_matrix, gpytorch_kernel_wrapper
from typing import Union

#def nystrom_posterior_weights(kernel, inducing_points, train_X, train_y, noise_variance, jitter=1e-6):
#    device = inducing_points.device
#    dtype  = inducing_points.dtype
#    M = inducing_points.shape[0]
#
#    # K_MM and inverse sqrt
#    K_MM = compute_kernel_matrix(inducing_points, inducing_points, kernel)
#    if isinstance(K_MM, tuple): K_MM = K_MM[0]
#    K_MM = K_MM.to(device=device, dtype=dtype)
#    L_MM = torch.linalg.cholesky(K_MM + jitter * torch.eye(M, device=device, dtype=dtype))
#    L_inv = torch.linalg.solve(L_MM, torch.eye(M, device=device, dtype=dtype))
#    K_MM_inv_root = L_inv.T  # MxM
#
#    # Training features Φ = K_XM K_MM^{-1/2}  (n x M)
#    K_XM = compute_kernel_matrix(train_X, inducing_points, kernel)
#    if isinstance(K_XM, tuple): K_XM = K_XM[0]
#    Phi = K_XM.to(device=device, dtype=dtype) @ K_MM_inv_root
#
#    # ---- Ensure y is [n,1] ----
#    y = train_y.to(device=device, dtype=dtype)
#    if y.dim() == 1:            # [n] -> [n,1]
#        y = y.unsqueeze(1)
#    elif y.dim() == 2 and y.shape[0] == 1 and y.shape[1] != 1:
#        y = y.t()               # [1,n] -> [n,1]
#    # Now y is [n,1]
#    assert y.shape[0] == Phi.shape[0], f"y has {y.shape[0]} rows, Phi has {Phi.shape[0]}"
#
#    sigma2 = torch.as_tensor(noise_variance, device=device, dtype=dtype)
#
#    # A = I + (ΦᵀΦ)/σ²  (M x M)
#    A   = torch.eye(M, device=device, dtype=dtype) + (Phi.T @ Phi) / sigma2
#    L_A = torch.linalg.cholesky(A)   # lower-tri
#
#    # b = (Φᵀ y)/σ²   -> [M,1]
#    b = (Phi.T @ y) / sigma2         # [M,1]
#
#    # μ_w = A^{-1} b via Cholesky solve -> [M,1] -> [M]
#    mu_w = torch.cholesky_solve(b, L_A).squeeze(1)
#
#    return mu_w, L_A, K_MM_inv_root

def Kxz(kernel, X, Z):
    K = kernel(X, Z)
    if isinstance(K, tuple):
        K = K[0]
    if hasattr(K, "evaluate"):
        K = K.evaluate()
    return K

@torch.no_grad()
def nystrom_posterior_weights(
    kernel,
    inducing_points,   # Z: [m, d]
    train_X,           # [n, d]
    train_y,           # [n] or [n,1]
    train_Yvar=None,   # [n] or [n,1] (per-point noise) OR None
    sigma2_scalar=None,# float if homoscedastic
    batch_size=100_000,
    jitter_KMM=1e-6,
    jitter_A=1e-8,
    debug_shapes=False,
):
    device = inducing_points.device
    dtype  = inducing_points.dtype
    Z = inducing_points
    m = Z.shape[0]

    # -- K_MM and its inverse square root
    KMM = Kxz(kernel, Z, Z).to(device=device, dtype=dtype)
    KMM = KMM + jitter_KMM * torch.eye(m, device=device, dtype=dtype)
    L_MM = torch.linalg.cholesky(KMM)
    I_m  = torch.eye(m, device=device, dtype=dtype)
    L_MM_inv = torch.linalg.solve(L_MM, I_m)          # L_MM @ X = I  -> X = L_MM^{-1}
    KMM_inv_root = L_MM_inv.transpose(-2, -1)         # KMM^{-1/2} = (L_MM^{-1})^T

    # -- Accumulators A = I + Φ^T W Φ,  c = Φ^T W y
    A = I_m.clone()
    c = torch.zeros(m, device=device, dtype=dtype)

    n = train_X.shape[0]
    hetero = (train_Yvar is not None)
    if (not hetero) and (sigma2_scalar is None):
        raise ValueError("Provide train_Yvar (per-point variances) or sigma2_scalar (homoscedastic).")

    printed_debug = False

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = train_X[start:end].to(device=device, dtype=dtype)  # [B,d]
        yb = train_y[start:end].to(device=device, dtype=dtype).reshape(-1)  # [B]

        # Cross-kernel and features
        K_XM = Kxz(kernel, Xb, Z).to(device=device, dtype=dtype)
        # Force shape to [B, m]
        B = Xb.shape[0]
        if K_XM.ndim != 2:
            K_XM = K_XM.squeeze()
        if K_XM.shape == (m, B):
            K_XM = K_XM.transpose(-2, -1)
        elif K_XM.shape != (B, m):
            raise RuntimeError(f"K_XM has unexpected shape {tuple(K_XM.shape)}; expected (B={B}, m={m}) or (m, B).")
        Phi_b = K_XM @ KMM_inv_root  # [B, m]

        if hetero:
            wb = (1.0 / train_Yvar[start:end].to(device=device, dtype=dtype).reshape(-1))  # [B]
            # A += Φ_b^T (diag(wb)) Φ_b; implement as row-scaling then normal matmul
            A  = A + Phi_b.transpose(-2, -1) @ (wb[:, None] * Phi_b)
            c  = c + Phi_b.transpose(-2, -1) @ (wb * yb)
        else:
            inv_sigma2 = 1.0 / float(sigma2_scalar)
            A  = A + inv_sigma2 * (Phi_b.transpose(-2, -1) @ Phi_b)
            c  = c + inv_sigma2 * (Phi_b.transpose(-2, -1) @ yb)

        if debug_shapes and (not printed_debug):
            print(f"[DEBUG] B={B}, m={m}")
            print(f"[DEBUG] K_XM shape: {tuple(K_XM.shape)} (should be [B,m])")
            print(f"[DEBUG] Phi_b shape: {tuple(Phi_b.shape)}")
            print(f"[DEBUG] yb shape: {tuple(yb.shape)}")
            if hetero:
                print(f"[DEBUG] wb shape: {tuple(wb.shape)}")
            printed_debug = True

    # Stabilize & factorize
    A = A + jitter_A * I_m
    L_A = torch.linalg.cholesky(A)

    # Posterior mean: solve A mu_w = c
    mu_w = torch.cholesky_solve(c.unsqueeze(1), L_A).squeeze(1)

    return mu_w, L_A, KMM_inv_root

@torch.no_grad()
def run_nystrom_ts_top1_online(
    kernel,
    inducing_points,      # [m, d]
    candidate_set,        # [N, d]
    num_samples,          # S
    mu_w,                 # [m]
    L_A,                  # [m, m]
    KMM_inv_root,         # [m, m]
    batch_size=100_000,
    verbose=False,
):
    """
    Draw S weight samples from N(mu_w, A^{-1}) and pick the best candidate per sample.
    Returns: top_idxs [S] (global indices into candidate_set)
    """
    device = candidate_set.device
    dtype  = candidate_set.dtype
    Z = inducing_points

    m = Z.shape[0]
    S = int(num_samples)

    # Sample weights: w = mu_w + L_A^{-T} xi, xi ~ N(0, I_m)
    xi = torch.randn(m, S, device=device, dtype=dtype)              # [m, S]
    # Solve L_A^T U = xi  -> U = L_A^{-T} xi    (triangular solve)
    U  = torch.linalg.solve(L_A.transpose(-2, -1), xi)              # [m, S]
    W  = mu_w.reshape(-1, 1) + U                                    # [m, S]

    # Track per-sample best value and index
    top_vals = torch.full((S,), float("-inf"), device=device, dtype=dtype)
    top_idxs = torch.full((S,), -1, dtype=torch.long, device=device)

    N = candidate_set.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if verbose:
            print(f"[TS] candidates {start:,}..{end:,}")

        Xb = candidate_set[start:end].to(device=device, dtype=dtype)      # [B, d]
        K_XM = Kxz(kernel, Xb, Z)                                         # [B, m]
        Phi_b = K_XM @ KMM_inv_root                                       # [B, m]
        scores = Phi_b @ W                                                # [B, S]

        # For each sample (column), get the local max and index
        max_vals, max_local = scores.max(dim=0)                           # [S], [S]
        better = max_vals > top_vals
        top_vals[better] = max_vals[better]
        top_idxs[better] = start + max_local[better]

    return top_idxs

#def run_nystrom_ts_top1_online(
#    kernel,
#    inducing_points: torch.Tensor,     # [M, D]
#    candidate_set: torch.Tensor,       # [N, D]
#    num_samples: int,                  # S
#    K_MM_inv_root: torch.Tensor,       # [M, M]
#    mu_w: torch.Tensor,                # [M]
#    L_A: torch.Tensor,                 # [M, M]  Cholesky of A
#    batch_size: int = 100_000,
#    verbose: bool = True,
#) -> torch.Tensor:
#
#    device, dtype = candidate_set.device, candidate_set.dtype
#    M, S, N = inducing_points.shape[0], num_samples, candidate_set.shape[0]
#
#    # --- Posterior weight samples: w = μ_w + A^{-1/2} ξ
#    xi = torch.randn(M, S, device=device, dtype=dtype)  # [M, S]
#    # Solve L_A^T Δ = ξ  (upper triangular)
#    delta = torch.linalg.solve_triangular(L_A.transpose(-1, -2), xi, upper=True)
#    W  = mu_w[:, None] + delta  # [M, S]
#
#    # --- Your existing streaming argmax stays the same
#    top_vals = torch.full((S,), float("-inf"), device=device, dtype=dtype)
#    top_idxs = torch.full((S,), -1, dtype=torch.long, device=device)
#
#    for start in range(0, N, batch_size):
#        end = min(start + batch_size, N)
#        if verbose:
#            print(f"[Chunk] Processing candidates {start:,} to {end:,}...")
#
#        X_chunk = candidate_set[start:end]
#        K_NM_chunk = compute_kernel_matrix(X_chunk, inducing_points, kernel)
#        if isinstance(K_NM_chunk, tuple): K_NM_chunk = K_NM_chunk[0]
#        K_NM_chunk = K_NM_chunk.to(device=device, dtype=dtype)
#
#        Phi_chunk = K_NM_chunk @ K_MM_inv_root     # [chunk, M]
#        scores    = Phi_chunk @ W                  # [chunk, S]
#
#        max_vals, max_local_idxs = scores.max(dim=0)
#        mask = max_vals > top_vals
#        top_vals[mask] = max_vals[mask]
#        top_idxs[mask] = start + max_local_idxs[mask]
#
#    return top_idxs

def run_global_nystrom_ts(kernel, inducing_points, candidate_set, num_samples,
                          train_X, train_Y, noise_variance, batch_size=100_000):
    with torch.no_grad():
        mu_w, L_A, K_MM_inv_root = nystrom_posterior_weights(
            kernel, inducing_points, train_X, train_Y, noise_variance
        )
        top_idxs = run_nystrom_ts_top1_online(
            kernel=kernel,
            inducing_points=inducing_points,
            candidate_set=candidate_set,
            num_samples=num_samples,
            K_MM_inv_root=K_MM_inv_root,
            mu_w=mu_w,
            L_A=L_A,
            batch_size=batch_size,
            verbose=True,
        )

    return top_idxs
@torch.no_grad()
def thompson_unique_multi_draw(
    kernel, inducing_points, candidate_set, K,
    K_MM_inv_root, mu_w, L_A,
    S=None, topL=32, batch_size=100_000, verbose=True,
):
    # ---- NEW: unify dtype & device ----
    device = mu_w.device
    dtype  = mu_w.dtype  # or torch.float64 if you want to force double

    inducing_points = inducing_points.to(device=device, dtype=dtype)
    candidate_set   = candidate_set.to(device=device, dtype=dtype)
    K_MM_inv_root   = K_MM_inv_root.to(device=device, dtype=dtype)
    mu_w            = mu_w.to(device=device, dtype=dtype)
    L_A             = L_A.to(device=device, dtype=dtype)

    M = inducing_points.shape[0]
    N = candidate_set.shape[0]
    if S is None: S = K

    # Posterior samples of weights: w_s = mu_w + L_A^{-T} ξ_s
    Xi = torch.randn(M, S, device=device, dtype=dtype)
    V  = torch.linalg.solve(L_A.transpose(-2, -1), Xi)   # [M,S]
    W  = mu_w[:, None] + V                               # [M,S]

    # Per-sample topL buffers
    top_vals = torch.full((topL, S), float("-inf"), device=device, dtype=dtype)
    top_idx  = torch.full((topL, S), -1, device=device, dtype=torch.long)

    # Stream over candidates
    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        Xb    = candidate_set[start:end]                 # already on device/dtype
        K_XM  = compute_kernel_matrix(Xb, inducing_points, kernel)
        if isinstance(K_XM, tuple): K_XM = K_XM[0]
        K_XM  = K_XM.to(dtype=dtype)                     # ---- NEW: coerce dtype ----

        Phi_b = K_XM @ K_MM_inv_root                     # [B,M]
        Sc    = Phi_b @ W                                # [B,S]

        vals_b, idx_b = torch.topk(Sc, k=min(topL, Sc.shape[0]), dim=0)
        idx_b = idx_b + start

        vals_cat = torch.cat([top_vals, vals_b], dim=0)
        idx_cat  = torch.cat([top_idx,  idx_b],  dim=0)
        new_vals, new_pos = torch.topk(vals_cat, k=topL, dim=0)
        top_vals = new_vals
        top_idx  = idx_cat.gather(0, new_pos)

        if verbose and start == 0:
            print(f"[TS] Streaming candidates with S={S}, topL={topL}...")

    # Greedy de-dup
    chosen, chosen_set = [], set()
    order = torch.randperm(S, device=device).tolist()
    p = 0
    while len(chosen) < K and p < topL:
        for s in order:
            if len(chosen) >= K: break
            idx = int(top_idx[p, s].item())
            if idx >= 0 and idx not in chosen_set:
                chosen.append(idx); chosen_set.add(idx)
        p += 1

    if len(chosen) < K:
        for idx in top_idx.view(-1).tolist():
            if idx >= 0 and idx not in chosen_set:
                chosen.append(idx); chosen_set.add(idx)
                if len(chosen) == K: break

    return torch.tensor(chosen, dtype=torch.long, device='cpu')

#@torch.no_grad()
#def thompson_unique_multi_draw(
#    kernel, inducing_points, candidate_set, K,            # want K unique picks
#    K_MM_inv_root, mu_w, L_A,                             # posterior over weights
#    S=None, topL=32, batch_size=100_000, verbose=True,
#):
#    device = candidate_set.device
#    dtype  = mu_w.dtype
#    M      = inducing_points.shape[0]
#    N      = candidate_set.shape[0]
#    if S is None: S = K  # #samples (≥K recommended)
#
#    # Posterior samples of weights: w_s = mu_w + L_A^{-T} ξ_s
#    Xi = torch.randn(M, S, device=device, dtype=dtype)
#    V  = torch.linalg.solve(L_A.transpose(-2, -1), Xi)  # [M,S]
#    W  = mu_w[:, None] + V                              # [M,S]
#
#    # Per-sample topL buffers across the full candidate set
#    top_vals = torch.full((topL, S), float("-inf"), device=device, dtype=dtype)
#    top_idx  = torch.full((topL, S), -1, device=device, dtype=torch.long)
#
#    # Stream candidates and keep topL per sample
#    for start in range(0, N, batch_size):
#        end   = min(start + batch_size, N)
#        Xb    = candidate_set[start:end]
#        K_XM  = compute_kernel_matrix(Xb, inducing_points, kernel)
#        if isinstance(K_XM, tuple): K_XM = K_XM[0]
#        Phi_b = K_XM @ K_MM_inv_root           # [B,M]
#        Sc    = Phi_b @ W                      # [B,S] scores for all samples
#
#        # topL per column (per sample) in this chunk
#        vals_b, idx_b = torch.topk(Sc, k=min(topL, Sc.shape[0]), dim=0)  # [L,S]
#        idx_b = idx_b + start
#
#        # Merge chunk topL with global topL per sample (concat then take topL)
#        vals_cat = torch.cat([top_vals, vals_b], dim=0)        # [2L,S]
#        idx_cat  = torch.cat([top_idx,  idx_b],  dim=0)
#        new_vals, new_pos = torch.topk(vals_cat, k=topL, dim=0)
#        top_vals = new_vals
#        top_idx  = idx_cat.gather(0, new_pos)
#
#        if verbose and start == 0:
#            print(f"[TS] Streaming candidates with S={S}, topL={topL}...")
#
#    # Greedy de-dup: traverse down the per-sample lists until we collect K uniques
#    chosen = []
#    chosen_set = set()
#    # Optionally randomize sample order to reduce bias
#    order = torch.randperm(S).tolist()
#    p = 0
#    while len(chosen) < K and p < topL:
#        for s in order:
#            if len(chosen) >= K: break
#            idx = int(top_idx[p, s].item())
#            if idx >= 0 and idx not in chosen_set:
#                chosen.append(idx)
#                chosen_set.add(idx)
#        p += 1
#    if len(chosen) < K:
#        # fallback: fill with next-best globally among remaining (rare)
#        rest = top_idx.view(-1).tolist()
#        for idx in rest:
#            if idx >= 0 and idx not in chosen_set:
#                chosen.append(idx); chosen_set.add(idx)
#                if len(chosen) == K: break
#
#    return torch.tensor(chosen, dtype=torch.long)
