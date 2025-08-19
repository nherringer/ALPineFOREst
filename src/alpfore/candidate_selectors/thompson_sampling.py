import torch
import numpy as np
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from alpfore.utils.kernel_utils import compute_kernel_matrix, gpytorch_kernel_wrapper


import torch
import numpy as np
from typing import Union

def run_stratified_batched_ts(
    model,
    candidate_set: torch.Tensor,
    batch_size: Union[int, list] = 1000,
    k_per_seqlen: Union[int, list] = 5,
    seqlen_col: int = 3,
    stratify_set: bool = True,
    seqlen_round_decimals: int = 3,
    seed: int = None
):
    """
    Stratified batched Thompson sampling.

    Parameters
    ----------
    model : GPyTorch model
        The GP model with a `.posterior()` method.
    candidate_set : torch.Tensor
        Candidate set of shape [N, d].
    batch_size : int
        Number of candidates to evaluate at a time.
    k_per_seqlen : int or list
        How many final selections to return per seqlen group.
    seqlen_col : int
        Column index where seqlen is stored.
    stratify_set : bool
        Whether to group candidates by seqlen.
    seqlen_round_decimals : int
        Decimal precision to round seqlens to when grouping.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Final selected candidates of shape [sum(k_per_seqlen), d].
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    selected_all = []
    device = candidate_set.device
    seqlens_raw = candidate_set[:, seqlen_col].cpu().numpy()
    seqlens_rounded = np.round(seqlens_raw, decimals=seqlen_round_decimals)

    # Unique seqlens and group indices
    unique_seqlens = np.unique(seqlens_rounded)

    if isinstance(batch_size, int):
        batch_dict = {s: batch_size for s in unique_seqlens}
    elif isinstance(batch_size, list) and len(batch_size) == len(unique_seqlens):
        batch_dict = dict(zip(unique_seqlens, batch_size))
    else:
        raise ValueError("batch_size must be an int or a list of same length as number of unique seqlens")


    if isinstance(k_per_seqlen, int):
        k_dict = {s: k_per_seqlen for s in unique_seqlens}
    elif isinstance(k_per_seqlen, list) and len(k_per_seqlen) == len(unique_seqlens):
        k_dict = dict(zip(unique_seqlens, k_per_seqlen))
    else:
        raise ValueError("k_per_seqlen must be an int or a list of length equal to number of unique seqlens")

    for seqlen in unique_seqlens:
        # Indices of candidates with this seqlen
        mask = seqlens_rounded == seqlen
        X_seqlen = candidate_set[mask]
        print(f"Batching seqlen: {seqlen}")
        # --- Round 1: batch-wise greedy selection ---
        batch_winners = []
        batch_sz = batch_dict[seqlen]
        for i in range(0, X_seqlen.shape[0], batch_sz):
            X_batch = X_seqlen[i:i+batch_sz]
            posterior = model.posterior(X_batch)
            f_sample = posterior.rsample(sample_shape=torch.Size([1]))[0].squeeze()
            if f_sample.dim() == 0:
                best_idx = 0
            else:
                best_idx = torch.argmax(f_sample).item()
            batch_winners.append(X_batch[best_idx])

        # Stack winners into tensor
        winners_tensor = torch.stack(batch_winners, dim=0)

        # --- Round 2: Thompson sampling on batch winners ---
        n_to_select = k_dict[seqlen]
        winners_remaining = winners_tensor.clone()
        selected_for_this_seqlen = []

        for _ in range(n_to_select):
            posterior = model.posterior(winners_remaining)
            f_sample = posterior.rsample(sample_shape=torch.Size([1]))[0].squeeze()
            best_idx = torch.argmax(f_sample).item()
            selected_for_this_seqlen.append(winners_remaining[best_idx])
            winners_remaining = torch.cat([
                winners_remaining[:best_idx],
                winners_remaining[best_idx+1:]
            ], dim=0)

        selected_all.extend(selected_for_this_seqlen)

    return torch.stack(selected_all, dim=0)


def run_global_nystrom_ts(kernel, inducing_points, candidate_set, num_samples,
                          train_X=None, train_Y=None, batch_size=50000):
    """
    Thompson sampling using Nyström approximation with compute_kernel_matrix.

    Args:
        kernel: callable kernel (used by compute_kernel_matrix)
        inducing_points: [M, D] inducing set
        candidate_set: [N, D] candidate/test set
        num_samples: number of Thompson samples to draw
        train_X: Input data for pretested datapoints
        train_Y: Output data for pretested datapoints
        batch_size: batch size to use for computing K_XstarXbar

    Returns:
        samples: [num_samples, N] sampled function values over candidate_set
    """
    device = candidate_set.device
    N, D = candidate_set.shape
    M = inducing_points.shape[0]
    num_tested = train_X.shape[0]

    with torch.no_grad():
        # Step 1: Compute kernel matrices
        K_XbarXbar = compute_kernel_matrix(inducing_points, inducing_points, kernel)[0]  # [M, M]
        print("Inducing point kernel calculated")
        K_XstarXbar = compute_kernel_matrix(candidate_set, inducing_points, kernel, prefix="K_XstarXbar",save_dir=".", batch_size=batch_size)[0]  # [N, M]
        print("K_X*Xbar calculated") 
        K_XX = compute_kernel_matrix(train_X, train_X, kernel)[0] # [num_tested, num_tested]
        print("K_XX computed")
        K_XXstart = compute_kernel_matrix(candidate_set, train_X, kernel)[0] # [num_tested, N]
        print("Kernels computed")

        # Step 2: Stabilize and invert K_MM
        jitter = 1e-6 * torch.eye(num_tested, device=device)
        K_XX_inv = torch.inverse(K_XX+jitter)
        mu = K_XXstar.t @ K_XX_inv @ train_Y
        # SVD of K_XstarXbar:
        U_Xstar, S, U_Xbart = torch.linalg.svd(X, full_matrices=False)
        S_inv = 1/S
        var = (K_XstarXbar @ U_Xbart.t @ torch.diag(S_inv)@U_Xbart) - K_XXstar.t @ K_XX_inv @ K_XXstar
        var_A = torch.linalg.cholesky(A)
        print("New sampling params calculated")

        # Step 3: Sample in low-rank space
        Z = torch.randn(num_samples, N, device=device)  # [S, M]

        # Step 4: Project to candidate space
        samples = mu + torch.einsum('ij,sj->si', var_A, Z) # [S, N]
    return samples

def select_ts_candidates(model, candidate_set, inducing_points,
                          kernel, train_X, train_Y, k2,
                          strat_batch_size=1000, k_per_seqlen=None, stratify_set=True):
    """
    Combines stratified batched TS with global TS from Nyström posterior.
    """
    stratified_candidates = run_stratified_batched_ts(
        model, candidate_set, batch_size=strat_batch_size,
        k_per_seqlen=k_per_seqlen, stratify_set=stratify_set
    )
    print(np.shape(np.asarray(stratified_candidates)))
    global_candidates = run_global_nystrom_ts(kernel, inducing_points, candidate_set, k2, train_X, train_Y)
    print(np.shape(np.asarray(global_candidates)))
    # Either convert both to lists:
    return stratified_candidates.tolist() + global_candidates.tolist()

