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
    batch_size: int = 1000,
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
    e    candidate_set: torch.Tensor,
    batch_size: int = 1000,
    k_per_seqlen: Union[int, list] = 5,
    seqlen_col: int = 3,
    stratify_set: bool = True,
    seqlen_round_decimals: int = 3,
    seed: int = Nonelse:
        raise ValueError("batch_size must be an int or a list of same length as number of unique seqlens")


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


def run_global_nystrom_ts(kernel, inducing_points, candidate_set, k_global, train_X, train_Y, excluded_ids=None):
    """
    Thompson Sampling with global Nyström approximation.

    Parameters
    ----------
    kernel : GPyTorch kernel
    inducing_points : torch.Tensor, shape [M, d]
    candidate_set : torch.Tensor, shape [N, d]
    k_global : int
    train_X : torch.Tensor, shape [n, d]
    train_Y : torch.Tensor, shape [n]
    excluded_ids : Optional[Set[int]] of candidate indices to ignore

    Returns
    -------
    selected_indices : List[int]
    """
    N = candidate_set.shape[0]
    M = inducing_points.shape[0]

    # Compute kernel matrices using compute_kernel_matrix()
    K_NM, *_ = compute_kernel_matrix(candidate_set, inducing_points, kernel)   # [N, M]
    K_MM, *_ = compute_kernel_matrix(inducing_points, inducing_points, kernel) # [M, M]
    K_TM, *_ = compute_kernel_matrix(train_X, inducing_points, kernel)         # [n, M]

    # Regularize and invert K_MM
    jitter = 1e-6
    K_MM_inv = torch.linalg.pinv(K_MM + jitter * torch.eye(M))

    K_MM_inv = K_MM_inv.float()
    K_TM = K_TM.float()
    train_Y = train_Y.float()

    # alpha = K_MM_inv @ K_TM.T @ train_Y
    alpha = K_MM_inv @ K_TM.T @ train_Y  # [M]

    # Mean and Covariance
    mean_N = K_NM @ alpha                # [N]
    cov_N = K_NM @ K_MM_inv @ K_NM.T     # [N, N]
    cov_N_reg = cov_N + jitter * torch.eye(N)

# Defensive reshape if needed
    if mean_N.ndim > 1:
        mean_N = mean_N.squeeze()
    if cov_N_reg.ndim > 2:
        cov_N_reg = cov_N_reg.squeeze()

    # Confirm shape: mean [N], cov [N, N]
    assert mean_N.ndim == 1, f"Expected mean_N to be 1D but got {mean_N.shape}"
    assert cov_N_reg.ndim == 2, f"Expected cov_N_reg to be 2D but got {cov_N_reg.shape}"
    assert cov_N_reg.shape[0] == cov_N_reg.shape[1] == mean_N.shape[0], "Covariance shape mismatch"

    # Sample from posterior
    mvn = MultivariateNormal(mean_N.detach(), cov_N_reg.detach())
    f_samples = mvn.rsample(sample_shape=torch.Size([k_global]))  # [k_global, N]



    # Exclusion logic
    excluded_ids = set(excluded_ids) if excluded_ids is not None else set()
    selected_indices = []

    for i in range(k_global):
        sample_i = f_samples[i].clone()
        sample_i[list(excluded_ids)] = float('-inf')
        top_idx = torch.argmax(sample_i).item()
        selected_indices.append(top_idx)
        excluded_ids.add(top_idx)

    return candidate_set[selected_indices]


def select_ts_candidates(model, candidate_set, inducing_points,
                          kernel, train_X, train_Y, k2,
                          strat_batch_size=1000, k_per_batch=1, k_per_seqlen=None, stratify_set=False):
    """
    Combines stratified batched TS with global TS from Nyström posterior.
    """
    stratified_candidates = run_stratified_batched_ts(
        model, candidate_set, batch_size=strat_batch_size,
        k_per_batch=k_per_batch, k_per_seqlen=k_per_seqlen, stratify_set=stratify_set
    )
    print(np.shape(np.asarray(stratified_candidates)))
    global_candidates = run_global_nystrom_ts(kernel, inducing_points, candidate_set, k2, train_X, train_Y)
    print(np.shape(np.asarray(global_candidates)))
    # Either convert both to lists:
    return stratified_candidates.tolist() + global_candidates.tolist()

