import torch
import numpy as np
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from alpfore.utils.kernel_utils import compute_kernel_matrix, gpytorch_kernel_wrapper


def run_stratified_batched_ts(model, candidate_set, batch_size=1000, k_per_batch=1, k_per_seqlen=None, stratify_set=False):
    selected_inds = []
    N = candidate_set.shape[0]

    if stratify_set:
        seqlens_all = candidate_set[:, 3].cpu().numpy()
        sort_idx = np.argsort(seqlens_all)
        candidate_set = candidate_set[sort_idx]

    # Round and scale seqlens to use as discrete keys
    selected_counts = {}
    scale = 1000  # or whatever scale matches seqlen granularity

    for i in range(0, N, batch_size):
        X_batch = candidate_set[i:i + batch_size]
        posterior = model.posterior(X_batch)
        f_sample = posterior.rsample(sample_shape=torch.Size([1]))[0].squeeze()

        if k_per_seqlen is not None:
            seqlens = np.round(X_batch[:, 3].cpu().numpy(), decimals=3)
            seqlens_int = (seqlens * scale).astype(int)
            f_values = f_sample.detach().cpu().numpy()

            for seqlen in np.unique(seqlens_int):
                mask = seqlens_int == seqlen
                idxs = np.where(mask)[0]

                already_selected = selected_counts.get(seqlen, 0)
                remaining_quota = max(0, k_per_seqlen - already_selected)

                if remaining_quota == 0 or len(idxs) == 0:
                    continue

                topk = min(remaining_quota, len(idxs))
                topk_local = idxs[np.argsort(f_values[idxs])[-topk:]]

                # Extend with indices rather than X_batch values
                selected_inds.extend((i + topk_local).tolist())
                selected_counts[seqlen] = already_selected + topk
        else:
            topk_idx = torch.topk(f_sample, k=min(k_per_batch, f_sample.shape[0])).indices
            selected_inds.extend((i + topk_idx).tolist())

#                topk = min(remaining_quota, len(idxs))
#                topk_local = idxs[np.argsort(f_values[idxs])[-topk:]]
#
#                selected.extend(X_batch[topk_local].tolist())
#                selected_counts[seqlen] = already_selected + topk
#        else:
#            topk_idx = torch.topk(f_sample, k=min(k_per_batch, f_sample.shape[0])).indices
#            selected.extend(X_batch[topk_idx].tolist())

    return candidate_set[torch.tensor(selected_inds)]

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


#def run_global_nystrom_ts(kernel, inducing_points, candidate_set, k_global, train_X, train_Y):
#    """
#    Perform global Thompson Sampling using Nyström kernel approximation.
#    """
#    # Compute kernel matrices
#    K_NM = compute_kernel_matrix(candidate_set, inducing_points, kernel)
#    K_MM = compute_kernel_matrix(inducing_points, inducing_points, kernel)
#
#    # Invert K_MM (or use Cholesky solve)
#    K_MM_inv = np.linalg.pinv(K_MM + 1e-6 * np.eye(K_MM.shape[0]))  # regularization for stability
#
#    # Compute approximate mean: mu_N = K_NM K_MM_inv alpha
#    # where alpha = K_train_train^{-1} Y (assuming train_X = inducing_points)
#    # If train_X != inducing_points, use a separate K_TM
#    K_TM = compute_kernel_matrix(inducing_points, train_X, kernel)
#    K_TT = compute_kernel_matrix(train_X, train_X, kernel)
#    alpha = np.linalg.solve(K_TT + 1e-6 * np.eye(K_TT.shape[0]), train_Y)
#    mean_N = K_NM @ K_MM_inv @ K_TM @ alpha
#    # Compute approximate covariance: K_NN_approx = K_NM K_MM^{-1} K_MN
#    K_MN = K_NM.T
#    cov_N = K_NM @ K_MM_inv @ K_MN
#    # Sample from the approximate posterior
#    cov_N = 0.5 * (cov_N + cov_N.T)
#    jitter = 1e-5 * torch.eye(cov_N.shape[0], device=cov_N.device)
#    cov_N_reg = cov_N + jitter
#    
#    mvn = MultivariateNormal(mean_N.detach(), cov_N_reg.detach())
#    f_sample = mvn.rsample(sample_shape=torch.Size([1]))[0, 0]  # shape: [N]
#
#    # Select top-k candidates
#    topk_indices = torch.topk(f_sample, k_global).indices
#    selected = [candidate_set[i] for i in topk_indices]
#    return selected


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

