# src/alpfore/utils/kernel_utils.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from botorch.models import FixedNoiseGP
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
import joblib
import math
from pathlib import Path
import os
import torch
from typing import Optional, Callable, Tuple
from joblib import Parallel, delayed

def gpytorch_kernel_wrapper(x1, x2, kernel):
    x1_torch = torch.tensor(x1, dtype=torch.float32)
    x2_torch = torch.tensor(x2, dtype=torch.float32)

    # Ensure shape [1, 1, D]
    if x1_torch.dim() == 1:
        x1_torch = x1_torch.unsqueeze(0).unsqueeze(0)
    elif x1_torch.dim() == 2:
        x1_torch = x1_torch.unsqueeze(1)

    if x2_torch.dim() == 1:
        x2_torch = x2_torch.unsqueeze(0).unsqueeze(0)
    elif x2_torch.dim() == 2:
        x2_torch = x2_torch.unsqueeze(1)

    with torch.no_grad():
        K = kernel(x1_torch, x2_torch).evaluate()

    if K.shape == torch.Size([1, 1]):
        K = K.unsqueeze(0)

    # Final safety squeeze
    return K.squeeze().item()  # return scalar

def compute_kernel_matrix(
    X1: torch.Tensor,
    X2: torch.Tensor,
    kernel_func: Callable,
    Y_train: Optional[torch.Tensor] = None,
    model: Optional[object] = None,
    clamp_var: float = 1e-6,
    batch_size: int = 1000,
    save_dir: Optional[str] = None,
    prefix: str = "",
    verbose: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Efficient torch-native kernel matrix computation with optional posterior prediction.
    """

    N = X1.shape[0]
    M = X2.shape[0]
    K_full = torch.empty((N, M), device=X1.device)

    # Compute K(X1, X2) in batches
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        with torch.no_grad():
            K_full[i:end] = kernel_func(X1[i:end], X2).evaluate()

    means, vars_ = None, None

    # Save kernel if requested
    if save_dir is not None:
        k_path = Path(save_dir) / f"{prefix}_full.pt"
        torch.save(K_full, k_path)

    if Y_train is not None:
        try:
            K_train_train = torch.load(Path(save_dir) / f"{prefix}_train_train.pt")
        except FileNotFoundError:
            K_train_train = torch.empty((M, M), device=X2.device)
            for i in range(0, M, batch_size):
                end = min(i + batch_size, M)
                with torch.no_grad():
                    K_train_train[i:end] = kernel_func(X2[i:end], X2).evaluate()
            if save_dir:
                torch.save(K_train_train, Path(save_dir) / f"{prefix}_train_train.pt")

        # Posterior mean and variance
        K_inv = torch.inverse(K_train_train + clamp_var * torch.eye(M, device=X2.device))
        means = K_full @ K_inv @ Y_train

        cov_term = torch.einsum("ij,jk,ik->i", K_full, K_inv, K_full)
        K_diag = torch.ones_like(cov_term)
        vars_ = (K_diag - cov_term).clamp(min=clamp_var)

        if model is not None:
            mu_Y = model.outcome_transform.means
            std_Y = model.outcome_transform.stdvs
            means = means * std_Y + mu_Y
            vars_ = vars_ * (std_Y ** 2)

        if save_dir:
            torch.save(means, Path(save_dir) / f"{prefix}_means.pt")
            torch.save(vars_, Path(save_dir) / f"{prefix}_vars.pt")

    if verbose:
        print(f"[compute_kernel_matrix_torch] Done.")

    return K_full, means, vars_

#def compute_kernel_matrix(X1, X2, kernel_func, save_dir=None, prefix="kernel",
#                          verbose=True, Y_train=None, clamp_var=1e-6, model=None):
#    """
#    Computes full kernel matrix between X1 and X2 in one shot using the provided kernel_func.
#    Also optionally computes posterior means/variances and saves everything to disk.
#
#    Parameters
#    ----------
#    X1 : np.ndarray
#    X2 : np.ndarray
#    kernel_func : GPyTorch kernel
#    save_dir : str or Path
#    prefix : str
#    verbose : bool
#    Y_train : torch.Tensor or None
#    clamp_var : float
#    model : GP model (to unstandardize predictions)
#
#    Returns
#    -------
#    K_full : torch.Tensor
#        Full kernel matrix of shape (len(X1), len(X2))
#    means : torch.Tensor or None
#        Posterior mean predictions (only if Y_train is provided)
#    vars_ : torch.Tensor or None
#        Posterior variances (only if Y_train is provided)
#    """
#    if save_dir is not None:
#        Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#    if verbose:
#        print(f"[compute_kernel_matrix] Computing full kernel matrix ({len(X1)} x {len(X2)})...")
#
#    if isinstance(X1, torch.Tensor):
#        X1 = X1.cpu().detach().numpy()
#    if isinstance(X2, torch.Tensor):
#        X2 = X2.cpu().detach().numpy()
#
#
#    K_full = torch.tensor(pairwise_kernels(
#        X1, X2, metric=lambda a, b: gpytorch_kernel_wrapper(a, b, kernel=kernel_func)
#    ))
#
#    means, vars_ = None, None
#
#    if save_dir is not None:
#        k_path = Path(save_dir) / f"{prefix}_full.pkl"
#        joblib.dump(K_full, k_path)
#    if Y_train is None:
#        if verbose:
#            print("[compute_kernel_matrix] Done.")
#        return K_full, means, vars_
#
#    if Y_train is not None:
#        try:
#            K_train_train = torch.tensor(joblib.load(Path(save_dir) / f"{prefix}_train_train.pkl"))
#        except FileNotFoundError:
#            K_train_train = torch.tensor(pairwise_kernels(
#                X2, X2, metric=lambda a, b: gpytorch_kernel_wrapper(a, b, kernel=kernel_func)
#            ))
#            joblib.dump(K_train_train, Path(save_dir) / f"{prefix}_train_train.pkl")
#
#        K_inv = torch.inverse(K_train_train + clamp_var * torch.eye(K_train_train.shape[0]))
#        K_test_batch = K_full.to(dtype=torch.float64)
#
#        means = K_test_batch @ K_inv @ Y_train
#        cov_term = torch.einsum("ij,jk,ik->i", K_test_batch, K_inv, K_test_batch)
#        K_diag = torch.ones_like(cov_term)  # assume normalized kernel
#        vars_ = (K_diag - cov_term).clamp(min=clamp_var)
#
#        if model is not None:
#            mu_Y = model.outcome_transform.means
#            std_Y = model.outcome_transform.stdvs
#            means = means * std_Y + mu_Y
#            vars_ = vars_ * (std_Y ** 2)
#
#        if save_dir:
#            torch.save(means, Path(save_dir) / f"means_full.pt")
#            torch.save(vars_, Path(save_dir) / f"vars_full.pt")
#
#    if verbose:
#        print(f"[compute_kernel_matrix] Done.")
#
#    return K_full, means, vars_

def load_kernel_matrix(path):
    """
    Load a precomputed kernel matrix from disk.
    """
    return joblib.load(path)

def compute_kernel_pca(K_train_train, n_components=2):
    """
    Perform kernel PCA given a precomputed train-train kernel matrix.

    Parameters:
    - K_train_train: ndarray, shape (n_train, n_train)
    - n_components: int, number of KPCs to retain

    Returns:
    - kpca: fitted KernelPCA object
    - transformed_train: shape (n_train, n_components)
    """
    kpca = KernelPCA(n_components=n_components, kernel='precomputed')
    transformed_train = kpca.fit_transform(K_train_train)
    return kpca, transformed_train

def transform_with_kpca(kpca, K_test_train):
    """
    Use fitted KPCA object to transform test points using test-train kernel matrix.

    Parameters:
    - kpca: fitted KernelPCA object
    - K_test_train: ndarray, shape (n_test, n_train)

    Returns:
    - transformed_test: shape (n_test, n_components)
    """
    return kpca.transform(K_test_train)

