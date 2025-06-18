# src/alpfore/utils/kernel_utils.py
from botorch.models import FixedNoiseGP
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
import joblib
import math
from pathlib import Path
import os
import torch
from typing import Optional

def gpytorch_kernel_wrapper(x1, x2, kernel=None):
    """
    Wraps a GPyTorch kernel for use with scikit-learn's pairwise_kernels.

    Parameters:
    - x1: ndarray of shape (n1, d)
    - x2: ndarray of shape (n2, d)
    - kernel: a GPyTorch kernel instance (must be set externally before use)

    Returns:
    - similarity matrix of shape (n1, n2)
    """
    if kernel is None:
        raise ValueError("You must provide a GPyTorch kernel instance to gpytorch_kernel_wrapper.")
    
    # Convert to torch tensors with batch dimension
    import torch
    x1_torch = torch.tensor(x1).unsqueeze(1).double()
    x2_torch = torch.tensor(x2).unsqueeze(1).double()

    with torch.no_grad():
        K = kernel(x1_torch, x2_torch).evaluate()

    return K.numpy()

def compute_kernel_matrix(
    X1,
    X2,
    kernel_func,
    batch_size=1000,
    save_dir=None,
    prefix="K_test_train",
    return_file_paths=True,
    verbose=True,
    Y_train=None,
    clamp_var=1e-6,
    model: Optional[FixedNoiseGP] = None
):
    """
    Compute and save the kernel similarity matrix between X1 and X2 in batches,
    and compute posterior mean/variance for each batch if K_train_train and Y_train are provided.

    Returns:
    - file_paths: list of paths to K_batch files (if return_file_paths is True)
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    n1 = X1.shape[0]
    file_paths = []

    K_train_train = pairwise_kernels(
            X2, X2, metric=lambda a, b: gpytorch_kernel_wrapper(a, b, kernel=kernel_func)
        )

    if Y_train is not None:
        K_inv = torch.linalg.inv(K_train_train)
        Y_train = Y_train.squeeze()

    for i in range(0, n1, batch_size):
        X1_batch = X1[i : i + batch_size]
        K_batch = pairwise_kernels(
            X1_batch, X2, metric=lambda a, b: gpytorch_kernel_wrapper(a, b, kernel=kernel_func)
        )

        # Save kernel matrix batch
        if save_dir is not None:
            k_path = Path(save_dir) / f"{prefix}_{i:05d}.pkl"
            joblib.dump(K_batch, k_path)
            file_paths.append(str(k_path))

        # === Posterior mean/var computation ===
        if Y_train is not None:
            K_test_batch = torch.tensor(K_batch, dtype=torch.float64)
            means = K_test_batch @ K_inv @ Y_train
            cov_term = torch.einsum("ij,jk,ik->i", K_test_batch, K_inv, K_test_batch)
            K_diag = torch.ones_like(cov_term)  # assume normalized kernel
            vars_ = (K_diag - cov_term).clamp(min=clamp_var)

            if model is not None:
                mu_Y = model.outcome_transform.means
                std_Y = model.outcome_transform.stdvs
                means = means * std_Y + mu_Y
                vars_ = vars_ * (std_Y ** 2)
            # Save mean and var
            torch.save(means, Path(save_dir) / f"means_{i:05d}.pt")
            torch.save(vars_, Path(save_dir) / f"vars_{i:05d}.pt")

        if verbose:
            print(f"[compute_kernel_matrix] Batch {i:05d}-{min(i+batch_size, n1):05d} processed.")

    return file_paths if return_file_paths else None

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

