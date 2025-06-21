# Auto-generated test stubs for `utils.kernel_utils`
import pytest
import torch
import numpy as np
from sklearn.decomposition import KernelPCA
from alpfore.utils.kernel_utils import (
    gpytorch_kernel_wrapper,
    compute_kernel_matrix,
    load_kernel_matrix,
    compute_kernel_pca,
    transform_with_kpca,
)
from gpytorch.kernels import RBFKernel
import tempfile
import joblib
import os

def test_gpytorch_kernel_wrapper():
    kernel = RBFKernel()
    x1 = np.random.rand(5)
    x2 = np.random.rand(5)

    val = gpytorch_kernel_wrapper(x1, x2, kernel)
    assert isinstance(val, float)
    assert val >= 0.0  # RBF kernel is positive

def test_compute_kernel_matrix():
    X = torch.randn(10, 5)
    Y = torch.randn(10, 1)
    K_full1, mean, var = compute_kernel_matrix(X, X, kernel_type='rbf', Y_train=None)
    K_full2, mean, var = compute_kernel_matrix(X, X, kernel_type='rbf', Y_train=Y)
    assert K_full1.shape == (10, 10)
    assert K_full2.shape == (10, 10)
    assert mean.shape == (10, 1)
    assert var.shape == (10,)

    assert torch.allclose(K_full1, K_full1.T, atol=1e-5)
    assert torch.allclose(K_full2, K_full2.T, atol=1e-5)
    assert torch.all(var >= 0)

    # Optional: test mean sanity for constant Y
    Y_const = torch.full((10, 1), 5.0)
    _, mean_const, _ = compute_kernel_matrix(X, X, kernel_type='rbf', Y_train=Y_const)
    assert torch.allclose(mean_const, Y_const, atol=1e-1)

def test_load_kernel_matrix():
    mat = np.random.rand(3, 3)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        joblib.dump(mat, tmp.name)
        tmp_path = tmp.name

    loaded = load_kernel_matrix(tmp_path)
    os.unlink(tmp_path)  # Clean up

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (3, 3)
    np.testing.assert_allclose(loaded, mat)


def test_compute_kernel_pca():
    # Use identity kernel for easy sanity check
    K = np.eye(5)
    kpca, X_trans = compute_kernel_pca(K, n_components=2)

    assert isinstance(kpca, KernelPCA)
    assert X_trans.shape == (5, 2)


def test_transform_with_kpca():
    # Create a toy dataset with easy-to-follow kernel
    X = np.random.rand(5, 2)
    K_train = pairwise_rbf_kernel(X, X)
    kpca, X_trans = compute_kernel_pca(K_train, n_components=2)

    X_test = np.random.rand(2, 2)
    K_test_train = pairwise_rbf_kernel(X_test, X)
    X_test_trans = transform_with_kpca(kpca, K_test_train)

    assert X_test_trans.shape == (2, 2)

