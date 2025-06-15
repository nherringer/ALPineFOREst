# src/alpfore/utils/kernel_utils.py

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
import joblib

def compute_kernel_matrix(X1, X2, kernel_func, save_path=None):
    """
    Compute the kernel similarity matrix between two datasets.
    
    Parameters:
    - X1: ndarray of shape (n_samples_1, n_features)
    - X2: ndarray of shape (n_samples_2, n_features)
    - kernel_func: callable, function to compute pairwise similarity
    - save_path: optional path to save the kernel matrix
    
    Returns:
    - K: Kernel matrix of shape (n_samples_1, n_samples_2)
    """
    K = pairwise_kernels(X1, X2, metric=kernel_func)
    if save_path is not None:
        joblib.dump(K, save_path)
    return K

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

