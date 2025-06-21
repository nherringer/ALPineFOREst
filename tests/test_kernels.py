import torch
from alpfore.models.kernels import TanimotoKernel, CustomKernel
import pytest

def test_tanimoto_kernel_forward_basic():
    kernel = TanimotoKernel()
    x = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32)
    K = kernel(x, x).evaluate()
    assert K.shape == (2, 2)
    assert torch.all(K <= 1.0) and torch.all(K >= 0.0)
    assert torch.allclose(K, K.T, atol=1e-6)


def test_tanimoto_kernel_forward_extremes():
    kernel = TanimotoKernel()
    x1 = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    x2 = torch.tensor([[0, 1, 0]], dtype=torch.float32)
    sim = kernel(x1, x2).evaluate().item()
    assert 0.0 <= sim <= 1.0
    assert sim == 0.0  # no shared 1's


def test_custom_kernel_forward_shapes():
    kernel = CustomKernel()
    x = torch.rand(5, 41)
    K = kernel(x, x).evaluate()
    assert K.shape == (5, 5)
    assert torch.allclose(K, K.T, atol=1e-6)


