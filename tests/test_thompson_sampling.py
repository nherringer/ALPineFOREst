import torch
import pytest
from alpfore.candidate_selectors.thompson_sampling import select_ts_candidates


class DummyKernel:
    def forward(self, X1, X2):
        return torch.mm(X1, X2.T)  # Simplified kernel (linear dot product)


def test_select_ts_candidates_basic():
    train_X = torch.rand(5, 3)
    train_Y = torch.rand(5)
    test_X = torch.rand(10, 3)
    test_ids = torch.arange(10)

    selected = select_ts_candidates(
        model=None,
        test_X=test_X,
        test_ids=test_ids,
        kernel=DummyKernel(),
        train_X=train_X,
        train_Y=train_Y,
        num_samples=2,
        k_per_batch=3,
    )
    assert isinstance(selected, list)
    assert len(selected) == 6
    assert all(isinstance(idx, int) for idx in selected)


