import torch
import pytest
from alpfore.evaluators.ddG import evaluate_ddG


def test_evaluate_nonzero_ddG_positive_sem():
    F1 = torch.tensor([-1.0, -2.0, -3.0])
    F2 = torch.tensor([-0.5, -1.5, -2.5])
    T1 = 300.0
    T2 = 310.0

    ddg, sem = evaluate_ddG(F1, F2, T1, T2)
    assert isinstance(ddg, float)
    assert isinstance(sem, float)
    assert ddg != 0.0
    assert sem >= 0.0


