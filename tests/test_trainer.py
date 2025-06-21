# Auto-generated test stubs for `models.trainer`
import pytest
from alpfore.models.trainer import train_gp_model
import torch

def test_train_gp_model_raises_on_empty():
    with pytest.raises(ValueError):
        train_gp_model(torch.empty(0, 3), torch.empty(0), torch.empty(0))


def test_train_gp_model_runs_on_small_data():
    X = torch.rand(5, 3)
    Y = torch.rand(5)
    Yvar = 0.1 * torch.ones_like(Y)

    model = train_gp_model(X, Y, Yvar)
    assert hasattr(model, "posterior")
    post = model.posterior(X)
    assert post.mean.shape == (5, 1)
    assert post.variance.shape == (5, 1)


