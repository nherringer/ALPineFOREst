from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from botorch.models.transforms import Standardize
from botorch import settings as bts
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

import numpy as np
from alpfore.core.model import BaseModel


class FixedNoiseGPModel(BaseModel):
    """
    Minimal Fixed-Noise GP surrogate using BoTorch/GPyTorch.

    Notes
    -----
    • Assumes *single-output* regression.
    • Uses BoTorch's default exact marginal likelihood optimiser (L-BFGS-B).
    • Standardises Y internally; predictions are returned in *original* units.
    """

    def __init__(self, ard_dims: int | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ard_dims = ard_dims
        self.gp = None           # will hold the BoTorch model
        self._y_trsf = None      # Standardize transform

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """X : (n, d),  Y : (n, 2) where col-0 = value, col-1 = sem"""
        if Y.shape[1] != 2:
            raise ValueError("Expect Y[:, 0]=value, Y[:, 1]=SEM")

        x_t = torch.as_tensor(X, dtype=torch.double, device=self.device)
        y_t = torch.as_tensor(Y[:, 0:1], dtype=torch.double, device=self.device)
        y_var = torch.as_tensor(Y[:, 1:2] ** 2, dtype=torch.double, device=self.device)

        # Standardize the outputs to aid optimisation
        self._y_trsf = Standardize(m=1).to(self.device)
        y_t_std = self._y_trsf(y_t)

        self.gp = FixedNoiseGP(
            train_X=x_t, train_Y=y_t_std, train_Yvar=y_var,
            outcome_transform=None,
            input_transform=None,
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray):
        """
        Returns
        -------
        mean : (n, 1)  numpy
        var  : (n, 1)  numpy
        """
        if self.gp is None:
            raise RuntimeError("Model has not been fit() yet")

        self.gp.eval()
        self.gp.likelihood.eval()

        x_t = torch.as_tensor(X, dtype=torch.double, device=self.device)
        with torch.no_grad(), bts.fast_pred_var(True):
            posterior = self.gp.posterior(x_t)
            mean_std = self._y_trsf.untransform(posterior.mean)
            var_std = posterior.variance * (self._y_trsf.std**2)

        return mean_std.cpu().numpy(), var_std.cpu().numpy()

