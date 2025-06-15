import torch
import numpy as np
from alpfore.core.model import BaseModel
from botorch.models import FixedNoiseGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import Kernel
from botorch.fit import fit_gpytorch_model

class GPRModel(BaseModel):
    def __init__(self, X_train, Y_train, Yvar, kernel: Kernel):
        self._X_train = torch.tensor(X_train, dtype=torch.float64)
        self._Y_train = torch.tensor(Y_train, dtype=torch.float64)
        self._Yvar = torch.tensor(Yvar, dtype=torch.float64)

        self.model = FixedNoiseGP(
            train_X=self._X_train,
            train_Y=self._Y_train,
            train_Yvar=self._Yvar,
            covar_module=kernel,
            outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

        self.model.eval()

    def predict(self, X: np.ndarray):
        X_torch = torch.tensor(X, dtype=torch.float64)
        with torch.no_grad():
            posterior = self.model(X_torch)
            mean = posterior.mean
            variance = posterior.variance
            mean_un, var_un = self.model.outcome_transform.untransform(mean, variance)
        return mean_un.numpy(), var_un.numpy()

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> torch.Tensor:
        X1_torch = torch.tensor(X1, dtype=torch.float64).unsqueeze(0)
        X2_torch = torch.tensor(X2, dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            return self.model.covar_module(X1_torch, X2_torch).evaluate().squeeze(0)

    @property
    def X_train(self):
        return self._X_train.numpy()

    @property
    def Y_train(self):
        return self._Y_train.numpy()

