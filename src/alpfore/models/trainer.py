import torch
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

def train_gp_model(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    Y_var: torch.Tensor,
    kernel,
    standardize: bool = True,
) -> Tuple[FixedNoiseGP, torch.Tensor, torch.Tensor]:
    """
    Fit a GPyTorch + BoTorch GP model to labeled data using a custom kernel.

    Args:
        X_train: [n, d] input tensor.
        Y_train: [n, 1] target values.
        Y_var: [n, 1] variances (squared SEM).
        kernel: GPyTorch kernel instance (e.g., CustomKernel()).
        standardize: Whether to apply outcome standardization.

    Returns:
        model: Trained GP model.
        mean: Posterior mean (unstandardized if standardize=True).
        variance: Posterior variance (unstandardized if standardize=True).
    """
    transform = Standardize(m=1) if standardize else None

    model = FixedNoiseGP(
        train_X=X_train,
        train_Y=Y_train,
        train_Yvar=Y_var,
        covar_module=kernel,
        outcome_transform=transform,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll, max_retries=10)

    model.eval()
    with torch.no_grad():
        posterior = model(X_train)
        mean, variance = posterior.mean, posterior.variance

    # Undo standardization if needed
    if standardize:
        mean, variance = model.outcome_transform.untransform(mean, variance)

    return model, mean, variance

def plot_loo_parity(train_X, train_Y, train_Yvar, kernel, save_path=None):
    """
    Performs leave-one-out cross-validation and plots predicted vs actual ddG values.

    Args:
        train_X (torch.Tensor): Feature tensor of shape [n, d].
        train_Y (torch.Tensor): Target tensor of shape [n, 1].
        train_Yvar (torch.Tensor): Noise variance tensor of shape [n, 1].
        kernel (gpytorch.kernels.Kernel): GPyTorch kernel module.
        save_path (str or Path, optional): If provided, saves the plot to this path.

    Returns:
        actuals (np.ndarray): True ddG values.
        preds (np.ndarray): Predicted ddG values.
    """
    loo = LeaveOneOut()
    actuals, preds = [], []

    train_Y_np = train_Y.squeeze().numpy()
    train_Yvar_np = train_Yvar.squeeze().numpy()

    for train_idx, test_idx in loo.split(train_X):
        X_train = train_X[train_idx]
        Y_train = train_Y[train_idx]
        Yvar_train = train_Yvar[train_idx]
        X_test = train_X[test_idx]

        model = train_gp_model(X_train, Y_train, Yvar_train, kernel)
        model.eval()
        with torch.no_grad():
            pred = model.posterior(X_test).mean.item()

        preds.append(pred)
        actuals.append(train_Y[test_idx].item())

    actuals = np.array(actuals)
    preds = np.array(preds)

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(preds, actuals, color="dodgerblue", edgecolors="k")
    min_val, max_val = min(actuals.min(), preds.min()), max(actuals.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ideal")
    plt.xlabel("Predicted ddG")
    plt.ylabel("Actual ddG")
    plt.title("Leave-One-Out Parity Plot")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return actuals, preds

