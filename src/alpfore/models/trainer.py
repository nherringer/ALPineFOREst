import torch
from gpytorch.settings import fast_pred_var
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from alpfore.utils.kernel_utils import compute_kernel_matrix, gpytorch_kernel_wrapper 


def train_gp_model(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    Y_var: torch.Tensor,
    kernel=None,
    standardize: bool = True,
) -> Tuple[FixedNoiseGP]:
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
    return model

# from sklearn.model_selection import LeaveOneOut
# def plot_loo_parity(train_X, train_Y, train_Yvar, kernel, save_path=None, batch_size=1000):
#     """
#     Leave-one-out cross-validation using matrix-based kernel prediction.

#     Args:
#         train_X (np.ndarray): Features (n, d)
#         train_Y (torch.Tensor): Target tensor (n, 1)
#         train_Yvar (torch.Tensor): Noise variances (n, 1) — unused for now
#         kernel: trained GPyTorch kernel
#         save_path (str): Optional path to save plot
#         batch_size (int): Batch size for kernel computation

#     Returns:
#         actuals (np.ndarray): True ddG values
#         preds (np.ndarray): Predicted ddG values
#     """
#     # Convert input
#     train_X_np = train_X if isinstance(train_X, np.ndarray) else train_X.numpy()
#     train_Y_np = train_Y.squeeze().numpy()

#     n = train_X_np.shape[0]
#     preds = []
#     actuals = train_Y_np

#     # Precompute K_train_train
#     K_full = compute_kernel_matrix(
#         train_X_np,
#         train_X_np,
#         kernel_func=kernel,
#         batch_size=batch_size,
#         save_dir=None,
#         return_file_paths=False,
#         verbose=False,
#     )

#     K_full = torch.tensor(K_full, dtype=torch.float64)

#     # Compute LOO predictions
#     for i in range(n):
#         # Exclude i-th row and column
#         mask = np.arange(n) != i
#         K_loo = K_full[mask][:, mask]
#         k_i = K_full[mask, i]  # (n-1,)
#         y_loo = torch.tensor(train_Y_np[mask], dtype=torch.float64)

#         try:
#             K_inv = torch.linalg.inv(K_loo)
#         except RuntimeError:
#             print(f"[Warning] Matrix inversion failed at index {i}")
#             preds.append(np.nan)
#             continue

#         # μ_i = k_i^T @ K_inv @ y_loo
#         mu_i = (k_i @ K_inv @ y_loo).item()
#         preds.append(mu_i)

#     preds = np.array(preds)

#     # Plot
#     plt.figure(figsize=(6, 6))
#     plt.scatter(preds, actuals, color="dodgerblue", edgecolors="k")
#     min_val, max_val = min(actuals.min(), preds.min()), max(actuals.max(), preds.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ideal")
#     plt.xlabel("Predicted ddG")
#     plt.ylabel("Actual ddG")
#     plt.title("Leave-One-Out Parity Plot")
#     plt.legend()
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300)
#     else:
#         plt.show()

#     return actuals, preds

