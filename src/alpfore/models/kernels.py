import torch
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.priors import GammaPrior


class TanimotoKernel(Kernel):
    """Implements the Tanimoto similarity kernel for binary sequence vectors."""
    has_lengthscale = True

    def forward(self, x1, x2, **params):
        # Handle potential 3D input by squeezing
        if x1.dim() > 2:
            x1 = x1.squeeze(1)
        if x2.dim() > 2:
            x2 = x2.squeeze(1)

        dot = torch.matmul(x1, x2.T)
        norm_x1 = torch.sum(x1**2, dim=-1, keepdim=True)
        norm_x2 = torch.sum(x2**2, dim=-1, keepdim=True).T
        tanimoto = dot / (norm_x1 + norm_x2 - dot + 1e-6)
        distance = 1.0 - tanimoto

        return torch.exp(-distance / self.lengthscale)


class CustomKernel(Kernel):
    """Product kernel combining RBFs for scalar features and a Tanimoto kernel for sequences."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # RBF kernels for scalar features
        self.rbf_ssl = ScaleKernel(
            RBFKernel(ard_num_dims=1, lengthscale_prior=GammaPrior(3.0, 3.0))
        )
        self.rbf_lsl = ScaleKernel(
            RBFKernel(ard_num_dims=1, lengthscale_prior=GammaPrior(3.0, 3.0))
        )
        self.rbf_sgd = ScaleKernel(
            RBFKernel(ard_num_dims=1, lengthscale_prior=GammaPrior(3.0, 3.0))
        )
        self.rbf_seqL = ScaleKernel(
            RBFKernel(ard_num_dims=1, lengthscale_prior=GammaPrior(3.0, 3.0))
        )
        # Tanimoto kernel for sequence features
        self.tanimoto = ScaleKernel(
            TanimotoKernel(lengthscale_prior=GammaPrior(3.0, 3.0))
        )

    def forward(self, x1, x2, **params):
        # Force batch dimension
        if x1.dim() < 3:
            x1 = x1.unsqueeze(1)
        if x2.dim() < 3:
            x2 = x2.unsqueeze(1)

        # Split features
        x1_ssl, x1_lsl, x1_sgd, x1_seqL, x1_seq = (
            x1[:, 0, 0].unsqueeze(-1),
            x1[:, 0, 1].unsqueeze(-1),
            x1[:, 0, 2].unsqueeze(-1),
            x1[:, 0, 3].unsqueeze(-1),
            x1[:, 0, 4:],
        )
        x2_ssl, x2_lsl, x2_sgd, x2_seqL, x2_seq = (
            x2[:, 0, 0].unsqueeze(-1),
            x2[:, 0, 1].unsqueeze(-1),
            x2[:, 0, 2].unsqueeze(-1),
            x2[:, 0, 3].unsqueeze(-1),
            x2[:, 0, 4:],
        )

        # Kernel components
        k_ssl = self.rbf_ssl(x1_ssl, x2_ssl, **params)
        k_lsl = self.rbf_lsl(x1_lsl, x2_lsl, **params)
        k_sgd = self.rbf_sgd(x1_sgd, x2_sgd, **params)
        k_seqL = self.rbf_seqL(x1_seqL, x2_seqL, **params)
        k_seq = self.tanimoto(x1_seq, x2_seq, **params)

        return k_ssl * k_lsl * k_sgd * k_seqL * k_seq

