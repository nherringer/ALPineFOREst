import torch
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.priors import GammaPrior
import numpy as np
from gpytorch.lazy import NonLazyTensor

class TanimotoKernel(Kernel):
    """Implements the Tanimoto similarity kernel for binary sequence vectors."""
    has_lengthscale = True

    def forward(self, x1, x2, **params):
        if x1.dim() > 2:
            x1 = x1.squeeze(1)
        if x2.dim() > 2:
            x2 = x2.squeeze(1)

        # Efficient pairwise Tanimoto similarity
        x1 = x1.float()
        x2 = x2.float()
        x1_sq = (x1 ** 2).sum(dim=1).unsqueeze(1)  # [N, 1]
        x2_sq = (x2 ** 2).sum(dim=1).unsqueeze(0)  # [1, M]
        dot = x1 @ x2.T                            # [N, M]
        tanimoto = dot / (x1_sq + x2_sq - dot + 1e-6)
        distance = 1.0 - tanimoto

        return NonLazyTensor(torch.exp(-distance / self.lengthscale))

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

        # Defensive check
        if x1.shape[-1] < 5 or x2.shape[-1] < 5:
            raise ValueError(f"Expected input feature dim ≥ 5, but got shapes {x1.shape} and {x2.shape}")

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

        # Product kernel
        result = k_ssl * k_lsl * k_sgd * k_seqL * k_seq

        # Explicit fix for scalar pairwise evaluations
        if result.shape == torch.Size([1, 1]):
            result = result.evaluate().unsqueeze(0)  # now [1, 1, 1]
            return result  # tensor, not LazyTensor — this is OK in pairwise

        # Remove batch dim for full [N, N] kernel matrix during training
        if result.dim() == 3 and result.shape[0] == 1 and result.shape[1] == result.shape[2]:
            result = result.squeeze(0)

        return result


#    def forward(self, x1, x2, **params):
#        # Ensure input is at least 3D
#        if x1.dim() < 3:
#            x1 = x1.unsqueeze(1)
#        if x2.dim() < 3:
#            x2 = x2.unsqueeze(1)
#
#        # Debug print
#        print("x1.shape:", x1.shape)
#        print("x2.shape:", x2.shape)
#
#        # Defensive check
#        if x1.shape[-1] < 5 or x2.shape[-1] < 5:
#            raise ValueError(f"Expected input feature dim ≥ 5, but got shapes {x1.shape} and {x2.shape}")
#
#        # Split features
#        x1_ssl, x1_lsl, x1_sgd, x1_seqL, x1_seq = (
#            x1[:, 0, 0].unsqueeze(-1),
#            x1[:, 0, 1].unsqueeze(-1),
#            x1[:, 0, 2].unsqueeze(-1),
#            x1[:, 0, 3].unsqueeze(-1),
#            x1[:, 0, 4:],
#        )
#        x2_ssl, x2_lsl, x2_sgd, x2_seqL, x2_seq = (
#            x2[:, 0, 0].unsqueeze(-1),
#            x2[:, 0, 1].unsqueeze(-1),
#            x2[:, 0, 2].unsqueeze(-1),
#            x2[:, 0, 3].unsqueeze(-1),
#            x2[:, 0, 4:],
#        )
#
#        # Kernel components
#        k_ssl = self.rbf_ssl(x1_ssl, x2_ssl, **params)
#        k_lsl = self.rbf_lsl(x1_lsl, x2_lsl, **params)
#        k_sgd = self.rbf_sgd(x1_sgd, x2_sgd, **params)
#        k_seqL = self.rbf_seqL(x1_seqL, x2_seqL, **params)
#        k_seq = self.tanimoto(x1_seq, x2_seq, **params)
#
#        result = k_ssl * k_lsl * k_sgd * k_seqL * k_seq
#
#        # CASE 1: Singleton pairwise eval — expand to [1, 1, 1]
#        if result.shape == torch.Size([1, 1]):
#            result = result.unsqueeze(0)
#
#        # CASE 2: Batched scalar eval (e.g., pairwise_kernels with multiple points) — keep [B, 1, 1]
#
#        # CASE 3: Full training eval — GPyTorch expects [N, N] not [1, N, N]
#        if result.dim() == 3 and result.shape[0] == 1 and result.shape[1] == result.shape[2]:
#            result = result.squeeze(0)
#                
#        return result



#    def forward(self, x1, x2, **params):
#        # Force batch dimension
#        if x1.dim() < 3:
#            x1 = x1.unsqueeze(1)
#        if x2.dim() < 3:
#            x2 = x2.unsqueeze(1)
#
#        print(x1.dim())
#        print(x2.dim())
#        print(np.shape(x1))
#        print(np.shape(x2))
#
#        # Split features
#        x1_ssl, x1_lsl, x1_sgd, x1_seqL, x1_seq = (
#            x1[:, 0, 0].unsqueeze(-1),
#            x1[:, 0, 1].unsqueeze(-1),
#            x1[:, 0, 2].unsqueeze(-1),
#            x1[:, 0, 3].unsqueeze(-1),
#            x1[:, 0, 4:],
#        )
#        x2_ssl, x2_lsl, x2_sgd, x2_seqL, x2_seq = (
#            x2[:, 0, 0].unsqueeze(-1),
#            x2[:, 0, 1].unsqueeze(-1),
#            x2[:, 0, 2].unsqueeze(-1),
#            x2[:, 0, 3].unsqueeze(-1),
#            x2[:, 0, 4:],
#        )
#
#        # Kernel components
#        k_ssl = self.rbf_ssl(x1_ssl, x2_ssl, **params)
#        k_lsl = self.rbf_lsl(x1_lsl, x2_lsl, **params)
#        k_sgd = self.rbf_sgd(x1_sgd, x2_sgd, **params)
#        k_seqL = self.rbf_seqL(x1_seqL, x2_seqL, **params)
#        k_seq = self.tanimoto(x1_seq, x2_seq, **params)
#
#        return k_ssl * k_lsl * k_sgd * k_seqL * k_seq
