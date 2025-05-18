from .kernels import TanimotoRBFKernel
import gpytorch

class GPyTorchGP(BaseModel):
    def __init__(self, ard_dims: int):
        self.covar_module = gpytorch.kernels.ScaleKernel(
            TanimotoRBFKernel()
        )

