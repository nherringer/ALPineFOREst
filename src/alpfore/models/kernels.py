# kernels.py
import gpytorch

class TanimotoRBFKernel(gpytorch.kernels.Kernel):
    def forward(self, x1, x2, diag=False, **params):
        # custom similarity here
        ...

