from gpytorch.lazy import LazyTensor

class TanimotoLazyTensor(LazyTensor):
    def __init__(self, sim_matrix):
        super().__init__()
        self.sim_matrix = sim_matrix

    def _matmul(self, rhs):
        return self.sim_matrix @ rhs

    def _size(self):
        return self.sim_matrix.size()

    def _transpose_nonbatch(self):
        return TanimotoLazyTensor(self.sim_matrix.transpose(-1, -2))

    def evaluate(self):
        return self.sim_matrix

