import numpy as np
from alpfore.core.trajectory_interface import Trajectory


class SystemFeatureAdapter(Trajectory):
    def __init__(self, traj, features: np.ndarray, run_dir=None):
        super().__init__(run_dir=run_dir)
        self._traj = traj
        self._features = features

    def __getitem__(self, idx):
        return self._traj[idx]

    def mdtraj(self):
        return self._traj

    def n_frames(self):
        return self._traj.n_frames
