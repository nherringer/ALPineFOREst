# src/alpfore/trajectory/system_adapter.py
from __future__ import annotations
import numpy as np

class ConstantFeatureAdapter:
    """
    Wrap a trajectory object but override `frame_descriptors`
    to return the same feature vector for every frame.
    """
    def __init__(self, traj, features: np.ndarray):
        self._traj = traj
        self._features = features[None, :]          # shape (1, d)

    # The protocol attributes
    @property
    def n_frames(self) -> int:
        return self._traj.n_frames

    def frame_descriptors(self) -> np.ndarray:
        # broadcast to (n_frames, d) on demand
        return np.repeat(self._features, self._traj.n_frames, axis=0)

    # expose raw mdtraj object if evaluators need it
    @property
    def mdtraj(self):
        return self._traj

