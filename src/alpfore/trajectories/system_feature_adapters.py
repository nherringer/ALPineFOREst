# src/alpfore/trajectory/system_feature_adapter.py
from __future__ import annotations
import numpy as np

class SystemFeatureAdapter:
    """
    Minimal adapter that exposes **one** feature row for the whole trajectory.

    It still holds the underlying mdtraj.Trajectory so evaluators that need
    per-frame geometry can access it via the `.mdtraj` attribute.
    """

    def __init__(self, traj, features: np.ndarray):
        self._traj = traj
        self._features = features.reshape(1, -1)   # (1, d)

    # ------------------------------------------------------------------ #
    # Trajectory-protocol attributes
    # ------------------------------------------------------------------ #
    @property
    def n_frames(self) -> int:
        return 1                    # logical sample count for the Model

    def frame_descriptors(self) -> np.ndarray:
        return self._features       # (1, d)

    # Raw trajectory for evaluators that need geometry
    @property
    def mdtraj(self):
        return self._traj

