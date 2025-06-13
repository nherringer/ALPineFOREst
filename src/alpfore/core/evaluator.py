"""
Core abstractions for the *Evaluation* stage of ALPine FOREst.

A concrete Evaluator transforms a Trajectory into **per‑frame target values** Y
(e.g. hybridization fraction, RMSD, etc.).  The shape contract is

    Y.shape == (n_frames, output_dim)

so that downstream models know the dimensionality of the quantity being
predicted.

Only the abstract interface lives here—no heavy MD/ML libraries are imported.
Concrete implementations belong in ``alpfore.evaluations.*``.
"""

from __future__ import annotations

import abc
from typing_extensions import Protocol, TYPE_CHECKING
from alpfore.core.trajectory_interface import Trajectory

if TYPE_CHECKING:  # avoid importing NumPy at runtime for speed
    import numpy as np


# --------------------------------------------------------------------------- #
# Abstract base class: Evaluator
# --------------------------------------------------------------------------- #
class BaseEvaluator(abc.ABC):
    """
    Transform a trajectory into target values *Y*.

    Sub‑classes **must** set ``output_dim`` (class attribute) and implement
    ``evaluate``.
    """

    # Each concrete evaluator must override this with an int > 0
    output_dim: int

    @abc.abstractmethod
    def evaluate(self, traj: "Trajectory") -> "np.ndarray":
        """
        Parameters
        ----------
        traj
            Trajectory returned by a Simulation.

        Returns
        -------
        np.ndarray
            Shape ``(n_frames, self.output_dim)``.
        """
        ...


__all__ = ["BaseEvaluator"]
