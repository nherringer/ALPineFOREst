"""
Core abstractions for the *Loader* stage of ALPine FOREst.

A concrete Loader subclass is expected to:
1.  Generate or load trajectory data (e.g. launch LAMMPS, read a dump file).
2.  Return an object that exposes `frame_descriptors()` so the pipeline can
    turn frames into input vectors for the model.

Only the abstract interface lives here—no heavy MD libraries are imported.
Concrete implementations belong in `alpfore.loaders.*`.
"""

from __future__ import annotations  # allows "Trajectory" forward reference
import abc
from typing import Protocol


# --------------------------------------------------------------------------- #
# A minimal typing contract for whatever object your simulations return
# --------------------------------------------------------------------------- #
class Trajectory(Protocol):
    """Any trajectory object must provide per‑frame feature vectors."""
    def frame_descriptors(self) -> "np.ndarray": ...  # noqa: D401,E701


# --------------------------------------------------------------------------- #
# Abstract base class: Loader
# --------------------------------------------------------------------------- #
class BaseLoader(abc.ABC):
    """Abstract contract for the loader stage."""

    @abc.abstractmethod
    def run(self) -> Trajectory:
        """Load data and return a Trajectory."""
        ...
        # Concrete subclasses override this.


__all__ = ["Trajectory", "BaseLoader"]

