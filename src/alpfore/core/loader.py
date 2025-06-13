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
from .trajectory_interface import Trajectory

try:
    from typing import Protocol  # Python 3.8+
except ImportError:
    from typing_extensions import Protocol  # Python 3.7 fallback

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


__all__ = ["BaseLoader"]
