"""
alpfore.trajectory.adapters
---------------------------

Thin wrappers (“adapters”) that turn third-party trajectory objects into the
small interface expected by ALPine FOREst:

    * frame_descriptors()  -> np.ndarray[n_frames, d]
    * n_frames property    -> int
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import mdtraj as md
except ImportError:  # pragma: no cover
    md = None  # type: ignore[assignment]
    # Users who call MDTrajAdapter without mdtraj installed will get a RuntimeError


class MDTrajAdapter:
    """
    Wrap an :class:`mdtraj.Trajectory` and present the minimal Trajectory protocol.

    Parameters
    ----------
    traj
        An in-memory mdtraj.Trajectory already loaded by the caller.
    flatten_xyz
        If ``True`` (default) the per-frame descriptor is xyz flattened to
        shape (n_atoms * 3).  Override by subclassing or modifying the method
        to compute your own descriptor (e.g., distances, dihedrals, etc.).
    """

    def __init__(self, traj: "md.Trajectory", *, flatten_xyz: bool = True) -> None:
        if md is None:  # pragma: no cover
            raise RuntimeError(
                "MDTrajAdapter requires the mdtraj package. "
                "Install via `pip install mdtraj` or `pip install alpfore[io]`."
            )
        self._traj = traj
        self._flatten = flatten_xyz

    # ------------------------------------------------------------------ #
    # Properties and protocol methods                                    #
    # ------------------------------------------------------------------ #
    @property
    def n_frames(self) -> int:
        return self._traj.n_frames

    def frame_descriptors(self) -> np.ndarray:
        """
        Return an array of shape (n_frames, d).

        By default ``d = n_atoms * 3`` (flattened xyz).  Override if you need a
        different representation.
        """
        if self._flatten:
            return self._traj.xyz.reshape(self.n_frames, -1)
        # Example alternative: center-of-mass per residue, etc.
        raise NotImplementedError(
            "Custom frame descriptors not implemented; set `flatten_xyz=True`."
        )

    # ------------------------------------------------------------------ #
    # Convenience passthroughs (optional)                                #
    # ------------------------------------------------------------------ #
    @property
    def time(self) -> np.ndarray:
        """Simulation time in picoseconds (same as mdtraj units)."""
        return self._traj.time

    @property
    def topology(self) -> "md.Topology":
        return self._traj.topology

    def __repr__(self) -> str:  # for nicer debugging printouts
        atoms = self._traj.n_atoms
        return f"<MDTrajAdapter: {self.n_frames} frames, {atoms} atoms>"


