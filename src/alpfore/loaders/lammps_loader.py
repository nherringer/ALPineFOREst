# src/alpfore/simulations/lammps_loader.py
from pathlib import Path
from typing import Union

import mdtraj as md

from alpfore.core.loader import BaseLoader, Trajectory
from alpfore.trajectories.adapters import MDTrajAdapter

class LAMMPSDumpLoader(BaseLoader):
    """
    Load an existing LAMMPS trajectory and present it as a Trajectory object.

    Parameters
    ----------
    run_dir : Path-like
        Directory that contains `dump.lammpstrj` and `topology.pdb`.
    stride : int
        Keep every `stride`-th frame.
    n_equil : int
        Drop the first `n_equil` frames after loading.
    """

    def __init__(
        self,
        trj_path: Union[str, Path],
        struct_path: Union[str, Path],
        stride: int = 1,
        n_equil: int = 0,
        features: np.ndarray
    ):
        self.trj_path = Path(trj_path)
        self.struct_path = Path(struct_path)
        self.stride = stride
        self.n_equil = n_equil
        self.features = features

    def run(self) -> Trajectory:
        dump = self.trj_path 
        top  = self.struct_path
        traj = md.load(dump, top=top, stride=self.stride)
        if self.n_equil:
            traj = traj[self.n_equil:]      # MDTraj slice view
        return ConstantFeatureAdapter(traj, self.features)


