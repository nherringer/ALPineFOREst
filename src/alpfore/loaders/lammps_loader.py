# src/alpfore/simulations/lammps_loader.py
from pathlib import Path
from typing import Union
import numpy as np
import mdtraj as md

from alpfore.core.loader import BaseLoader, Trajectory
from alpfore.trajectories.adapters import MDTrajAdapter
from alpfore.trajectories.system_feature_adapters import SystemFeatureAdapter

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
        features: np.ndarray,
        stride: int = 1,
        n_equil: int = 0
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
        return SystemFeatureAdapter(traj, self.features)


