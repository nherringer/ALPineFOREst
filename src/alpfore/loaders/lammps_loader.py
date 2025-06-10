# src/alpfore/simulations/lammps_loader.py
from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import mdtraj as md
import glob

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
    n_equil_drop : int
        Drop the first `n_equil_drop` frames after loading.
    """

    def __init__(
        self,
        trj_path: Union[str, Path],
        struct_path: Union[str, Path],
        features: np.ndarray,
        stride: int = 1,
        n_equil_drop: int = 0,
        cand_list: Optional[List[int]] = None
    ):
        self.trj_path = Path(trj_path)
        self.struct_path = Path(struct_path)
        self.stride = stride
        self.n_equil_drop = n_equil_drop
        self.features = features
        self.cand_list = cand_list if cand_list is not None else []


    def run(self) -> Trajectory:
        dump = self.trj_path 
        top  = self.struct_path
        traj = md.load(dump, top=top, stride=self.stride)
        if self.n_equil_drop:
            traj = traj[self.n_equil_drop:]      # MDTraj slice view
        return SystemFeatureAdapter(traj, self.features)
    
    @classmethod
    def from_multi_dump(
        cls,
        traj_pattern: Union[str, Path],
        struct_path: Union[str, Path],
        features: np.ndarray,
        stride: int = 1,
        n_equil_drop: int = 0
    ) -> Trajectory:
        """
        Alternate constructor to concatenate multiple dump files.

        Parameters
        ----------
        traj_pattern : str or Path
            Glob pattern matching multiple dump files (e.g., 'prod*.lammpstrj').
        struct_path : Path-like
            Path to topology file.
        features : np.ndarray
            Feature array corresponding to all frames (or placeholder if computed later).
        stride : int
            Keep every `stride`-th frame from concatenated trajectories.
        n_equil_drop : int
            Drop the first `n_equil_drop` frames after loading.
        """
        traj_paths = sorted(glob.glob(str(traj_pattern)))
        if not traj_paths:
            raise FileNotFoundError(f"No trajectories matching pattern {traj_pattern}")

        traj = md.load(traj_paths, top=struct_path, stride=stride)
        if n_equil_drop:
            traj = traj[n_equil_drop:]

        return SystemFeatureAdapter(traj, features)

    @classmethod
    def from_candidate_list(
        cls,
        candidate_list: List[tuple],
        encoder: "SystemEncoder",
        struct_pattern: str,
        traj_pattern: str,
        stride: int = 1,
        n_equil_drop: int = 0,
        use_parallel: bool = False,
        n_jobs: int = -1,
    ):
        try:
            from joblib import Parallel, delayed
            joblib_available = True
        except ImportError:
            joblib_available = False
            use_parallel = False  # gracefully degrade to sequential

        def load_one(seq, ssl, lsl, sgd):
            features = encoder.encode(seq, ssl, lsl, sgd)
            struct_path_str = struct_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd)
            traj_pattern_str = traj_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd)
            return cls.from_multi_dump(
                traj_pattern=traj_pattern_str,
                struct_path=struct_path_str,
                features=features,
                stride=stride,
                n_equil_drop=n_equil_drop
            )

        if use_parallel and joblib_available:
            results = Parallel(n_jobs=n_jobs)(
                delayed(load_one)(seq, ssl, lsl, sgd) for (seq, ssl, lsl, sgd) in candidate_list
            )
            yield from results
        else:
            for seq, ssl, lsl, sgd in candidate_list:
                yield load_one(seq, ssl, lsl, sgd)
