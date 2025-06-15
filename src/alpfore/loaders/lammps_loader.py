# src/alpfore/simulations/lammps_loader.py
from pathlib import Path
from typing import Union, Optional, List, Tuple
import numpy as np
import mdtraj as md
import glob

from alpfore.core.loader import BaseLoader, Trajectory
from alpfore.trajectories.lammps_trajectory import LAMMPSTrajectory

class LAMMPSDumpLoader:
    @classmethod
    def from_candidate_list(cls, candidate_list, encoder, struct_pattern, traj_pattern, **kwargs):
        for system_features in candidate_list:
            encoded = encoder(*system_features)
            run_dir = Path(encoded)

            # Resolve paths
            struct_file = run_dir / struct_pattern
            traj_files = sorted(run_dir.glob(traj_pattern))

            # Load each dump file independently
            trajs = [md.load(str(f), top=str(struct_file)) for f in traj_files]

            # Wrap in a unified Trajectory interface
            yield LAMMPSTrajectory(trajs=trajs, run_dir=run_dir)

