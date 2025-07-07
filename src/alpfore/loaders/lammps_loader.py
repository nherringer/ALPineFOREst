# src/alpfore/simulations/lammps_loader.py
from pathlib import Path
from typing import Union, Optional, List, Tuple
import numpy as np
import mdtraj as md
import glob
import os

from alpfore.core.loader import BaseLoader, Trajectory
from alpfore.trajectories.lammps_trajectory import LAMMPSTrajectory

class LAMMPSDumpLoader:
    @classmethod
    def from_candidate_list(cls, candidate_list, encoder, struct_pattern, traj_pattern, **kwargs):
        for system_features in candidate_list:
            #encoded = encoder.encode(*system_features)
            seq, ssl, lsl, sgd = system_features
            
            # Resolve paths
            struct_file = glob.glob(struct_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd))[0]
            if not struct_file:
                raise FileNotFoundError(
                        f"No structure (pdb) file found matching pattern: {struct_pattern}"
                )

            traj_files = sorted(glob.glob(traj_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd)))
            if not traj_files:
                raise FileNotFoundError(
                        f"No trajectory files found matching pattern: {traj_pattern}"
                )
            
            run_dir = os.getcwd()

            # Load each dump file independently
            trajs = [md.load(f, top=struct_file) for f in traj_files]

            # Wrap in a unified Trajectory interface
            yield LAMMPSTrajectory(trajs=trajs, run_dir=run_dir)

