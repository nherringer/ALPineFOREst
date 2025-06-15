import pandas as pd
import numpy as np
import glob
from pathlib import Path
from typing import List, Tuple, Union, Optional
from alpfore.core.trajectory_interface import Trajectory
from alpfore.encoder import SystemEncoder
from alpfore.trajectories.colvar_trajectory import COLVARTrajectory

class COLVARLoader:
    def __init__(self, colvar_paths: List[Path], features: np.ndarray, names: List[str]):
        self.colvar_paths = colvar_paths
        self.features = features
        self.names = names

    def run(self) -> Trajectory:
        frames = []
        for path in self.colvar_paths:
            df = pd.read_csv(path, delim_whitespace=True, comment="#", names=self.names)
            frames.append(df)

        df_full = pd.concat(frames, ignore_index=True)
        run_dir = self.colvar_paths[0].parent  # use first file's directory
        return COLVARTrajectory(df=df_full, run_dir=run_dir)

    @classmethod
    def from_candidate_list(
        cls,
        candidate_list: List[Tuple],
        encoder: SystemEncoder,
        colvar_pattern: str,
        names:  List[str],
    ):
        for seq, ssl, lsl, sgd in candidate_list:
            features = encoder.encode(seq, ssl, lsl, sgd)
            colvar_glob = colvar_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd)
            colvar_paths = sorted(glob.glob(colvar_glob))

            if not colvar_paths:
                raise FileNotFoundError(
                    f"No COLVAR files found matching pattern: {colvar_glob}"
                )

            colvar_paths = [Path(p) for p in colvar_paths]
            loader = cls(colvar_paths=colvar_paths, features=features, names=names)
            yield loader.run()

    def join_all(trajectories: List[COLVARTrajectory]) -> COLVARTrajectory:
        """Concatenate data from multiple COLVARTrajectories into one."""
        if not trajectories:
            raise ValueError("No trajectories to join.")

        # Concatenate all DataFrames
        dfs = [traj.df for traj in trajectories]
        full_df = pd.concat(dfs, ignore_index=True)

        # Use shared run_dir and encoder from the first walker
        run_dir = trajectories[0].run_dir
        encoder = getattr(trajectories[0], "encoder", None)

        return COLVARTrajectory(df=full_df, run_dir=run_dir, encoder=encoder)

