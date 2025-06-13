import pandas as pd
import numpy as np
import glob
from pathlib import Path
from typing import List, Tuple, Union, Optional
from alpfore.core.trajectory_interface import Trajectory
from alpfore.encoder import SystemEncoder

class COLVARLoader:
    def __init__(self, colvar_paths: List[Path], features: np.ndarray):
        self.colvar_paths = colvar_paths
        self.features = features

    def run(self) -> Trajectory:
        frames = []
        for path in self.colvar_paths:
            df = pd.read_csv(path, delim_whitespace=True, comment="#")
            frames.append(df)

        df_full = pd.concat(frames, ignore_index=True)
        run_dir = self.colvar_paths[0].parent  # use first file's directory
        return Trajectory(data=df_full, run_dir=run_dir)

    @classmethod
    def from_candidate_list(
        cls,
        candidate_list: List[Tuple],
        encoder: SystemEncoder,
        colvar_pattern: str,
    ):
        for seq, ssl, lsl, sgd in candidate_list:
            features = encoder.encode(seq, ssl, lsl, sgd)
            colvar_glob = colvar_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd)
            colvar_paths = sorted(glob.glob(colvar_glob))

            if not colvar_paths:
                raise FileNotFoundError(f"No COLVAR files found matching pattern: {colvar_glob}")

            colvar_paths = [Path(p) for p in colvar_paths]
            loader = cls(colvar_paths=colvar_paths, features=features)
            yield loader.run()

