import pandas as pd
from pathlib import Path
from alpfore.core.trajectory_interface import Trajectory


class COLVARTrajectory(Trajectory):
    def __init__(self, df: pd.DataFrame, run_dir: Path):
        super().__init__(run_dir)
        self.df = df

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def get_cv(self, key: str):
        return self.df[key].values

    def n_frames(self):
        return len(self.df)
