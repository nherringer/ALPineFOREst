from pathlib import Path
from typing import Any, Union, List
import mdtraj as md

from alpfore.core.trajectory_interface import Trajectory


class LAMMPSTrajectory(Trajectory):
    def __init__(self, trajs: Union[md.Trajectory, List[md.Trajectory]], run_dir: Path):
        super().__init__(run_dir)

        if isinstance(trajs, list):
            self.traj = trajs[0].join(trajs)
        else:
            self.traj = trajs

    def __getitem__(self, idx: Any):
        return self.traj[idx]

    def mdtraj(self) -> md.Trajectory:
        return self.traj

    def n_frames(self) -> int:
        return self.traj.n_frames

    def join_all(self):
        return self  # Already joined in constructor

