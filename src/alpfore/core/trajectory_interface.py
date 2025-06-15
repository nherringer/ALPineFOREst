from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Trajectory(ABC):
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    @abstractmethod
    def __getitem__(self, idx: Any):
        """Return a frame or slice of the trajectory."""
        pass

    def get_cv(self, key: str) -> Any:
        """Return a collective variable vector for this key, if available."""
        raise NotImplementedError("This trajectory type does not support CVs.")

    def mdtraj(self) -> Any:
        """Return the underlying mdtraj.Trajectory, if available."""
        raise NotImplementedError("This trajectory type does not support geometry.")

    def n_frames(self) -> int:
        raise NotImplementedError

    def join_all(self):
        """Return a single trajectory or dataframe joined across walkers (if applicable)."""
        raise NotImplementedError("This trajectory type does not support joining.")
