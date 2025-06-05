# src/alpfore/simulations/lammps_loader.py
from pathlib import Path
from typing import Union
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

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Inverts `encode`.

        Parameters
        ----------
        X : array_like, shape (n, 40) or (40,)
            4 scaled meta-features  +  36 one-hot bits (12 × 3).

        Returns
        -------
        ndarray, shape (n, 4)  (dtype=object)
            Columns: sequence (str), ssl (int), lsl (int), sgd (int)
        """
        # ---------- normalise shape ----------
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != 40:
            raise ValueError("decode expects 40 columns")

        meta_scaled, one_hot = X[:, :4], X[:, 4:]

        # ---------- un-scale meta ----------
        rng = self.scalers  # config["scales"]
        def _unscale(key, v):
            lo, hi = rng[key]["min"], rng[key]["max"]
            return v * (hi - lo) + lo

        ssl    = _unscale("ssl",    meta_scaled[:, 0]).round().astype(int)
        lsl    = _unscale("lsl",    meta_scaled[:, 1]).round().astype(int)
        sgd    = _unscale("sgd",    meta_scaled[:, 2]).round().astype(int)
        L_true = _unscale("seqlen", meta_scaled[:, 3]).round().astype(int)  # 4–12

        # ---------- decode sequences ----------
        vocab = np.array(list("TAØ"))           # 0→T, 1→A, 2→Ø
        oh = one_hot.reshape(X.shape[0], 12, 3)

        seqs = []
        for row_oh, L in zip(oh, L_true):
            idx  = row_oh.argmax(axis=1)        # (12,)
            chars = vocab[idx]                  # array(['T','Ø','A',...])
            # keep only non-padding symbols
            seq  = "".join(c for c in chars if c != "Ø")
            if len(seq) != L:                   # guard against mismatch
                raise ValueError(
                    f"Decoded length {len(seq)} ≠ expected {L}. "
                    "Ensure encode/decode use identical padding."
                )
            seqs.append(seq)

        # ---------- assemble object array ----------
        out = np.empty((X.shape[0], 4), dtype=object)
        out[:, 0] = seqs
        out[:, 1] = ssl
        out[:, 2] = lsl
        out[:, 3] = sgd
        return out
    
    @classmethod
    def from_multi_dump(
        cls,
        traj_pattern: Union[str, Path],
        struct_path: Union[str, Path],
        features: np.ndarray,
        stride: int = 1,
        n_equil: int = 0
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
        n_equil : int
            Drop the first `n_equil` frames after loading.
        """
        traj_paths = sorted(glob.glob(str(traj_pattern)))
        if not traj_paths:
            raise FileNotFoundError(f"No trajectories matching pattern {traj_pattern}")

        traj = md.load(traj_paths, top=struct_path, stride=stride)
        if n_equil:
            traj = traj[n_equil:]

        return SystemFeatureAdapter(traj, features)

    @classmethod
    def from_candidate_list(
        cls,
        candidate_list: List[tuple],
        encoder: "SystemEncoder",
        struct_pattern: str,
        traj_pattern: str,
        stride: int = 1,
        n_equil: int = 0
    ) -> List[Trajectory]:
        """
        Load and process all candidate systems in a list using their encoded features.

        Parameters
        ----------
        candidate_list : list of (seq, ssl, lsl, sgd)
            Each tuple describes one system.
        encoder : SystemEncoder
            Object used to encode (seq, ssl, lsl, sgd) into numeric features.
        struct_pattern : str
            Format string with placeholders for {seq}, {ssl}, {lsl}, {sgd}.
            Example: '../.../{seq}/ssl{ssl}_lsl{lsl}_lgd1_sgd{sgd}/topology.pdb'
        traj_pattern : str
            Format string with placeholders for {seq}, {ssl}, {lsl}, {sgd}.
            Example: '../.../{seq}/ssl{ssl}_lsl{lsl}_lgd1_sgd{sgd}/prod*.lammpstrj'
        stride : int
            Stride applied to all trajectories.
        n_equil : int
            Number of equilibration frames to drop.

        Returns
        -------
        List[Trajectory]
            One trajectory per candidate system.
        """
        trajs = []

        for seq, ssl, lsl, sgd in candidate_list:
            features = encoder.encode(seq, ssl, lsl, sgd)

            struct_path = struct_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd)
            traj_path   = traj_pattern.format(seq=seq, ssl=ssl, lsl=lsl, sgd=sgd)

            traj = cls.from_multi_dump(
                traj_pattern=traj_path,
                struct_path=struct_path,
                features=features,
                stride=stride,
                n_equil=n_equil
            )
            trajs.append(traj)

        return trajs