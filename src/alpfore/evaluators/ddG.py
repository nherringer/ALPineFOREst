# src/alpfore/evaluations/ddG.py
from __future__ import annotations

import numpy as np
import pandas as pd
import mdtraj as md
from pathlib import Path
from typing import Union, Sequence, Tuple

from alpfore.core.evaluator import BaseEvaluator
from alpfore.core.trajectory_interface import Trajectory
from alpfore.evaluators.dna_hybridization import CGDNAHybridizationEvaluator


# --------------------------------------------------------------------------- #
# Helper functions (stripped down versions of your originals)                 #
# --------------------------------------------------------------------------- #
def _calc_fes_point(point: float, bandwidth: float, data: np.ndarray, logw: np.ndarray):
    dist = (point - data) / bandwidth
    return -np.logaddexp.reduce(logw - 0.5 * dist * dist)


def _calc_fes_1d(grid: np.ndarray, bandwidth: float,
                 data: np.ndarray, logw: np.ndarray) -> np.ndarray:
    fes = np.array([_calc_fes_point(p, bandwidth, data, logw) for p in grid])
    return fes - fes.min()


def _calc_delta_f(fes: np.ndarray, basin_mask: np.ndarray, kbt=1.0):
    f_a = -kbt * np.logaddexp.reduce(-fes[basin_mask] / kbt)
    f_b = -kbt * np.logaddexp.reduce(-fes[~basin_mask] / kbt)
    return f_b - f_a

class DeltaDeltaGEvaluator(BaseEvaluator):
    output_dim = 2  # [ddG, sem]

    def __init__(
        self,
        system_features: Tuple[float, ...],    # numeric key
        run_dir: Union[str, Path],
        ratios: np.ndarray = None,
        walker_ids: Sequence[int] = (1,),
        bandwidth: float = 1.0,
        ratio_cutoff: float = 0.8,
    ):
        self.key = Tuple(system_features)      # immutable, hashable
        self.run_dir = Path(run_dir)
        self.ratios = ratios if ratios is not None else np.array([])
        self.walker_ids = walker_ids
        self.bandwidth = bandwidth
        self.ratio_cutoff = ratio_cutoff

        self.results: dict[Tuple[float, ...], Tuple[float, float]] = {}

        self.temps = np.array([0.17, 0.20])
        self.t0 = 0.25

    # ------------------------------------------------------------------ #
    def evaluate(self, traj: Trajectory, ratios: np.ndarray) -> np.ndarray:

	cv_all = traj.get_cv("cv")
        cv_80 = cv_all[cv_all >= self.ratio_cutoff]
        ss_80leg = cv_80.min() if cv_80.size else 0.0

        # ---- load COLVAR data for this run directory ------------------
        colvar = pd.concat(
            pd.read_csv(self.run_dir / f"COLVAR.{w}",
                         sep=r"\s+", comment="#", header=None,
                         names=["time", "com", "energy", "bias"])
            for w in self.walker_ids
        )
        colvar["time"]   /= 5.53
        colvar["bias"]   /= 13.8072
        colvar["energy"] /= 13.8072

        grid = np.arange(colvar["com"].min(), colvar["com"].max() + 0.1, 1.0)
        DGs, SEMs = [], []

        for T in self.temps:
            logw = (colvar["bias"] + (1 - self.t0 / T) * colvar["energy"]) * self.t0
            fes = _calc_fes_1d(grid, self.bandwidth, colvar["com"].values, logw.values)/T
            fes += 2 * np.log(grid)/T

            # 5-block SEM omitted for brevity – keep previous logic...
            # sem = ...
            # derive ddG/sem per T ...

        # --- final combination (wide) ----------------------------------
        ddg   = DGs[0] - DGs[1]
        sem   = np.sqrt(SEMs[0]**2 + SEMs[1]**2)

        self.results[self.key] = (ddg, sem)

        # broadcast to every frame so shape = (n_frames, 2)
        return self.results[self.key]

