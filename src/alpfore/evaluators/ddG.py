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


def _calc_fes_1d(
    grid: np.ndarray, bandwidth: float, data: np.ndarray, logw: np.ndarray
) -> np.ndarray:
    fes = np.array([_calc_fes_point(p, bandwidth, data, logw) for p in grid])
    return fes


def _calc_delta_f(fes: np.ndarray, basin_mask: np.ndarray, kbt=1.0):
    f_a = -kbt * np.logaddexp.reduce(-fes[basin_mask] / kbt)
    f_b = -kbt * np.logaddexp.reduce(-fes[~basin_mask] / kbt)
    return f_b - f_a


class DeltaDeltaGEvaluator(BaseEvaluator):
    output_dim = 2  # [ddG, sem]

    def __init__(
        self,
        system_features: Tuple[float, ...],  # numeric key
        run_dir: Union[str, Path],
        cv_cutoff: np.ndarray = None,
        walker_ids: Sequence[int] = (1,),
        bandwidth: float = 1.0,
    ):
        self.key = tuple(system_features)  # immutable, hashable
        self.run_dir = Path(run_dir)
        self.walker_ids = walker_ids
        self.bandwidth = bandwidth
        self.cv_cutoff = cv_cutoff

        self.results: dict[Tuple[float, ...], Tuple[float, float]] = {}

        self.temps = np.array([0.17, 0.20])
        self.t0 = 0.25

        # Unpack from key
        seq, ssl, lsl, sgd = system_features
        NP_r = 5  # or make this a global constant or input
        slen = len(seq)

        # Save system-specific derived constant
        self.ss_max2 = (ssl + NP_r + slen) * 2 - 1
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        traj: COLVARTrajectory,
        cv_cutoff: int = 20,
        cv_col: str = "cv",
        energy_col: str = "energy",
        bias_col: str = "bias",
        time_col: str = "time",
    ) -> Tuple[float, float]:

        colvar_df = traj.df.copy()

        # Unit conversions
        colvar_df[time_col] /= 5.53
        colvar_df[bias_col] /= 13.8072
        colvar_df[energy_col] /= 13.8072
        colvar_df = colvar_df[colvar_df[time_col] > 1000]
        cv_shuffled = colvar_df.sample(frac=1).reset_index(drop=True)
        start=int(len(cv_shuffled)%5)
        end=int(len(cv_shuffled[start:])/5)

        grid = np.arange(colvar_df[cv_col].min(), colvar_df[cv_col].max() + 0.1, 1.0)
        DGs, SEMs = [], []

        for T in self.temps:
            # --- Full FES calculation ---
            logw = (colvar_df[bias_col] + (1 - self.t0 / T) * colvar_df[energy_col]) * self.t0
            fes_full = _calc_fes_1d(grid, self.bandwidth, colvar_df[cv_col].values, logw.values) / T
            fes_full += 2 * np.log(grid) / T
            fes_full -= fes_full[-1]  # normalize so last value is zero

            # --- Block-averaged FES and SEM ---
            block_fes = []
            for b in range(5):
                block = slice(start + b * end, start + (b + 1) * end)
                logw_b = (cv_shuffled[bias_col][block] + (1 - self.t0 / T) * cv_shuffled[energy_col][block]) * self.t0
                fes_b = _calc_fes_1d(grid, self.bandwidth, cv_shuffled[cv_col][block], logw_b) / T
                fes_b += 2 * np.log(grid) / T
                fes_b -= fes_b[-1]
                block_fes.append(fes_b)

            sem = np.std(block_fes, axis=0) / np.sqrt(5)

            # --- ddG Calculation ---
            # Define grid-based regions
            ss_mask = (grid >= cv_cutoff) & (grid < self.ss_max2)
            unbound_mask = grid > self.ss_max2

            # Extract min FES values from defined regions
            T_min1 = np.min(fes_full[ss_mask])
            T_min2 = np.min(fes_full[unbound_mask])
            dg = T_min2 - T_min1

            # Get SEM at points corresponding to T_min1 and T_min2
            sem_T_min1 = sem[ss_mask][np.argmin(fes_full[ss_mask])]
            sem_T_min2 = sem[unbound_mask][np.argmin(fes_full[unbound_mask])]
            sem_dg = np.sqrt(sem_T_min1 ** 2 + sem_T_min2 ** 2)

            DGs.append(dg)
            SEMs.append(sem_dg)

        ddg = DGs[0] - DGs[1]
        sem = np.sqrt(SEMs[0] ** 2 + SEMs[1] ** 2)

        self.results[self.key] = (ddg, sem)
        return ddg, sem

