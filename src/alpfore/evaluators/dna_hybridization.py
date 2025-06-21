# src/alpfore/evaluations/dna_hybridization.py
from __future__ import annotations

import numpy as np
import mdtraj as md
import pandas as pd

from alpfore.core.evaluator import BaseEvaluator
from alpfore.core.trajectory_interface import Trajectory
from typing import List


class CGDNAHybridizationEvaluator(BaseEvaluator):
    """Counts anti-parallel (“legal”) and parallel (“illegal”) sticky-strand
    hybridisations for each frame of a trajectory."""

    output_dim = 2  # columns: [legal_count, illegal_count]

    def __init__(
        self,
        gd_long: int,
        gd_short: int,
        short_length: int,
        long_length: int,
        sticky_length: int,
        walker: int = 1,
    ):
        self.gd_long = gd_long
        self.gd_short = gd_short
        self.short_length = short_length
        self.long_length = long_length
        self.sticky_length = sticky_length
        self.walker = walker

        self.NP2_center = (
            163
            + gd_long * (2 * long_length + 24)
            + gd_short * (2 * short_length + 2 * sticky_length)
        )

def evaluate(self, trj: md.Trajectory) -> pd.DataFrame:

    traj = trj.mdtraj()
    # Compute interparticle distance per frame (CV)
    dists = md.compute_distances(traj, [[0, self.NP2_center]])[:, 0] * 10  # nm → Å

    legal = np.zeros(traj.n_frames, dtype=int)
    illegal = np.zeros(traj.n_frames, dtype=int)

    # Identify hybridized pairs using compute_neighbors
    NP1_inds = np.concatenate(self.NP1_short_inds) + 1
    NP2_inds = np.concatenate(self.NP2_short_inds) + 1
    hybrid_pairs = md.compute_neighbors(traj, self.hybrid_cutoff / 10, NP2_inds, NP1_inds)

    for frame_idx, neighbors in enumerate(hybrid_pairs):
        for i2, i1 in neighbors:
            # Convert from 1-based back to 0-based
            atom1 = i1 - 1
            atom2 = i2 - 1

            # Figure out which strand this pair belongs to
            for strand in self.strand_pairs:
                if atom1 in strand and atom2 in strand:
                    alpha_idx, omega_idx = strand
                    break
            else:
                continue  # Skip if no match

            # Direction vector of strand (omega - alpha)
            r = traj.xyz[frame_idx, omega_idx] - traj.xyz[frame_idx, alpha_idx]
            r /= np.linalg.norm(r)

            # Vector from alpha to NP2 center
            v = traj.xyz[frame_idx, self.NP2_center] - traj.xyz[frame_idx, alpha_idx]
            v /= np.linalg.norm(v)

            dot = np.dot(r, v)

            if dot < 0:
                legal[frame_idx] += 1
            else:
                illegal[frame_idx] += 1

    df = pd.DataFrame({
        "CV": dists,
        "Legal": legal,
        "Illegal": illegal
    })

    # Save result so it can be reused
    out_path = Path(self.run_dir) / "hybridization_data.csv"
    df.to_csv(out_path, index=False)

    return df

from typing import Union, Tuple, List
from pathlib import Path

def compute_cv_cutoff(
    traj: Union[str, pd.DataFrame, List[Tuple[str, int, int, int]]],
    system_features: Tuple[str, int, int, int] = None,
    legal_thresh: float = 0.8,
    write_path: Union[str, Path, None] = None,
    base_dir: Union[str, Path] = "../DFP_ActiveLearning/2.Calc_ddG"
) -> Union[int, pd.DataFrame]:
    """
    If `traj` is a string or DataFrame, compute cutoff for a single system.
    If `traj` is a list of tuples (candidate list), compute cutoffs for each system.
    If `write_path` is given, write resulting table of CV cutoffs to disk.
    """
    if isinstance(traj, (str, pd.DataFrame)):
        # --- Single-system behavior (original)
        if isinstance(traj, str):
            hat_df = pd.read_csv(traj)
        else:
            hat_df = traj

        hat_df["CV_bin"] = np.round(hat_df["CV"]).astype(int)
        col_df = (
            hat_df.groupby("CV_bin", as_index=False)[["Legal", "Illegal"]]
            .sum()
            .assign(total=lambda df: df["Legal"] + df["Illegal"])
            .assign(frac_legal=lambda df: df["Legal"] / (df["total"] + 1e-10))
        )

        passing_bins = col_df[col_df["frac_legal"] > legal_thresh]
        if passing_bins.empty:
            raise ValueError("No bins exceed legal threshold.")

        return passing_bins["CV_bin"].iloc[0]

    elif isinstance(traj, list):
        # --- Multi-system mode (loop over candidate list)
        records = []
        for seq, ssl, lsl, sgd in traj:
            path = Path(base_dir) / f"{seq}/ssl{ssl}_lsl{lsl}_lgd1_sgd{sgd}/ssl{ssl}_lsl{lsl}_lgd1_sgd{sgd}_allwalkers.csv"
            try:
                cutoff = compute_cv_cutoff(str(path), legal_thresh=legal_thresh)
                records.append({
                    "seq": seq,
                    "ssl": ssl,
                    "lsl": lsl,
                    "sgd": sgd,
                    "cutoff": cutoff
                })
            except Exception as e:
                print(f"[WARN] Skipping {path.name} due to: {e}")
                continue

        result_df = pd.DataFrame.from_records(records)

        if write_path is not None:
            Path(write_path).parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(write_path, index=False)

        return result_df

    else:
        raise TypeError("traj must be a filepath, DataFrame, or list of system tuples.")


