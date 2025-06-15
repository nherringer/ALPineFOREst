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

    # ------------------------------------------------------------------ #
    def evaluate(self, traj: Trajectory) -> np.ndarray:
        md_t = traj.mdtraj()

        # Build sticky-strand backbone index arrays for NP1
        first_sticky = (
            163 + self.gd_long * (2 * self.long_length + 24) + 2 * self.short_length
        )
        NP1_short_inds = [
            list(range(first_sticky, first_sticky + 2 * self.sticky_length, 2))
        ]
        for i in range(1, self.gd_short):
            start = NP1_short_inds[0][0] + i * 2 * (
                self.short_length + self.sticky_length
            )
            NP1_short_inds.append(list(range(start, start + 2 * self.sticky_length, 2)))
        NP1_short_inds = np.array(NP1_short_inds)

        # Build sticky indices for NP2
        second_sticky = (
            self.NP2_center
            + 163
            + self.gd_long * (2 * self.long_length + 24)
            + 2 * self.short_length
        )
        NP2_short_inds = [
            list(range(second_sticky, second_sticky + 2 * self.sticky_length, 2))
        ]
        for j in range(1, self.gd_short):
            start = NP2_short_inds[0][0] + j * 2 * (
                self.short_length + self.sticky_length
            )
            NP2_short_inds.append(list(range(start, start + 2 * self.sticky_length, 2)))
        NP2_short_inds = np.array(NP2_short_inds)

        # Pre-compute neighbours (NP1 bases close to NP2 bases)
        NP1_close = md.compute_neighbors(
            md_t,
            0.2,
            np.concatenate(NP2_short_inds) + 1,
            np.concatenate(NP1_short_inds) + 1,
        )

        legal, illegal = [], []

        for f in range(md_t.n_frames):
            legal_cnt = illegal_cnt = 0
            NP1_vecs, NP2_vecs = {}, {}

            for base_idx in NP1_close[f]:
                strand_id = int(
                    np.where(np.isin(NP1_short_inds, [base_idx - 1, base_idx]))[0]
                )

                if strand_id not in NP1_vecs:
                    # NP1 direction vector
                    np1_start = NP1_short_inds[strand_id][0]
                    np1_end = NP1_short_inds[strand_id][-1]
                    vec1 = md_t.xyz[f][np1_end] - md_t.xyz[f][np1_start]
                    vec1 /= np.linalg.norm(vec1)
                    NP1_vecs[strand_id] = vec1

                    # Partner strand on NP2
                    partner = md.compute_neighbors(
                        md_t[f], 0.2, [base_idx], np.concatenate(NP2_short_inds) + 1
                    )[0][0]
                    strand2_id = int(
                        np.where(np.isin(NP2_short_inds, [partner - 1, partner]))[0]
                    )

                    np2_start = NP2_short_inds[strand2_id][0]
                    np2_end = NP2_short_inds[strand2_id][-1]
                    vec2 = md_t.xyz[f][np2_end] - md_t.xyz[f][np2_start]
                    vec2 /= np.linalg.norm(vec2)
                    NP2_vecs[strand2_id] = vec2

                    if np.dot(vec1, vec2) < 0:
                        legal_cnt += 1
                    else:
                        illegal_cnt += 1

            legal.append(legal_cnt)
            illegal.append(illegal_cnt)

        return np.column_stack((legal, illegal))

def evaluate(self, traj: md.Trajectory) -> pd.DataFrame:

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


def compute_cv_cutoff(
    traj: Union[str, pd.DataFrame],
    system_features: Tuple[str, int, int, int],
    legal_thresh: float = 0.8,
) -> int:
    """
    Computes the CV cutoff value at which the fraction of legal bonds exceeds a threshold.
    Can accept either a precomputed CSV path or a raw dataframe.
    """
    if isinstance(traj, str):
        hat_df = pd.read_csv(traj)
    elif isinstance(traj, pd.DataFrame):
        hat_df = traj
    else:
        raise TypeError("traj must be a filepath or DataFrame.")

    # Bin CV values
    hat_df["CV_bin"] = np.round(hat_df["CV"]).astype(int)


    # Group and compute legal fraction per CV bin
    col_df = (
        hat_df.groupby("CV_bin", as_index=False)[["Legal", "Illegal"]]
        .sum()
        .assign(total=lambda df: df["Legal"] + df["Illegal"])
        .assign(frac_legal=lambda df: df["Legal"] / (df["total"] + 1e-10))
    )

    # Find first CV bin where fraction of legal bonds exceeds threshold
    passing_bins = col_df[col_df["frac_legal"] > legal_thresh]
    if passing_bins.empty:
        raise ValueError("No bins exceed legal threshold.")

    val = passing_bins["CV_bin"].iloc[0]
    return val

