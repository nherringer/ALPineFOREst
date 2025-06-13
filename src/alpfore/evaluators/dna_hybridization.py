# src/alpfore/evaluations/dna_hybridization.py
from __future__ import annotations

import numpy as np
import mdtraj as md

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
