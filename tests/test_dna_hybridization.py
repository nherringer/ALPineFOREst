import pandas as pd
from alpfore.evaluators.dna_hybridization import compute_cv_cutoff
import pytest
import numpy as np
from alpfore.evaluators.dna_hybridization import CGDNAHybridizationEvaluator
from unittest.mock import MagicMock


def test_cgdna_evaluate_mocked():
    evaluator = CGDNAHybridizationEvaluator(
        gd_long=1, gd_short=1, short_length=2, long_length=4, sticky_length=2
    )
    evaluator.NP1_short_inds = [[0, 2]]
    evaluator.NP2_short_inds = [[4, 6]]
    evaluator.strand_pairs = [(0, 2)]
    evaluator.hybrid_cutoff = 5.0
    evaluator.run_dir = "."

    # Fake trajectory with 1 frame, 10 atoms
    class DummyTraj:
        n_frames = 1
        xyz = np.zeros((1, 10, 3))
        def __getitem__(self, idx): return self
    dummy_traj = DummyTraj()
    dummy_traj.xyz[0, 0] = [0, 0, 0]
    dummy_traj.xyz[0, 2] = [0, 0, 1]  # r vector
    dummy_traj.xyz[0, 6] = [0, 0, -1]  # NP2 center → v vector

    # Monkey patch
    from alpfore.evaluators import dna_hybridization
    dna_hybridization.md.compute_distances = MagicMock(return_value=np.array([[1.5]]))
    dna_hybridization.md.compute_neighbors = MagicMock(return_value=[[(6, 2)]])

    class DummyWrapper:
        def mdtraj(self): return dummy_traj

    out_df = evaluator.evaluate(DummyWrapper())
    assert isinstance(out_df, pd.DataFrame)
    assert out_df["Legal"].iloc[0] == 1
    assert out_df["Illegal"].iloc[0] == 0


def test_compute_cv_cutoff_single_df():
    df = pd.DataFrame({
        "CV": [1.0, 1.0, 2.0, 2.0],
        "Legal": [4, 3, 1, 0],
        "Illegal": [0, 1, 2, 3]
    })

    cutoff = compute_cv_cutoff(df, legal_thresh=0.6)
    assert isinstance(cutoff, int)
    assert cutoff == 1  # CV_bin 1 has 7 legal, 1 illegal → 87.5%


