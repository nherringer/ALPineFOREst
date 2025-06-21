import pytest
import numpy as np
import mdtraj as md
from alpfore.trajectories.lammps_trajectory import LAMMPSTrajectory
from pathlib import Path


@pytest.fixture
def synthetic_traj():
    # Create a tiny synthetic mdtraj.Trajectory with 3 atoms and 5 frames
    positions = np.random.rand(5, 3, 3)  # shape: (n_frames, n_atoms, 3)
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    for _ in range(3):
        topology.add_atom("CA", element=md.element.carbon, residue=residue)

    traj = md.Trajectory(positions, topology)
    return traj


def test_lammpstrajectory_mdtraj(synthetic_traj):
    lammpstraj = LAMMPSTrajectory(synthetic_traj, run_dir=Path("."))
    assert isinstance(lammpstraj.mdtraj(), md.Trajectory)
    assert lammpstraj.mdtraj().n_atoms == 3
    assert lammpstraj.mdtraj().n_frames == 5


def test_lammpstrajectory_n_frames(synthetic_traj):
    lammpstraj = LAMMPSTrajectory(synthetic_traj, run_dir=Path("."))
    assert lammpstraj.n_frames() == 5


def test_lammpstrajectory_join_all(synthetic_traj):
    lammpstraj = LAMMPSTrajectory(synthetic_traj, run_dir=Path("."))
    joined = lammpstraj.join_all()
    assert isinstance(joined, LAMMPSTrajectory)
    assert joined.mdtraj().n_frames == 5
