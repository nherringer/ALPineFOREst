import mdtraj as md
from alpfore.core.simulation import BaseSimulation

class LAMMPSDumpLoader(TrajPrepMixIn, BaseSimulation):
    """
    Load an existing LAMMPS trajectory

    Parameters
    ----------
    dump_path : str | Path          Path to dump.lammpstrj (or .xtc)
    top_path  : str | Path          Path to topology (PSF/PDB/etc.)
    stride    : int                 Keep every `stride`‑th frame
    n_equil   : int                 Skip first n_equil frames
    """

    def __init__(self, dump_path, top_path, stride=1, n_equil=0):
        self.dump_path, self.top_path = dump_path, top_path
        self.stride, self.n_equil = stride, n_equil

    def run(self):
        traj = md.load(self.dump_path, top=self.top_path, stride=self.stride)
        if self.n_equil:
            traj = traj[self.n_equil:]          # mdtraj slice view
        return MDTrajAdapter(traj)

