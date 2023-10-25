import os
import sys
import numpy as np
import ase.io
from ase.io.trajectory import Trajectory
from ase.calculators.dftb import Dftb
import kgrid

fp_traj = Trajectory('fp_opt.traj')
DFTB_traj = Trajectory('opt.traj')
fp_count = 0
DFTB_count = 0
fp_max = 0
DFTB_max = 0
for atoms in fp_traj:
    fp_max = fp_max + 1

for atoms in DFTB_traj:
    DFTB_max = DFTB_max + 1

max_count = max(fp_max, DFTB_max)

for atoms in fp_traj:
    fp_count = fp_count + 1
    if fp_count <= max_count:
        calc = Dftb(atoms = atoms,
                     kpts = kgrid.calc_kpt_tuple(atoms),
                     label = 'dftb')
        atoms.calc = calc
        e = atoms.get_potential_energy() / len(atoms)
        f_max = np.amax( np.absolute( atoms.get_forces() ) )
        print(fp_count, e, f_max)

for atoms in DFTB_traj:
    DFTB_count = DFTB_count + 1
    if DFTB_count <= max_count:
        e = atoms.get_potential_energy() / len(atoms)
        f_max = np.amax( np.absolute( atoms.get_forces() ) )
        print(DFTB_count, e, f_max)
