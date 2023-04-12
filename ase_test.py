import os
import sys
import numpy as np
# import writekp
import ase.io
from ase.optimize import BFGS, LBFGS, BFGSLineSearch, QuasiNewton, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory

from fplib3_api4ase import fp_GD_Calculator

atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'
print("Number of atoms:", len(atoms))

calc = fp_GD_Calculator(
            cutoff = 6.0,
            contract = False,
            znucl = np.array([14], int),
            lmax = 0,
            nx = 300,
            ntyp = 1
            )
atoms.calc = calc

# calc.test_energy_consistency(atoms = atoms)
# calc.test_force_consistency(atoms = atoms)

print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())
print ("fp_stress:\n", atoms.get_stress())

############################## Relaxation type ############################## 
#     https ://wiki.fysik.dtu.dk/ase/ase/optimize.html#module-optimize      #
#     https ://wiki.fysik.dtu.dk/ase/ase/constraints.html                   #
#############################################################################

# af = atoms
# af = StrainFilter(atoms)
# mask = np.ones((3,3), dtype = int) - np.eye(3, dtype = int)
mask = np.eye(3, dtype = int)
af = UnitCellFilter(atoms, mask = mask, constant_volume = True, scalar_pressure = 0.0)

############################## Relaxation method ##############################\

# opt = BFGS(af, maxstep = 1.e-1, trajectory = trajfile)
opt = FIRE(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = True)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = SciPyFminBFGS(af, maxstep = 1.e-1, trajectory = trajfile)

opt.run(fmax = 1.e-3, steps = 5000)

traj = Trajectory(trajfile)
atoms_final = traj[-1]
ase.io.write('opt.vasp', atoms_final, direct = True, long_format=True, vasp5 = True)

final_cell = atoms.get_cell()
final_cell_par = atoms.cell.cellpar()
final_structure = atoms.get_scaled_positions()
final_energy_per_atom = float( atoms.get_potential_energy() / len(atoms_final) )
final_stress = atoms.get_stress()

print("Relaxed lattice vectors are \n{0:s}".\
      format(np.array_str(final_cell, precision=6, suppress_small=False)))
print("Relaxed cell parameters are \n{0:s}".\
     format(np.array_str(final_cell_par, precision=6, suppress_small=False)))
print("Relaxed structure in fractional coordinates is \n{0:s}".\
      format(np.array_str(final_structure, precision=6, suppress_small=False)))
print("Final energy per atom is \n{0:.6f}".format(final_energy_per_atom))
print("Final stress is \n{0:s}".\
      format(np.array_str(final_stress, precision=6, suppress_small=False)))



