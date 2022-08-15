import os
import sys
import numpy as np
import writekp
import ase.io
from ase.optimize import BFGS, LBFGS, BFGSLineSearch, QuasiNewton, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory

from fplib3_api4ase import fp_GD_Calculator
# from fplib3_mixing import MixedCalculator
# from ase.calculators.mixing import MixedCalculator
# from ase.calculators.vasp import Vasp

atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'

calc = fp_GD_Calculator(
            cutoff = 6.0,
            contract = False,
            znucl = np.array([14], int),
            lmax = 0,
            nx = 300,
            ntyp = 1
            )
atoms.calc = calc
print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())

############################## Relaxation type ############################## 
#     https ://wiki.fysik.dtu.dk/ase/ase/optimize.html#module-optimize      #
#     https ://wiki.fysik.dtu.dk/ase/ase/constraints.html                   #
#############################################################################

# af = atoms
# af = StrainFilter(atoms)
af = UnitCellFilter(atoms)

############################## Relaxation method ##############################\

opt = BFGS(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = FIRE(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = True)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = SciPyFminBFGS(af, maxstep = 1.e-1, trajectory = trajfile)

opt.run(fmax = 1.e-5)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)



