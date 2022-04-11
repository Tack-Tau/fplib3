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
from ase.calculators.vasp import Vasp
from ase.calculators.mixing import MixedCalculator

atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'

kpoints = writekp.writekp(kgrid=0.04)
calc1 = Vasp( command = 'mpirun -n 16 /home/lz432/apps/vasp.6.3.0_intel/bin/vasp_std', 
              xc = 'PBE', 
              setups = 'recommended', 
              txt = 'vasp.out',
              prec = 'Accurate',
              ediff = 1E-5,
              # ediffg = -1E-3,
              encut = 520.0,
              ibrion = 2,
              isif = 3,
              nsw = 1,
              ismear = 0,
              sigma = 0.05,
              potim = 0.2,
              # lwave = False,
              # lcharge = False,
              # lplane = False,
              isym = 0,
              npar = 4,
              kpts = kpoints,
              )
calc2 = fp_GD_Calculator()
calc = MixedCalculator(calc1, calc2, 1, 10)
# atoms.set_calculator(calc)
atoms.calc = calc

print (atoms.get_potential_energy())
print (atoms.get_forces())

# ############################## Relaxation type ##############################
# '''
# Ref : 
#     https ://wiki.fysik.dtu.dk/ase/ase/optimize.html#module-optimize
#     https ://wiki.fysik.dtu.dk/ase/ase/constraints.html
# '''
af = atoms
# af = StrainFilter(atoms)
# # af = UnitCellFilter(atoms)
# ############################## Relaxation method ##############################
# opt = BFGS(af, maxstep = 1.e-1, trajectory = trajfile)
opt = FIRE(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = LBFGS(af, maxstep = 1.e-2, trajectory = trajfile, memory = 10, use_line_search = True)
# # opt = LBFGS(af, maxstep = 1.e-3, trajectory = trajfile, memory = 10, use_line_search = False)
# # opt = SciPyFminCG(af, trajectory = trajfile)
# # opt = SciPyFminBFGS(af, trajectory = trajfile)

opt.run(fmax = 0.001)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)



