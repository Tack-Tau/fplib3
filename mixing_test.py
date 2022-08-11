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
from fplib3_mixing import MixedCalculator
# from ase.calculators.mixing import MixedCalculator
# from ase.calculators.vasp import Vasp
from ase.calculators.lj import LennardJones

atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'

'''
kpoints = writekp.writekp(kgrid=0.04)
calc1 = Vasp( command = 'mpirun -n 16 /home/lz432/apps/vasp.6.3.0_intel/bin/vasp_std', 
              xc = 'PBE', 
              setups = 'recommended', 
              txt = 'vasp.out',
              prec = 'Accurate',
              ediff = 1.0e-5,
              # ediffg = -1.0e-3,
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
'''

# calc1 = LennardJones()
# calc1.parameters.epsilon = 1.0
# calc1.parameters.sigma = 2.6
# calc1.parameters.rc = 12.0

from quippy.potential import Potential

calc1 = Potential(param_filename='./gp_iter6_sparse9k.xml')

calc2 = fp_GD_Calculator(
			cutoff = 6.0,
			contract = False,
			znucl = np.array([14], int),
			lmax = 0,
			nx = 300,
			ntyp = 1
			)
# calc = MixedCalculator(calc1, calc2)
# atoms.set_calculator(calc)

atoms.calc = calc1
print ("GAP_energy:\n", atoms.get_potential_energy())
print ("GAP_forces:\n", atoms.get_forces())
# fmax_1 = np.amax(np.absolute(atoms.get_forces()))

atoms.calc = calc2
print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())
# fmax_2 = np.amax(np.absolute(atoms.get_forces()))

# f_ratio = fmax_1 / fmax_2

calc = MixedCalculator(calc1, calc2)
atoms.calc = calc
print ("mixed_energy:\n", atoms.get_potential_energy())
print ("mixed_forces:\n", atoms.get_forces())

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


