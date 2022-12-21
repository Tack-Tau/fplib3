import os
import sys
import numpy as np
import writekp
import ase.io
from ase.optimize import BFGS, LBFGS, BFGSLineSearch, QuasiNewton, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory


atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'

'''
from ase.calculators.vasp import Vasp

kpoints = writekp.writekp(kgrid=0.07)
calc1 = Vasp( command = 'mpirun -n 16 /home/lz432/apps/vasp.6.3.0_intel/bin/vasp_std',
              xc = 'PBE',
              setups = 'recommended',
              txt = 'vasp.out',
              prec = 'Normal',
              # ediff = 1.0e-8,
              # ediffg = -1.0e-5,
              encut = 400.0,
              ibrion = -1, # No VASP relaxation
              nsw = 0, # Max. no of relaxation steps
              isif = 3,
              ismear = 0,
              sigma = 0.05,
              potim = 0.2,
              # lwave = False,
              # lcharge = False,
              # lplane = False,
              isym = 0,
              symprec = 1.0e-7,
              npar = 4,
              kpts = kpoints,
              gamma = True
              )


from ase.calculators.lj import LennardJones
calc1 = LennardJones()
calc1.parameters.epsilon = 1.0
calc1.parameters.sigma = 1.0
calc1.parameters.rc = 2.5
calc1.parameters.smooth = True

atoms.calc = calc1
print ("LJ_energy:\n", atoms.get_potential_energy())
print ("LJ_forces:\n", atoms.get_forces())
print ("LJ_stress:\n", atoms.get_stress())

##################################################################################################
Sigma gives a measurement of how close two nonbonding particles can get and is thus referred to as the van der Waals radius. It is equal to one-half of the internuclear distance between nonbonding particles.
Ideally, r_min == 2**(1/6) * sigma == 2.0 * r_cov, which means van der Waals radius is approximately two times larger than covalent radius.

Reference:
https://en.wikipedia.org/wiki/Lennard-Jones_potential
https://en.wikipedia.org/wiki/Van_der_Waals_radius
https://en.wikipedia.org/wiki/Covalent_radius
##################################################################################################
'''



from SF_LJ_api4ase import ShiftedForceLennardJones

calc1 = ShiftedForceLennardJones()
calc1.parameters.epsilon = np.array([1.00, 1.50, 0.50])
calc1.parameters.sigma = np.array([1.00, 0.80, 0.88])
calc1.parameters.rc = 2.5 * np.array([1.00, 0.80, 0.88])

atoms.calc = calc1
print ("SFLJ_energy:\n", atoms.get_potential_energy())
print ("SFLJ_forces:\n", atoms.get_forces())
print ("SFLJ_stress:\n", atoms.get_stress())



'''
from quippy.potential import Potential

calc1 = Potential(param_filename='./gp_iter6_sparse9k.xml')

from fplib3_api4ase import fp_GD_Calculator

atoms.calc = calc1
print ("GAP_energy:\n", atoms.get_potential_energy())
print ("GAP_forces:\n", atoms.get_forces())
print ("GAP_stress:\n", atoms.get_stress())
# fmax_1 = np.amax(np.absolute(atoms.get_forces()))



from ase.calculators.dftb import Dftb
import writekp

kpoints = writekp.writekp(kgrid=0.07)
calc1 = Dftb(atoms = atoms,
             kpts = tuple(kpoints),
             label = 'dftb')
atoms.calc = calc1
print ("DFTB_energy:\n", atoms.get_potential_energy())
print ("DFTB_forces:\n", atoms.get_forces())
print ("DFTB_stress:\n", atoms.get_stress())
# fmax_1 = np.amax(np.absolute(atoms.get_forces()))



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

atoms.calc = calc2
print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())
print ("fp_stress:\n", atoms.get_stress())
# fmax_2 = np.amax(np.absolute(atoms.get_forces()))

# f_ratio = fmax_1 / fmax_2



calc = MixedCalculator(calc1, calc2)
atoms.calc = calc
print ("mixed_energy:\n", atoms.get_potential_energy())
print ("mixed_forces:\n", atoms.get_forces())
print ("mixed_stress:\n", atoms.get_stress())
'''

############################## Relaxation type ############################## 
#     https ://wiki.fysik.dtu.dk/ase/ase/optimize.html#module-optimize      #
#     https ://wiki.fysik.dtu.dk/ase/ase/constraints.html                   #
#############################################################################

# af = atoms
# af = StrainFilter(atoms)
af = UnitCellFilter(atoms, scalar_pressure = 0.0)

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


