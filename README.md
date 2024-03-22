# Fingerprint library for crystal structures
### Implemented in Python3

## Dependencies
* Python >= 3.8.5
* Numpy >= 1.21.4
* Scipy >= 1.8.0
* Numba >= 0.56.2
* ASE >= 3.22.1

## Setup
For conda installation guide please visit their [website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) \
`conda create -n fplibenv python=3.8 pip ; conda activate fplibenv`\
`python3 -m pip install --user -U pip setuptools wheel numpy scipy ase numba`\
`git clone https://github.com/Tack-Tau/fplib3.git ./fplib3`

## Usage
### Basic ASE style documentation
See details for [ASE calculator class](https://wiki.fysik.dtu.dk/ase/development/calculators.html)
and [ASE calculator proposal](https://wiki.fysik.dtu.dk/ase/development/proposals/calculators.html#aep1)
```
    Fingerprint Calculator interface for ASE
    
        Implemented Properties:
        
            'energy': Sum of atomic fingerprint distance (L2 norm of two atomic 
                                                          fingerprint vectors)
            
            'forces': Gradient of fingerprint energy, using Hellmann–Feynman theorem
            
            'stress': Cauchy stress tensor using finite difference method
            
        Parameters:
        
            atoms:  object
                Attach an atoms object to the calculator.
                
            contract: bool
                Calculate fingerprint vector in contracted Guassian-type orbitals or not
            
            ntype: int
                Number of different types of atoms in unit cell
            
            nx: int
                Maximum number of atoms in the sphere with cutoff radius for specific cell site
                
            lmax: int
                Integer to control whether using s orbitals only or both s and p orbitals for 
                calculating the Guassian overlap matrix (0 for s orbitals only, other integers
                will indicate that using both s and p orbitals)
                
            cutoff: float
                Cutoff radius for f_c(r) (smooth cutoff function) [amp], unit in Angstroms
                
```


### Calling fplib3 calculator from ASE API
```python
import numpy as np
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

from functools import reduce

chem_nums = list(atoms.numbers)
znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
ntyp = len(znucl_list)
znucl = np.array(znucl_list, int)

calc = fp_GD_Calculator(
            cutoff = 6.0,
            contract = False,
            znucl = znucl,
            lmax = 0,
            nx = 300,
            ntyp = ntyp
            )

atoms.calc = calc

# calc.test_energy_consistency(atoms = atoms)
# calc.test_force_consistency(atoms = atoms)

print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())
print ("fp_stress:\n", atoms.get_stress())

# af = atoms
# af = StrainFilter(atoms)
af = UnitCellFilter(atoms, scalar_pressure = 0.0)

############################## Relaxation method ##############################\

# opt = BFGS(af, maxstep = 1.e-1, trajectory = trajfile)
opt = FIRE(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = True)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = SciPyFminBFGS(af, maxstep = 1.e-1, trajectory = trajfile)

opt.run(fmax = 1.e-5)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)
```
## Citation
If you use our code for your research please kindly cite our paper: [Accelerating Structural Optimization through Fingerprinting Space Integration on the Potential Energy Surface](https://pubs.acs.org/doi/10.1021/acs.jpclett.4c00275)
The portable BibTex style bibliography file are provided as following:
```
@article{taoAcceleratingStructuralOptimization2024,
  title = {Accelerating {{Structural Optimization}} through {{Fingerprinting Space Integration}} on the {{Potential Energy Surface}}},
  author = {Tao, Shuo and Shao, Xuecheng and Zhu, Li},
  year = {2024},
  month = mar,
  journal = {J. Phys. Chem. Lett.},
  volume = {15},
  number = {11},
  pages = {3185--3190},
  publisher = {American Chemical Society},
  doi = {10.1021/acs.jpclett.4c00275},
  abstract = {Structural optimization has been a crucial component in computational materials research, and structure predictions have relied heavily on this technique, in particular. In this study, we introduce a novel method that enhances the efficiency of local optimization by integrating extra fingerprint space into the optimization process. Our approach utilizes a mixed energy concept in the hyper potential energy surface (PES), combining real energy and a newly introduced fingerprint energy derived from the symmetry of the local atomic environment. This method strategically guides the optimization process toward high-symmetry, low-energy structures by leveraging the intrinsic symmetry of the atomic configurations. The effectiveness of our approach was demonstrated through structural optimizations of silicon, silicon carbide, and Lennard-Jones cluster systems. Our results show that the fingerprint space biasing technique significantly enhances the performance and probability of discovering energetically favorable, high-symmetry structures as compared to conventional optimizations. The proposed method is anticipated to streamline the search for new materials and facilitate the discovery of novel energetically favorable configurations.}
}
```
