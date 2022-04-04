import fplib_GD
import rcovdata
import sys
import numpy as np

import f90test
import fplib3
# import fplib_GD.readvasp as readvasp
import ase.io
# import ase.units as units
from ase.atoms import Atoms
from ase.cell import Cell
# from ase.calculators.genericfileio import (CalculatorTemplate,
#                                            GenericFileIOCalculator)
# from ase.calculators.calculator import BaseCalculator, FileIOCalculator
from ase.calculators.calculator import Calculator


class fp_GD_Calculator(object):
    """Fingerprint calculator for ase"""
    
    def __init__(self, parameters=None, atoms=None, **kwargs):
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = dict(parameters)
        # self.parameters = dict(parameters)
        self.atoms = {}
        self.energy = None
        self.forces = None
        # self.results = None

    
    
    def check_restart(self, atoms=None, **kwargs):
        if (
            self.atoms
            and np.allclose(self.atoms.cell[:], atoms.cell[:])
            and np.allclose(self.atoms.get_scaled_positions(), atoms.get_scaled_positions())
            and self.energy is not None
            and self.forces is not None
        ):
            return False
        else:
            return True

    def get_potential_energy(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            # ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            
        # # energy = self.results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        # # energy = self.results["density"].grid.mp.asum(energy)
        # energy = fplib_GD.get_fp_energy(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
        #                                 lmax = 0, znucl = np.array([3], int), cutoff = 4.5)
        znucl = np.array([3], int)
        fp, dfp = fplib3.get_fp(False, 1, 100, 0, lat, rxyz, types, znucl, 6.0)
        e,f = fplib3.get_ef(fp, dfp)
        energy = e
        return energy

    def get_forces(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            # ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            # self.get_potential_energy(atoms)
        # forces = fplib_GD.get_FD_forces(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
        #                                 lmax = 0, znucl = np.array([3], int), cutoff = 4.5, \
        #                                 iter_max = 1, step_size = 1e-4) 
        znucl = np.array([3], int)
        fp, dfp = fplib3.get_fp(False, 1, 100, 0, lat, rxyz, types, znucl, 6.0)
        e,f = fplib3.get_ef(fp, dfp)
        forces = f
        return forces



    def get_stress(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            lat = atoms.cell[:]
            pos = atoms.get_scaled_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            self.get_potential_energy(atoms)
        stress = fplib_GD.get_FD_stress(lat, pos, types, contract = False, ntyp = 1, nx = 300, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                                        iter_max = 1, step_size = 1e-4)
        return stress
     
