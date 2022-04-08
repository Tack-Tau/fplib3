import os
import sys
import numpy as np
import f90test
import fplib3
import rcovdata

# from contextlib import contextmanager
# from pathlib import Path
# from warnings import warn
# from typing import Dict, Any
# from xml.etree import ElementTree

# import ase
from ase.io import read, write
#from ase.utils import PurePath
# import ase.units as units
from ase.atoms import Atoms
from ase.cell import Cell
from ase.calculators.calculator import Calculator, all_changes
# from ase.calculators.calculator import BaseCalculator, FileIOCalculator
# from ase.calculators.vasp.create_input import GenerateVaspInput
# from ase.calculators.singlepoint import SinglePointDFTCalculator

################################ ASE Reference ####################################
#      https://gitlab.com/ase/ase/-/blob/master/ase/calculators/calculator.py     #
#      https://gitlab.com/ase/ase/-/blob/master/ase/calculators/vasp/vasp.py      #
#      https://wiki.fysik.dtu.dk/ase/development/calculators.html                 #
###################################################################################
class fp_GD_Calculator(Calculator):
    """ASE interface for fp_GD, with the Calculator interface.
    
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
                
            lamx: int
                Integer to control whether using s orbitals only or both s and p orbitals for 
                calculating the Guassian overlap matrix (0 for s orbitals only, other integers
                will indicate that using both s and p orbitals)
                
    """
    # name = 'fingerprint'
    # ase_objtype = 'fingerprint_calculator'  # For JSON storage

    implemented_properties = [ 'energy', 'forces', 'stress' ]

    
    default_parameters = {
                          'contract': False,
                          'ntyp': 1,
                          'nx': 300,
                          'lmax': 0,
                          'cutoff': 6.0,
                          }
    
    nolabel = True

    def __init__(self,
                 atoms = None,
                 # restart = None,
                 **kwargs
                ):

        self._atoms = None
        self.results = {}
        # Initialize parameter dictionaries
        self._store_param_state()  # Initialize an empty parameter state

        '''
        if isinstance(restart, bool):
            if restart is True:
                restart = self.label
            else:
                restart = None
        '''
        
        Calculator.__init__(self,
                            # restart = restart,
                            atoms = atoms,
                            **kwargs
                           )

    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        fingerprint Calculator, then call the create_input.set()
        on remaining input keys.

        Allows for setting ``label``, ``directory`` and ``txt``
        without resetting the results in the calculator.
        """
        changed_parameters = {}

        if 'atoms' in kwargs:
            atoms = kwargs.pop('atoms')
            self.atoms = atoms  # Resets results

        changed_parameters.update(Calculator.set(self, **kwargs))

    def reset(self):
        self.atoms = None
        self.clear_results()

    def clear_results(self):
        self.results.clear()

    def calculate(self,
                  atoms = None,
                  properties = [ 'energy', 'forces', 'stress' ],
                  system_changes = tuple(all_changes),
                 ):
        """Do a VASP calculation in the specified directory.

        This will generate the necessary VASP input files, and then
        execute VASP. After execution, the energy, forces. etc. are read
        from the VASP output files.
        """
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(atoms)

        self.clear_results()

        if atoms is not None:
            self.atoms = atoms.copy()
        
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        # self.update_atoms(atoms)
        
        self.results['energy'] = self.get_potential_energy(atoms)
        self.results['forces'] = self.get_forces(atoms)
        self.results['stress'] = self.get_stress(atoms)
        

    def check_state(self, atoms, tol = 1e-15):
        """Check for system changes since last calculation."""
        def compare_dict(d1, d2):
            """Helper function to compare dictionaries"""
            # Use symmetric difference to find keys which aren't shared
            # for python 2.7 compatibility
            if set(d1.keys()) ^ set(d2.keys()):
                return False

            # Check for differences in values
            for key, value in d1.items():
                if np.any(value != d2[key]):
                    return False
            return True

        # First we check for default changes
        system_changes = Calculator.check_state(self, atoms, tol=tol)

        # We now check if we have made any changes to the input parameters
        # XXX: Should we add these parameters to all_changes?
        for param_string, old_dict in self.param_state.items():
            param_dict = getattr(self, param_string)  # Get current param dict
            if not compare_dict(param_dict, old_dict):
                system_changes.append(param_string)

        return system_changes

    def _store_param_state(self):
        """Store current parameter state"""
        self.param_state = dict(
            default_parameters=self.default_parameters.copy()
            )

    '''
    def update_atoms(self, atoms):
        """Update the atoms object with new positions and cell"""
        if (self.int_params['ibrion'] is not None
                and self.int_params['nsw'] is not None):
            if self.int_params['ibrion'] > -1 and self.int_params['nsw'] > 0:
                # Update atomic positions and unit cell with the ones read
                # from CONTCAR.
                atoms_sorted = read(self._indir('CONTCAR'))
                atoms.positions = atoms_sorted[self.resort].positions
                atoms.cell = atoms_sorted.cell

        self.atoms = atoms  # Creates a copy
    '''

    # Below defines some functions for faster access to certain common keywords
    
    @property
    def contract(self):
        """Access the contract in input_params dict"""
        return self.default_parameters['contract']

    @contract.setter
    def contract(self, contract):
        """Set contract in input_params dict"""
        self.default_parameters['contract'] = contract

    @property
    def ntyp(self):
        """Access the ntyp in input_params dict"""
        return self.default_parameters['ntyp']

    @ntyp.setter
    def ntyp(self, ntyp):
        """Set ntyp in input_params dict"""
        self.default_parameters['ntyp'] = ntyp

    @property
    def nx(self):
        """Access the nx in input_params dict"""
        return self.default_parameters['nx']

    @nx.setter
    def nx(self, nx):
        """Set ntyp in input_params dict"""
        self.default_parameters['nx'] = nx

    @property
    def lmax(self):
        """Access the lmax in input_params dict"""
        return self.default_parameters['lmax']

    @lmax.setter
    def lmax(self, lmax):
        """Set ntyp in input_params dict"""
        self.default_parameters['lmax'] = lmax

    @property
    def cutoff(self):
        """Direct access to the cutoff parameter"""
        return self.default_parameters['cutoff']

    @cutoff.setter
    def cutoff(self, cutoff):
        """Direct access for setting the cutoff parameter"""
        self.set(cutoff = cutoff)

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if atoms is None:
            self._atoms = None
            self.clear_results()
        else:
            if self.check_state(atoms):
                self.clear_results()
            self._atoms = atoms.copy()

    def check_restart(self, atoms = None, **kwargs):
        if (
            self.atoms
            and np.allclose(self.atoms.cell[:], atoms.cell[:])
            and np.allclose(self.atoms.get_scaled_positions(), atoms.get_scaled_positions())
            and self.results['energy'] is not None
            and self.results['forces'] is not None
            and self.results['stress'] is not None
        ):
            return False
        else:
            return True

    def get_potential_energy(self, atoms = None, **kwargs):
        if self.check_restart(atoms):
            # ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib3.read_types('POSCAR')
            
        
        znucl = np.array([3], int)
        fp, dfp = fplib3.get_fp(lat, rxyz, types, znucl,
                                contract = False,
                                ntyp = 1,
                                nx = 100,
                                lamx = 0,
                                cutoff = 6.0)
        e,f = fplib3.get_ef(fp, dfp)
        energy = e
        return energy

    def get_forces(self, atoms = None, **kwargs):
        if self.check_restart(atoms):
            # ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib3.read_types('POSCAR')
            # self.get_potential_energy(atoms) 
        znucl = np.array([3], int)
        fp, dfp = fplib3.get_fp(lat, rxyz, types, znucl,
                                contract = False,
                                ntyp = 1,
                                nx = 100,
                                lamx = 0,
                                cutoff = 6.0)
        e,f = fplib3.get_ef(fp, dfp)
        forces = f
        return forces

    def get_stress(self, atoms = None, **kwargs):
        if self.check_restart(atoms):
            lat = atoms.cell[:]
            pos = atoms.get_scaled_positions()
            types = fplib3.read_types('POSCAR')
            self.get_potential_energy(atoms)
        # stress = fplib_GD.get_FD_stress(lat, pos, types, contract = False, ntyp = 1, nx = 300, \
        #                                 lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
        #                                 iter_max = 1, step_size = 1e-4)
        return np.zeros(6)


#####################################
# Below defines helper functions
# for the VASP calculator
#####################################

def check_atoms(atoms: Atoms) -> None:
    """Perform checks on the atoms object, to verify that
    it can be run by VASP.
    A CalculatorSetupError error is raised if the atoms are not supported.
    """

    # Loop through all check functions
    for check in (check_atoms_type, check_cell, check_pbc):
        check(atoms)


def check_cell(atoms: Atoms) -> None:
    """Check if there is a zero unit cell.
    Raises CalculatorSetupError if the cell is wrong.
    """
    if atoms.cell.rank < 3:
        raise calculator.CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise calculator.CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, Atoms):
        raise calculator.CalculatorSetupError(
            ('Expected an Atoms object, '
             'instead got object of type {}'.format(type(atoms))))
