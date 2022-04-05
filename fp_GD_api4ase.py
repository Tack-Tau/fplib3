import os
import sys
import numpy as np
import fplib_GD
import f90test
import fplib3
# import fplib_GD.readvasp as readvasp
import rcovdata

from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree

import ase
from ase.io import read, write
#from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
# import ase.units as units
from ase.atoms import Atoms
from ase.cell import Cell
# from ase.calculators.calculator import BaseCalculator, FileIOCalculator
from ase.calculators.calculator import Calculator
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.singlepoint import SinglePointDFTCalculator

# Ref: https://gitlab.com/ase/ase/-/blob/master/ase/calculators/calculator.py
#      https://gitlab.com/ase/ase/-/blob/master/ase/calculators/vasp/vasp.py
class fp_GD_Calculator(Calculator):
    """ASE interface for fp_GD, with the Calculator interface.

        Parameters:

            atoms:  object
                Attach an atoms object to the calculator.

            label: str
                Prefix for the output file, and sets the working directory.
                Default is 'fingerprint'.

            directory: str
                Set the working directory. Is prepended to ``label``.

            restart: str or bool
                Sets a label for the directory to load files from.
                if :code:`restart=True`, the working directory from
                ``directory`` is used.
                
    """
    name = 'fingerprint'
    ase_objtype = 'fingerprint_calculator'  # For JSON storage

    implemented_properties = [ 'energy', 'forces', 'stress' ]

    # Can be used later to set some ASE defaults
    default_parameters: Dict[str, Any] = {}

    def __init__(self,
                 atoms = None,
                 restart = None,
                 directory = '.',
                 label = 'fingerprint',
                 txt = 'vasp.out',
                 **kwargs):

        self._atoms = None
        self.results = {}
        # Initialize parameter dictionaries
        GenerateVaspInput.__init__(self)
        self._store_param_state()  # Initialize an empty parameter state

        # Store calculator from vasprun.xml here - None => uninitialized
        self._xml_calc = None

        # Set directory and label
        self.directory = directory
        
        if '/' in label:
            warn(('Specifying directory in "label" is deprecated, '
                  'use "directory" instead.'), np.VisibleDeprecationWarning)
            if self.directory != '.':
                raise ValueError('Directory redundantly specified though '
                                 'directory="{}" and label="{}".  '
                                 'Please omit "/" in label.'.format(
                                     self.directory, label))
            self.label = label
        else:
            self.prefix = label  # The label should only contain the prefix

        if isinstance(restart, bool):
            if restart is True:
                restart = self.label
            else:
                restart = None
        
        Calculator.__init__(
            self,
            restart=restart,
            label=self.label,
            atoms=atoms,
            **kwargs)

    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        fingerprint Calculator, then call the create_input.set()
        on remaining input keys.

        Allows for setting ``label``, ``directory`` and ``txt``
        without resetting the results in the calculator.
        """
        changed_parameters = {}

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

        if 'directory' in kwargs:
            # str() call to deal with pathlib objects
            self.directory = str(kwargs.pop('directory'))

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
                  atoms=None,
                  properties=('energy', ),
                  system_changes=tuple(calculator.all_changes)):
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
        
        '''
        command = self.make_command(self.command)
        self.write_input(self.atoms, properties, system_changes)

        with self._txt_outstream() as out:
            errorcode = self._run(command=command,
                                  out=out,
                                  directory=self.directory)

        if errorcode:
            raise calculator.CalculationFailed(
                '{} in {} returned an error: {:d}'.format(
                    self.name, self.directory, errorcode))
        '''
        
        # Read results from calculation
        self.update_atoms(atoms)
        self.read_results()

    def check_state(self, atoms, tol=1e-15):
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
            float_params=self.float_params.copy(),
            exp_params=self.exp_params.copy(),
            string_params=self.string_params.copy(),
            int_params=self.int_params.copy(),
            input_params=self.input_params.copy(),
            bool_params=self.bool_params.copy(),
            list_int_params=self.list_int_params.copy(),
            list_bool_params=self.list_bool_params.copy(),
            list_float_params=self.list_float_params.copy(),
            dict_params=self.dict_params.copy(),
            special_params=self.special_params.copy())

    def read(self, label=None):
        """Read results from VASP output files.
        Files which are read: OUTCAR, CONTCAR and vasprun.xml
        Raises ReadError if they are not found"""
        if label is None:
            label = self.label
        Calculator.read(self, label)

        # If we restart, self.parameters isn't initialized
        if self.parameters is None:
            self.parameters = self.get_default_parameters()

        # Check for existence of the necessary output files
        for f in ['OUTCAR', 'CONTCAR', 'vasprun.xml']:
            file = self._indir(f)
            if not file.is_file():
                raise calculator.ReadError(
                    'VASP outputfile {} was not found'.format(file))

        # Build sorting and resorting lists
        self.read_sort()

        # Read atoms
        self.atoms = self.read_atoms(filename=self._indir('CONTCAR'))

        # Read parameters
        self.read_incar(filename=self._indir('INCAR'))
        self.read_kpoints(filename=self._indir('KPOINTS'))
        self.read_potcar(filename=self._indir('POTCAR'))

        # Read the results from the calculation
        self.read_results()

    def _indir(self, filename):
        """Prepend current directory to filename"""
        return Path(self.directory).as_posix()+'/'+filename

    def read_sort(self):
        """Create the sorting and resorting list from ase-sort.dat.
        If the ase-sort.dat file does not exist, the sorting is redone.
        """
        sortfile = self._indir('ase-sort.dat')
        print(os.path.isfile(sortfile))
        if os.path.isfile(sortfile):
            self.sort = []
            self.resort = []
            with open(sortfile, 'r') as fd:
                for line in fd:
                    sort, resort = line.split()
                    self.sort.append(int(sort))
                    self.resort.append(int(resort))
        else:
            # Redo the sorting
            atoms = read(self._indir('CONTCAR'))
            self.initialize(atoms)


    def read_atoms(self, filename):
        """Read the atoms from file located in the VASP
        working directory. Normally called CONTCAR."""
        return read(filename)[self.resort]

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

    def read_results(self):
        """Read the results from VASP output files"""
        # Temporarily load OUTCAR into memory
        outcar = self.load_file('OUTCAR')

        # Read the data we can from vasprun.xml
        calc_xml = self._read_xml()
        xml_results = calc_xml.results

        # Fix sorting
        xml_results['forces'] = xml_results['forces'][self.resort]

        self.results.update(xml_results)

        # Stress is not always present.
        # Prevent calculation from going into a loop
        if 'stress' not in self.results:
            self.results.update(dict(stress=None))

        self._set_old_keywords()

        # Store the parameters used for this calculation
        self._store_param_state()

    def _set_old_keywords(self):
        """Store keywords for backwards compatibility wd VASP calculator"""
        self.spinpol = self.get_spin_polarized()
        self.energy_free = self.get_potential_energy(force_consistent=True)
        self.energy_zero = self.get_potential_energy(force_consistent=False)
        self.forces = self.get_forces()
        self.fermi = self.get_fermi_level()
        self.dipole = self.get_dipole_moment()
        # Prevent calculation from going into a loop
        self.stress = self.get_property('stress', allow_calculation=False)
        self.nbands = self.get_number_of_bands()


    def load_file(self, filename):
        """Reads a file in the directory, and returns the lines

        Example:
        >>> outcar = load_file('OUTCAR')
        """
        filename = self._indir(filename)
        with open(filename, 'r') as fd:
            return fd.readlines()

    @contextmanager
    def load_file_iter(self, filename):
        """Return a file iterator"""

        filename = self._indir(filename)
        with open(filename, 'r') as fd:
            yield fd

    def read_outcar(self, lines=None):
        """Read results from the OUTCAR file.
        Deprecated, see read_results()"""
        if not lines:
            lines = self.load_file('OUTCAR')

        # XXX: Do we want to read all of this again?
        self.energy_free, self.energy_zero = self.read_energy(lines=lines)
        self.forces = self.read_forces(lines=lines)
        self.stress = self.read_stress(lines=lines)

    def _read_xml(self) -> SinglePointDFTCalculator:
        """Read vasprun.xml, and return the last calculator object.
        Returns calculator from the xml file.
        Raises a ReadError if the reader is not able to construct a calculator.
        """
        file = self._indir('vasprun.xml')
        incomplete_msg = (
            f'The file "{file}" is incomplete, and no DFT data was available. '
            'This is likely due to an incomplete calculation.')
        try:
            _xml_atoms = read(file, index=-1, format='vasp-xml')
            # Silence mypy, we should only ever get a single atoms object
            assert isinstance(_xml_atoms, ase.Atoms)
        except ElementTree.ParseError as exc:
            raise calculator.ReadError(incomplete_msg) from exc

        if _xml_atoms is None or _xml_atoms.calc is None:
            raise calculator.ReadError(incomplete_msg)

        self._xml_calc = _xml_atoms.calc
        return self._xml_calc

    @property
    def _xml_calc(self) -> SinglePointDFTCalculator:
        if self.__xml_calc is None:
            raise RuntimeError(('vasprun.xml data has not yet been loaded. '
                                'Run read_results() first.'))
        return self.__xml_calc

    @_xml_calc.setter
    def _xml_calc(self, value):
        self.__xml_calc = value

    # Methods for reading information from OUTCAR files:
    def read_energy(self, all=None, lines=None):
        """Read energy from OUTCAR."""
        if not lines:
            lines = self.load_file('OUTCAR')

        [energy_free, energy_zero] = [0, 0]
        if all:
            energy_free = []
            energy_zero = []
        for line in lines:
            # Free energy
            if line.lower().startswith('  free  energy   toten'):
                if all:
                    energy_free.append(float(line.split()[-2]))
                else:
                    energy_free = float(line.split()[-2])
            # Extrapolated zero point energy
            if line.startswith('  energy  without entropy'):
                if all:
                    energy_zero.append(float(line.split()[-1]))
                else:
                    energy_zero = float(line.split()[-1])
        return [energy_free, energy_zero]

    def read_forces(self, all=False, lines=None):
        """Method that reads forces from OUTCAR file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""

        if not lines:
            lines = self.load_file('OUTCAR')

        if all:
            all_forces = []

        for n, line in enumerate(lines):
            if 'TOTAL-FORCE' in line:
                forces = []
                for i in range(len(self.atoms)):
                    forces.append(
                        np.array(
                            [float(f) for f in lines[n + 2 + i].split()[3:6]]))

                if all:
                    all_forces.append(np.array(forces)[self.resort])

        if all:
            return np.array(all_forces)
        return np.array(forces)[self.resort]

    def read_stress(self, lines=None):
        """Read stress from OUTCAR."""
        if not lines:
            lines = self.load_file('OUTCAR')

        stress = None
        for line in lines:
            if ' in kB  ' in line:
                stress = -np.array([float(a) for a in line.split()[2:]])
                stress = stress[[0, 1, 2, 4, 5, 3]] * 1e-1 * ase.units.GPa
        return stress


    def get_potential_energy(self, atoms=None, **kwargs):
        if self.restart:
            # ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('POSCAR')
            
        # energy = self.results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        # energy = self.results["density"].grid.mp.asum(energy)
        # energy = fplib_GD.get_fp_energy(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
        #                                 lmax = 0, znucl = np.array([3], int), cutoff = 4.5)
        znucl = np.array([3], int)
        fp, dfp = fplib3.get_fp(False, 1, 100, 0, lat, rxyz, types, znucl, 6.0)
        e,f = fplib3.get_ef(fp, dfp)
        energy = e
        return energy

    def get_forces(self, atoms=None, **kwargs):
        if self.restart:
            # ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('POSCAR')
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
        if self.restart:
            lat = atoms.cell[:]
            pos = atoms.get_scaled_positions()
            types = fplib_GD.read_types('POSCAR')
            self.get_potential_energy(atoms)
        stress = fplib_GD.get_FD_stress(lat, pos, types, contract = False, ntyp = 1, nx = 300, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                                        iter_max = 1, step_size = 1e-4)
        return stress
     
#####################################
# Below defines helper functions
# for the VASP calculator
#####################################


def check_atoms(atoms: ase.Atoms) -> None:
    """Perform checks on the atoms object, to verify that
    it can be run by VASP.
    A CalculatorSetupError error is raised if the atoms are not supported.
    """

    # Loop through all check functions
    for check in (check_atoms_type, check_cell, check_pbc):
        check(atoms)


def check_cell(atoms: ase.Atoms) -> None:
    """Check if there is a zero unit cell.
    Raises CalculatorSetupError if the cell is wrong.
    """
    if atoms.cell.rank < 3:
        raise calculator.CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: ase.Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise calculator.CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: ase.Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, ase.Atoms):
        raise calculator.CalculatorSetupError(
            ('Expected an Atoms object, '
             'instead got object of type {}'.format(type(atoms))))
