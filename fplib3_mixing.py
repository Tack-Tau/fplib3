import os
import sys
import numpy as np

from ase.cell import Cell
from ase.atoms import Atoms
from ase.calculators.calculator import BaseCalculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError, CalculatorSetupError

class Mixer:
    def __init__(self, calcs, atoms = None):
        if atoms is not None:
            self.atoms = atoms.copy()
        self.check_input(calcs, atoms)
        common_properties = set.intersection(*(set(calc.implemented_properties)
                                               for calc in calcs))
        self.implemented_properties = list(common_properties)
        if not self.implemented_properties:
            raise PropertyNotImplementedError('The provided Calculators have no'
                                              ' properties in common!')
        weights = np.ones(len(calcs)).tolist()
        pi_fmax = 1.0
        for i in range(len(calcs)):
            pi_fmax = pi_fmax*np.amax(np.absolute(clacs[i].get_forces(atoms)))
        for j in range(len(weights)):
            weights[j] = np.amax(np.absolute(clacs[j].get_forces(atoms))) / pi_fmax
        self.calcs = calcs
        self.weights = weights

    @staticmethod
    def check_input(calcs, atoms):
        if len(calcs) == 0:
            raise CalculatorSetupError('Please provide a list of Calculators')
        for calc in calcs:
            if not isinstance(calc, BaseCalculator):
                raise CalculatorSetupError('All Calculators should be inherited'
                                           ' form the BaseCalculator class')
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(atoms)

    def get_properties(self, properties, atoms):
        results = {}

        def get_property(prop):
            contributs = [calc.get_property(prop, atoms) for calc in self.calcs]
            results[f'{prop}_contributions'] = contributs
            results[prop] = sum(weight * value for weight, value
                                in zip(self.weights, contributs))

        for prop in properties:  # get requested properties
            get_property(prop)
        for prop in self.implemented_properties:  # cache all available props
            if all(prop in calc.results for calc in self.calcs):
                get_property(prop)
        return results


class LinearCombinationCalculator(BaseCalculator):
    """LinearCombinationCalculator for weighted summation of multiple calculators.
    """
    def __init__(self, calcs, weights):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        weights: list of float
            Weights for each calculator in the list.
        """
        super().__init__()
        self.mixer = Mixer(calcs, weights)
        self.implemented_properties = self.mixer.implemented_properties

    def calculate(self, atoms, properties, system_changes):
        """ Calculates all the specific property for each calculator and returns with the summed value.
        """
        self.atoms = atoms.copy()  # for caching of results
        self.results = self.mixer.get_properties(properties, atoms)

    def __str__(self):
        calculators = ', '.join(calc.__class__.__name__ for calc in self.mixer.calcs)
        return '{}({})'.format(self.__class__.__name__, calculators)


class MixedCalculator(LinearCombinationCalculator):
    """
    Mixing of two calculators with different weights

    H = weight1 * H1 + weight2 * H2

    Has functionality to get the energy contributions from each calculator

    Parameters
    ----------
    calc1 : ASE-calculator
    calc2 : ASE-calculator
    weight1 : float
        weight for calculator 1
    weight2 : float
        weight for calculator 2
    """
    def __init__(self, calc1, calc2, weight1, weight2):
        super().__init__([calc1, calc2], [weight1, weight2])

    def set_weights(self, w1, w2):
        self.mixer.weights[0] = w1
        self.mixer.weights[1] = w2

    def get_energy_contributions(self, atoms=None):
        """ Return the potential energy from calc1 and calc2 respectively """
        self.calculate(properties=['energy'], atoms=atoms, system_changes=all_changes)
        return self.results['energy_contributions']


class SumCalculator(LinearCombinationCalculator):
    """SumCalculator for combining multiple calculators.

    This calculator can be used when there are different calculators for the different chemical environment or
    for example during delta leaning. It works with a list of arbitrary calculators and evaluates them in sequence
    when it is required.
    The supported properties are the intersection of the implemented properties in each calculator.
    """
    def __init__(self, calcs):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        """

        weights = [1.] * len(calcs)
        super().__init__(calcs, weights)


class AverageCalculator(LinearCombinationCalculator):
    """AverageCalculator for equal summation of multiple calculators (for thermodynamic purposes)..
    """
    def __init__(self, calcs):
        """Implementation of average of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        """
        n = len(calcs)

        if n == 0:
            raise CalculatorSetupError('The value of the calcs must be a list of Calculators')

        weights = [1 / n] * n
        super().__init__(calcs, weights)


########################################################################################
####################### Helper functions for the VASP calculator #######################
########################################################################################

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
        raise CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, Atoms):
        raise CalculatorSetupError(
            ('Expected an Atoms object, '
             'instead got object of type {}'.format(type(atoms))))
