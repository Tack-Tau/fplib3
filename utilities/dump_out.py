import os
import sys
import numpy as np
import ase.io
from ase.io.lammpsrun import read_lammps_dump_text
f = open('.'+'/'+'lammps.dump', 'r')
atoms = read_lammps_dump_text(f, index = -1)
ase.io.write('opt.vasp', atoms, direct = True, long_format=True, vasp5 = True)
