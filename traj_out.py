import os
import sys
import numpy as np
import ase.io
from ase.io.trajectory import Trajectory
traj = Trajectory('opt.traj')
atoms = traj[-1]
ase.io.write('opt.vasp', atoms, direct = True, long_format=True, vasp5 = True)
