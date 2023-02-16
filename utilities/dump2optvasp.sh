#!/bin/bash

module purge
module use /projects/community/modulefiles
module load intel/17.0.4 python/3.8.5-gc563
ulimit -s unlimited

python3 dump_out.py
