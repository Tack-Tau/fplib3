#!/bin/sh
#SBATCH --partition main
#SBATCH --constraint="skylake|cascadelake"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=12:00:00

# For FP, QUIP, LJ, SFLJ
module purge
module use /projects/community/modulefiles
module load intel/17.0.4 python/3.8.5-gc563
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1
# For VASP
# export VASP_PP_PATH="/home/st962/apps/" 

# For DFTB+
# module load intel/17.0.4 python/3.8.5-gc563 gcc/10.2.0-bz186 cmake/3.19.5-bz186
# export DFTB_COMMAND=$HOME/.local/bin/dftb+/bin/dftb+
# export DFTB_PREFIX=$HOME/apps/dftbplus/external/slakos/origin/pbc-0-3/


python3 mixing_test.py
