#!/bin/sh
#SBATCH --partition main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=12:00:00

module purge
module use /projects/community/modulefiles
module load intel/17.0.4 python/3.8.5-gc563
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1
export PATH=$HOME/apps/julia-1.8.1/bin:$PATH
export LD_LIBRARY_PATH=$HOME/apps/julia-1.8.1/lib:$HOME/apps/julia-1.8.1/lib/julia:$LD_LIBRARY_PATH

python3 mixing_test.py