#!/bin/sh
#SBATCH --partition main
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=02:00:00

module purge
module use /projects/community/modulefiles
module load intel/17.0.4 python/3.8.5-gc563
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1
export VASP_PP_PATH="/home/st962/apps/"

cp Li1.vasp POSCAR
python3 ase_test.py