#!/bin/sh
#SBATCH --partition main
#SBATCH --constraint="skylake|cascadelake"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=12:00:00

module purge
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1

# For Fplib, QUIP, LJ, SFLJ
# module use /projects/community/modulefiles
# module load intel/17.0.4 python/3.8.5-gc563

# For GULP
# module use /projects/community/modulefiles
# module load intel/17.0.4 python/3.8.5-gc563
# export GULP_LIB=$HOME/apps/gulp-4.4/Libraries
# export GULP_DOC=$HOME/apps/gulp-4.4/Docs
# export ASE_GULP_COMMAND="$HOME/bin/gulp < PREFIX.gin > PREFIX.got"

# For LAMMPS
# module load intel/17.0.4 python
# export LAMMPS_COMMAND="mpirun -np 8 $HOME/.local/bin/lmp"
# export ASE_LAMMPSRUN_COMMAND="mpirun -np 8 $HOME/.local/bin/lmp"
# export LAMMPS_POTENTIALS="$HOME/apps/lammps-29Sep2021/potentials/"

# For VASP
# module load intel/17.0.4
# export VASP_PP_PATH="/home/st962/apps/"

# For DFTB+
# module load intel/17.0.4 python/3.8.5-gc563 gcc/10.2.0-bz186 cmake/3.19.5-bz186
# export DFTB_COMMAND=$HOME/.local/bin/dftb+/bin/dftb+
# export DFTB_PREFIX=$HOME/apps/dftbplus/external/slako/pbc/pbc-0-3/

# For M3GNet
# source $HOME/.bashrc
# $HOME/apps/miniconda3/condabin/conda activate m3gnet
# export TF_ENABLE_ONEDNN_OPTS=0

python3 mixing_test.py > fp_log

python3 ase_test.py > log
