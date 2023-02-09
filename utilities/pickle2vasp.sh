#!/bin/sh

#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=1:00:00

module purge
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1
# export VASP_PP_PATH="/home/st962/apps/"

source $HOME/.bashrc
$HOME/apps/miniconda3/condabin/conda activate m3gnet
python3 pickle_out.py 
