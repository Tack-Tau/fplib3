#!/bin/bash
module purge
module use /projects/community/modulefiles
module load intel/17.0.4 python/3.8.5-gc563
for i in {1..200} ; do grep -l "JOB\|CANCELLED" ${i}/slurm-* ; done > error_log
awk -F '/' '{print $1 }' error_log > error_dir
input="./error_dir"
while IFS= read -r line
do
  cd ./$line
  rm slurm-*
  python3 traj_out.py
  cd ..
done < "$input"
while IFS= read -r line
do
  cd ./$line
  cp opt.vasp POSCAR
  sbatch ase_fp_job.sh
  cd ..
done < "$input"
