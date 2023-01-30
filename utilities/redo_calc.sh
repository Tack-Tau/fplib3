#!/bin/bash
for i in {1..200} ; do grep -l "JOB\|CANCELLED" ${i}/slurm-* ; done 1> error_log 2> /dev/null
awk -F '/' '{print $1 }' error_log > error_dir
input="./error_dir" ; while IFS= read -r line ; do cd ./$line ; rm slurm-* ; bash traj2optvasp.sh ; cd .. ; done < "$input" 2> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cd ./$line ; cp opt.vasp POSCAR ; sbatch ase_fp_job.sh ; cd .. ; done < "$input"
