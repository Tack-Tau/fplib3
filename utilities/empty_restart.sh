#!/bin/bash
for i in {1..200} ; do grep -l "unlimited" ${i}/slurm-* ; done 1> /dev/null 2> no_log
awk -F '/' '{print $1 }' no_log | awk -F ': ' '{print $2 }' > no_dir
input="./no_dir" ; while IFS= read -r line ; do cd ./$line ; rm slurm-* opt.* POSCAR ; cd .. ; done < "$input"
input="./no_dir" ; while IFS= read -r line ; do cp ../../struct/$line/POSCAR ./$line/POSCAR  ; done < "$input"
input="./no_dir" ; while IFS= read -r line ; do cd ./$line ; sbatch ase_fp_job.sh ; cd .. ; done < "$input"
