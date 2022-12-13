#!/bin/bash
for i in {1..200} ; do grep -l "unlimited" ${i}/slurm-* ; done 1> /dev/null 2> no_log
awk -F '/' '{print $1 }' no_log | awk -F ': ' '{print $2 }' > no_dir
input="./no_dir" ; while IFS= read -r line ; do cd ./$line ; rm slurm-* opt.* KPOINTS POSCAR CONTCAR OUTCAR WAVECAR ; cd .. ; done < "$input" 2> /dev/null
input="./no_dir" ; while IFS= read -r line ; do cp ../../struct/$line/BPOSCAR ./$line/POSCAR  ; done < "$input"
input="./no_dir" ; while IFS= read -r line ; do cd ./$line ; sbatch ase_fp_job.sh ; cd .. ; done < "$input"
