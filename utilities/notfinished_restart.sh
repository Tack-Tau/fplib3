#!/bin/bash
for i in {1..200} ; do op_cmd="$(grep -A1 "Final energy per atom is" ${i}/slurm-* | tail -1 | awk '{print $1 }')" ; if [ -z $op_cmd ] ; then echo $i ; fi ; done > noresult_dir
input="./noresult_dir" ; while IFS= read -r line ; do cd ./$line ; rm slurm-* ; bash traj2optvasp.sh  ; cd .. ; done < "$input" 2> /dev/null
input="./noresult_dir" ; while IFS= read -r line ; do cd ./$line ; cp opt.vasp POSCAR ; sbatch ase_fp_job.sh ; cd .. ; done < "$input"
