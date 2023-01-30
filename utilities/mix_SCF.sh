#!/bin/bash
cd fV ; bash grep_E_min.sh ; cat -n E_min_list.dat | sort -gk 2 > sorted_E_min_list.dat ; sed -i 's/*//g' sorted_E_min_list.dat ; cd ..
cd fVfV ; bash grep_E_min.sh ; cat -n E_min_list.dat | sort -gk 2 > sorted_E_min_list.dat ; sed -i 's/*//g' sorted_E_min_list.dat ; cd ..
join -1 1 -2 1 <(sort -gk 1 fV/sorted_E_min_list.dat) <(sort -gk 1 fVfV/sorted_E_min_list.dat) | awk '($2 - $3) > 1.0 { printf("%s\n",$1) }' > redo_SCF_dir
input="./redo_SCF_dir" ; while IFS= read -r line ; do rm -r fV/$line ; cp -r fVfV/$line fV/ ; done < "$input"
input="./redo_SCF_dir" ; while IFS= read -r line ; do cp fVfV/$line/opt.vasp fp/$line/POSCAR ; mv fVfV/$line/opt.vasp fVfV/$line/bk.opt.vasp ; mv fVfV/$line/opt.traj fVfV/$line/bk.opt.traj ; done < "$input"
input="./redo_SCF_dir" ; while IFS= read -r line ; do cd fp/$line ; rm slurm-* opt.* ; sbatch ase_fp_job.sh ; cd ../.. ; done < "$input"
