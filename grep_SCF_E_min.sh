#!/bin/bash
function exists_in_list() {
  LIST=$1
  DELIMITER=$2
  VALUE=$3
  echo $LIST | tr "$DELIMITER" '\n' | grep -F -q -x "$VALUE"
}
redo_dir="$(input="./redo_SCF_dir" ; while IFS= read -r line ; do printf "%s " $line ; done < "$input")"
for i in {1..200} ; do if exists_in_list "$redo_dir" " " $i ; then grep "FIRE:" fVfV/${i}/slurm-* | tail -1 | awk '{print $4 }' ; else grep "FIRE:" fV/${i}/slurm-* | tail -1 | awk '{print $4 }' ; fi ; done > SCF_E_min_list.dat
cat -n SCF_E_min_list.dat | sort -gk 2 > sorted_SCF_E_min_list.dat ; sed -i 's/*//g' sorted_SCF_E_min_list.dat