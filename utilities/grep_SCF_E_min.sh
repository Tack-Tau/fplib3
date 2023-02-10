#!/bin/bash
function exists_in_list() {
  LIST=$1
  DELIMITER=$2
  VALUE=$3
  echo $LIST | tr "$DELIMITER" '\n' | grep -F -q -x "$VALUE"
}
redo_dir="$(input="./redo_SCF_dir" ; while IFS= read -r line ; do printf "%s " $line ; done < "$input")"
for i in {1..200} ; do if exists_in_list "$redo_dir" " " $i ; then grep -A1 "Final energy per atom is" fVfV/${i}/slurm-* | tail -1 | awk '{print $1 }' ; else grep -A1 "Final energy per atom is" fV/${i}/slurm-* | tail -1 | awk '{print $1 }' ; fi ; done > SCF_E_min_list.dat
sed -i 's/*//g' SCF_E_min_list.dat
cat -n SCF_E_min_list.dat | sort -gk 2 > sorted_SCF_E_min_list.dat