#!/bin/bash
function exists_in_list() {
  LIST=$1
  DELIMITER=$2
  VALUE=$3
  echo $LIST | tr "$DELIMITER" '\n' | grep -F -q -x "$VALUE"
}
redo_dir="$(input="./redo_SCF_dir" ; while IFS= read -r line ; do printf "%s " $line ; done < "$input")"
for i in {1..200} ; do if exists_in_list "$redo_dir" " " $i ; then cp fVfV/${i}/opt.vasp 10GPa/fp_result/${i}/POSCAR ; else cp fV/${i}/opt.vasp 10GPa/fp_result/${i}/POSCAR ; fi ; done
