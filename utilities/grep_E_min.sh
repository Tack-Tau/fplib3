#!/bin/bash
function exists_in_list() {
  LIST=$1
  DELIMITER=$2
  VALUE=$3
  echo $LIST | tr "$DELIMITER" '\n' | grep -F -q -x "$VALUE"
}
calculate() { printf "%.6s\n" "$@" | bc -l; }
for i in {1..200} ; do op_cmd="$(grep -A1 "Final energy per atom is" ${i}/log | tail -1 | awk '{print $1 }')" ; if [ -z $op_cmd ] ; then echo $i ; fi ; done > noresult_dir
notfinished_dir="$(input="./noresult_dir" ; while IFS= read -r line ; do printf "%s " $line ; done < "$input")"
for i in {1..200} ; do if exists_in_list "$notfinished_dir" " " $i ; then echo "scale=6; $(grep "FIRE:" ${i}/log | tail -1 | awk '{printf "%.6f", $4 }') / $(grep "Number of atoms:" ${i}/log | awk '{print $4 }') " | bc -l ; else grep -A1 "Final energy per atom is" ${i}/log | tail -1 | awk '{print $1 }' ; fi ; done > E_min_list.dat
sed -i 's/*//g' E_min_list.dat
cat -n E_min_list.dat | sort -gk 2 > sorted_E_min_list.dat