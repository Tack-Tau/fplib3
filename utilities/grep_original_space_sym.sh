#!/bin/bash
result_dir="$(pwd)"
cd ../../../struct
for i in {1..200} ; do cd ${i} ; vaspkit -task 601 > vaspkit.out ; cd .. ; done
cd $result_dir
for i in {1..200} ; do grep "Space Group:" ../../../struct/${i}/vaspkit.out | awk '{print $3 }' ; done > original_space_list.dat
for i in {1..200} ; do grep "Symmetry Operations:" ../../../struct/${i}/vaspkit.out | awk '{print $3 }' ; done > original_sym_op_list.dat
mv original_space_list.dat original_space_list.dat.bk ; cat -n original_space_list.dat.bk | sort -gk 1 > original_space_list.dat ; rm original_space_list.dat.bk
mv original_sym_op_list.dat original_sym_op_list.dat.bk ; cat -n original_sym_op_list.dat.bk | sort -gk 1 > original_sym_op_list.dat ; rm original_sym_op_list.dat.bk