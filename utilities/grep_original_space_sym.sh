#!/bin/bash
for i in {1..200} ; do cd ../../../struct/${i} ; vaspkit -task 601 > vaspkit.out ; cd .. ; done
for i in {1..200} ; do grep "Space Group:" ../../../${i}/vaspkit.out | awk '{print $3 }' ; done > original_space_list.dat
for i in {1..200} ; do grep "Symmetry Operations:" ../../../${i}/vaspkit.out | awk '{print $3 }' ; done > original_sym_op_list.dat
mv original_space_list.dat original_space_list.dat.bk ; cat -n original_space_list.dat.bk | sort -gk 1 > original_space_list.dat ; rm original_space_list.dat.bk
mv original_sym_op_list.dat original_sym_op_list.dat.bk ; cat -n original_sym_op_list.dat.bk | sort -gk 1 > original_sym_op_list.dat ; rm original_sym_op_list.dat.bk