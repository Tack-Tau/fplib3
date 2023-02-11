#!/bin/bash
for i in {1..200} ; do cd ${i} ; vaspkit -task 601 > vaspkit.out ; cd .. ; done
for i in {1..200} ; do grep "Space Group:" ${i}/vaspkit.out | awk '{print $3 }' ; done > space_list.dat
for i in {1..200} ; do grep "Symmetry Operations:" ${i}/vaspkit.out | awk '{print $3 }' ; done > sym_op_list.dat
