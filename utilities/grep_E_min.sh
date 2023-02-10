#!/bin/bash
for i in {1..200} ; do grep -A1 "Final energy per atom is" ${i}/slurm-* | tail -1 | awk '{print $1 }' ; done > E_min_list.dat
sed -i 's/*//g' E_min_list.dat
cat -n E_min_list.dat | sort -gk 2 > sorted_E_min_list.dat