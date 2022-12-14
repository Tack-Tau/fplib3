#!/bin/bash
for i in {1..200} ; do grep "BFGS:" ${i}/slurm-* | tail -1 | awk '{print $4 }' ; done > E_min_list.dat
cat -n E_min_list.dat | sort -gk 2 > sorted_E_min_list.dat ; sed -i 's/*//g' sorted_E_min_list.dat