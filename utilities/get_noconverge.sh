#!/bin/bash
for i in {1..200} ; do cmd="$(grep "FIRE:" ${i}/log | tail -1 | awk ' {printf "%.6f\n", $5 }')" ; echo $i $cmd ; done > fmax_list.dat
join -1 1 -2 1 <(sort -gk 1 sorted_E_min_list.dat) <(sort -gk 1 fmax_list.dat) | awk '$3 > 0.002 { printf("%s\n",$1) }' > error_dir
