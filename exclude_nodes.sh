#!/bin/bash
for i in {1..200} ; do grep -l "Traceback" ${i}/slurm-* ; done 1> downed_job 2> /dev/null
for i in {1..200} ; do grep -l "JOB\|CANCELLED" ${i}/slurm-* ; done 1>> downed_job 2> /dev/null
for i in {1..200} ; do grep -l "Killed" ${i}/slurm-* ; done 1>> downed_job 2> /dev/null
cat downed_job | awk -F '-' '{print $2 }' | awk -F '.' '{print $1 }' | sort -u > node_log
input="./node_log" ; while IFS= read -r line ; do sacct -j $line --format=NodeList | grep hal | sort -u ; done < "$input" 1> node_tmp
sed -i 's/ //g' node_tmp ; cat node_tmp | sort -u > nodes_list
input="./nodes_list" ; while IFS= read -r line ; do echo ${line} | awk '{printf $1","}' ;  done < "$input" ; printf "\n"
