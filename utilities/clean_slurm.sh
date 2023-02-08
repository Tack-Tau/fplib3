#!/bin/bash
for i in {1..200} ; do op_cmd=`ls -lt ${i} | grep slurm | head -1 | awk -F '-' '{print $7 }' | awk -F '.' '{print $1 }'` ; echo $op_cmd ; done > slurm_no
readarray -t a_tmp < ./slurm_no ; for i in {1..200} ; do rm ${i}/slurm-${a_tmp[${i}-1]}.out ; done
