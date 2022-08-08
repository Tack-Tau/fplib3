#!/bin/bash
for i in {1..200}
do
  cd ./${i}/
  grep "BFGS:" slurm-* | tail -1 | awk '{print $4 }' >> ../E_min_list.dat
  cd ..
done
