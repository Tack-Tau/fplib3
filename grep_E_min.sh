#!/bin/bash
for i in {1..200} ; do grep "BFGS:" ${i}/slurm-* | tail -1 | awk '{print $4 }' ; done > E_min_list.dat
