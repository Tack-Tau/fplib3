#!/bin/bash
for i in {1..200} ; do grep 'space_group_number:' ${i}/SYMM.out | awk '{printf $2; print ""}' ; done > space_list.dat
mv space_list.dat space_list.dat.bk ; cat -n space_list.dat.bk | sort -gk 1 > space_list.dat ; rm space_list.dat.bk