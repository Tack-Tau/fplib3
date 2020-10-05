#!/usr/bin/python -u

import numpy as np
import fplib2
import sys


def readvasp(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())

    lat = np.array(buff[2:5], float) 
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    nat = sum(typt)
    pos = np.array(buff[7:7 + nat], float)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    rxyz = np.dot(pos, lat)
    #rxyz = pos
    return lat, rxyz, types

def test(v1, v2):
    ntyp = 1
    natx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([26], int)
    lat1, rxyz1, types = readvasp(v1)
    lat2, rxyz2, types = readvasp(v2)
    contract = False
    fp1 = fplib2.get_fp(contract, ntyp, natx, lmax, lat1, rxyz1, types, znucl, cutoff)
    fp2 = fplib2.get_fp(contract, ntyp, natx, lmax, lat2, rxyz2, types, znucl, cutoff)

    dist = fplib2.get_fpdist(ntyp, types, fp1, fp2)
    print ('fingerprint distance: ', dist)


if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    v2 = args[2]
    test(v1, v2)
