#!/usr/bin/env python3

import math
import os
import numpy as np



def get_fp_and_dfp(fvasp):
    vasp2ascii(fvasp)
    os.system('./fp_crystal.x')
    fp = np.loadtxt('fingerprint.dat')
    fpdt = np.loadtxt('fingerprint_derivative.dat')
    lenfp = len(fp[0])
    nat = len(fp)
    dfp = np.zeros((nat, nat, 3, lenfp))
    l = 0
    for iat in range(nat):
        for k in range(3):
            for jat in range(nat):
                dfp[jat][iat][k][:] = fpdt[l][:]
                l += 1
    return fp, dfp

def get_ef(fvasp):
    fp, dfp = get_fp_and_dfp(fvasp)
    nat = len(fp)
    e = 0.
    for i in range(nat):
        for j in range(nat):
            vij = fp[i] - fp[j]
            print (vij)
            t = np.dot(vij, vij)
            print ('t', t)
            e += t

    force = np.zeros((nat, 3))
    for k in range(nat):
        for i in range(nat):
            for j in range(nat):
                vij = fp[i] - fp[j]
                dvijx = dfp[k][i][0] - dfp[k][j][0]
                dvijy = dfp[k][i][1] - dfp[k][j][1]
                dvijz = dfp[k][i][2] - dfp[k][j][2]
                tx = -2 * np.dot(vij, dvijx)
                ty = -2 * np.dot(vij, dvijy)
                tz = -2 * np.dot(vij, dvijz)
                force[k][0] += tx
                force[k][1] += ty
                force[k][2] += tz
    return e, force

                

def vasp2ascii(fvasp):
    lat, rxyz, types, symbs = readvasp(fvasp)
    cons = lat2lcons(lat)
    latt = lcons2lat(cons)
    nat = len(rxyz)
    with open('input.ascii', 'w') as f:
        f.write("%4d\n" % nat)
        for x in latt:
            f.write("%15.9f  %15.9f  %15.9f\n" % tuple(x))
        for i in range(nat):
            f.write("%15.9f  %15.9f  %15.9f  %s\n" % 
            (rxyz[i][0], rxyz[i][1], rxyz[i][2], symbs[i]))    


def readvasp(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())

    lat = np.array(buff[2:5], float)
    symb = buff[5]
    typt = np.array(buff[6], int)
    nat = sum(typt)
    pos = np.array(buff[8:8 + nat], float)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    symbs = []
    for i in range(nat):
        symbs.append(symb[types[i]-1])
    rxyz = np.dot(pos, lat)
    return lat, rxyz, types, symbs

def lat2lcons(lat):
    ra = math.sqrt(lat[0][0]**2 + lat[0][1]**2 + lat[0][2]**2)
    rb = math.sqrt(lat[1][0]**2 + lat[1][1]**2 + lat[1][2]**2)
    rc = math.sqrt(lat[2][0]**2 + lat[2][1]**2 + lat[2][2]**2)

    cosa = (lat[1][0] * lat[2][0] + lat[1][1] * lat[2][1] +
            lat[1][2] * lat[2][2]) / rb / rc
    cosb = (lat[0][0] * lat[2][0] + lat[0][1] * lat[2][1] +
            lat[0][2] * lat[2][2]) / ra / rc
    cosc = (lat[0][0] * lat[1][0] + lat[0][1] * lat[1][1] +
            lat[0][2] * lat[1][2]) / rb / ra

    alpha = math.acos(cosa)
    beta = math.acos(cosb)
    gamma = math.acos(cosc)

    return np.array([ra, rb, rc, alpha, beta, gamma], float)


def lcons2lat(cons):
    (a, b, c, alpha, beta, gamma) = cons

    bc2 = b**2 + c**2 - 2 * b * c * math.cos(alpha)

    h1 = a
    h2 = b * math.cos(gamma)
    h3 = b * math.sin(gamma)
    h4 = c * math.cos(beta)
    h5 = ((h2 - h4)**2 + h3**2 + c**2 - h4**2 - bc2) / (2 * h3)
    h6 = math.sqrt(c**2 - h4**2 - h5**2)

    lattice = np.array([[h1, h2, h3], [h4, h5, h6]], float)
    return lattice


if __name__ == "__main__":
    e, f = get_ef('Li1.vasp')
    print ('energy:', e)
    print ('forces:')
    print (f)
