import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from scipy.optimize import linear_sum_assignment
import rcovdata


# import numba


# @numba.jit()
def get_gom(int lseg, float[:,:] rxyz, float[:] rcov, float[:] amp):
    # s orbital only lseg == 1
    cdef int nat
    cdef int iat, jat, i
    cdef double d2, r, stv, sji
    cdef double[:] d
    cdef double[:,:] om
    nat = len(rxyz)    
    if lseg == 1:
        om = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                for i in range(3):
                    d[i] = rxyz[iat][i] - rxyz[jat][i]
                # d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[iat][jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1.0*d2*r) * amp[iat] * amp[jat]
    else:
        # for both s and p orbitals
        om = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                for i in range(3):
                    d[i] = rxyz[iat][i] - rxyz[jat][i]
                # d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                om[4*iat][4*jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                    * np.exp(-1*d2*r) * amp[iat] * amp[jat]
                
                # <s_i | p_j>
                sji = np.sqrt(4.0*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                stv = np.sqrt(8.0) * rcov[jat] * r * sji
                om[4*iat][4*jat+1] = stv * d[0] * amp[iat] * amp[jat]
                om[4*iat][4*jat+2] = stv * d[1] * amp[iat] * amp[jat]
                om[4*iat][4*jat+3] = stv * d[2] * amp[iat] * amp[jat]

                # <p_i | s_j> 
                stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                om[4*iat+1][4*jat] = stv * d[0] * amp[iat] * amp[jat]
                om[4*iat+2][4*jat] = stv * d[1] * amp[iat] * amp[jat]
                om[4*iat+3][4*jat] = stv * d[2] * amp[iat] * amp[jat]

                # <p_i | p_j>
                stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                om[4*iat+1][4*jat+1] = stv * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat]
                om[4*iat+1][4*jat+2] = stv * (d[1] * d[0]        ) * amp[iat] * amp[jat]
                om[4*iat+1][4*jat+3] = stv * (d[2] * d[0]        ) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+1] = stv * (d[0] * d[1]        ) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+2] = stv * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat]
                om[4*iat+2][4*jat+3] = stv * (d[2] * d[1]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+1] = stv * (d[0] * d[2]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+2] = stv * (d[1] * d[2]        ) * amp[iat] * amp[jat]
                om[4*iat+3][4*jat+3] = stv * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat]
    
    # for i in range(len(om)):
    #     for j in range(len(om)):
    #         if abs(om[i][j] - om[j][i]) > 1e-6:
    #             print ("ERROR", i, j, om[i][j], om[j][i])
    return om


# @numba.jit()
def get_fp_nonperiodic(float[:] rxyz, int[:] znucls):
    cdef double[:] rcov, amp, fp
    cdef int x, i, l
    cdef double[:,:] gom
    cdef double[:] amp
    l = rxyz.shape[0]
    # amp = [1.0] * len(rxyz)
    for i in range(l):
        rxyz[i] = 1.0
    for x in znucls:
        rcov.append(rcovdata.rcovdata[x][2])
    gom = get_gom(1, rxyz, rcov, amp)
    fp = np.linalg.eigvals(gom)
    fp = sorted(fp)
    fp = np.array(fp, float)
    return fp

# @numba.jit()
def get_fpdist_nonperiodic(float[:] fp1, float[:] fp2):
    cdef double[:] d
    cdef int i, n
    # d = fp1 - fp2
    n = fp1.shape[0]
    for i in range(n):
        d[i] = fp1[i] - fp2[i]
    return np.sqrt(np.vdot(d, d))

# @numba.jit()
def get_fp(bool contract, int ntyp, int nx, int lmax, float[:,:] lat, float[:,:] rxyz, int[:] types, int[:] znucl, float cutoff):
    cdef int lseg, iat, jat, nat, ix, iy, iz, il, ixyz, l, NC, ityp_sphere, n_sphere
    cdef int[:] ind, n_sphere_list
    cdef double wc, fc, cutoff2, xi, yi, zi
    cdef double[:] amp, rcov, lfp, fp0, sfp, sfp0, rxyz_sphere, rcov_sphere, val
    cdef int nid, nids, i
    cdef double[:,:] vec, pvec, gom, omx
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2
    ixyz = get_ixyz(lat, cutoff)
    NC = 3
    wc = cutoff / np.sqrt(2.* NC)
    fc = 1.0 / (2.0 * NC * wc**2)
    nat = len(rxyz)
    cutoff2 = cutoff**2 
    
    # n_sphere_list = [] 
    # lfp = []
    # sfp = []
    for iat in range(nat):
        rxyz_sphere = []
        rcov_sphere = []
        ind = [0] * (lseg * nx)
        amp = []
        xi, yi, zi = rxyz[iat]
        n_sphere = 0
        for jat in range(nat):
            for ix in range(-ixyz, ixyz+1):
                for iy in range(-ixyz, ixyz+1):
                    for iz in range(-ixyz, ixyz+1):
                        xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                        yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                        zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                        d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > nx:
                                print ("FP WARNING: the cutoff is too large.")
                            amp.append((1.0-d2*fc)**NC)
                            # print (1.0-d2*fc)**NC
                            rxyz_sphere.append([xj, yj, zj])
                            rcov_sphere.append(rcovdata.rcovdata[znucl[types[jat]-1]][2]) 
                            if jat == iat and ix == 0 and iy == 0 and iz == 0:
                                ityp_sphere = 0
                            else:
                                ityp_sphere = types[jat]
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] == ityp_sphere * l + 1
        n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere, float)
        # full overlap matrix
        # cdef int nid, nids, i
        # cdef float *val, *fp0, *lfp, *sfp0, *sfp
        # cdef float **vec, **pvec, **gom, **omx
        nid = lseg * n_sphere
        gom = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
        val, vec = np.linalg.eig(gom)
        val = np.real(val)
        fp0 = np.zeros(nx*lseg)
        for i in range(len(val)):
            fp0[i] = val[i]
        lfp.append(sorted(fp0))
        pvec = np.real(np.transpose(vec)[0])
        # contracted overlap matrix
        if contract:
            nids = l * (ntyp + 1)
            omx = np.zeros((nids, nids))
            for i in range(nid):
                for j in range(nid):
                    # print ind[i], ind[j]
                    omx[ind[i]][ind[j]] = omx[ind[i]][ind[j]] + pvec[i] * gom[i][j] * pvec[j]
            # for i in range(nids):
            #     for j in range(nids):
            #         if abs(omx[i][j] - omx[j][i]) > 1e-6:
            #             print ("ERROR", i, j, omx[i][j], omx[j][i])
            # print omx
            sfp0 = np.linalg.eigvals(omx)
            sfp.append(sorted(sfp0))


    print ("n_sphere_min", min(n_sphere_list))
    print ("n_shpere_max", max(n_sphere_list)) 

    if contract:
        sfp = np.array(sfp, float)
        return sfp
    else:
        lfp = np.array(lfp, float)
        return lfp

# @numba.jit()
def get_ixyz(float[:,:] lat, float cutoff):
    cdef int ixyz
    cdef double[:] vec
    cdef double[:,:] lat2
    lat2 = np.matmul(lat, np.transpose(lat))
    # print lat2
    vec = np.linalg.eigvals(lat2)
    # print (vec)
    ixyz = int(np.sqrt(1.0/max(vec))*cutoff) + 1
    return ixyz

# @numba.jit()
def get_fpdist(int ntyp, int[:] types, float[:] fp1, float[:] fp2):
    cdef int iat, jat, nat, lenfp, ityp, itype
    cdef double total, fpd
    cdef double[:] tfpd, row_ind, col_ind
    cdef double[:,:] MX
    nat, lenfp = np.shape(fp1)
    fpd = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        MX = np.zeros((nat, nat))
        for iat in range(nat):
            if types[iat] == itype:
                for jat in range(nat):
                    if types[jat] == itype:
                        tfpd = fp1[iat] - fp2[jat]
                        MX[iat][jat] = np.sqrt(np.vdot(tfpd, tfpd)/lenfp)

        row_ind, col_ind = linear_sum_assignment(MX)
        # print(row_ind, col_ind)
        total = MX[row_ind, col_ind].sum()
        fpd += total

    fpd = fpd / nat
    return fpd

