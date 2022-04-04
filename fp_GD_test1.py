import numpy as np
import fplib_GD
import sys
# np.random.seed(42)
# Move function `readvasp(vp)` from test set to `fplib_FD.py`


#Calculate crystal atomic finger print force and steepest descent update
def test1_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 4.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 300
    atol = 1e-9
    step_size = 1e-3
    # const_factor = 1.0e+31
    rxyz_new = rxyz.copy()
    # fp_dist = 0.0
    # fpdist_error = 0.0
    # fpdist_temp_sum = 0.0
    # fpdsit_temp_num = 0.0
    
    for i_iter in range(iter_max):
        del_fp = np.zeros((len(rxyz_new), 3))
        sum_del_fp = np.zeros(3)
        fp_dist = 0.0
        for i_atom in range(len(rxyz_new)):
            # del_fp = np.zeros(3)
            # temp_del_fp = np.zeros(3)
            # accum_error = np.zeros(3)
            # temp_sum = np.zeros(3)
            fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
            D_fp_mat_iat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
            for j_atom in range(len(rxyz_new)):

                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)

                D_fp_mat_jat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                diff_fp = fp_iat-fp_jat
                # common_count, i_rxyz_sphere_1, i_rxyz_sphere_2 = \
                # fplib_GD.get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, \
                #                                 znucl, cutoff, i_atom, j_atom)
                iat_in_j_sphere, iat_j = fplib_GD.get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_new, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere:
                    diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :] - D_fp_mat_jat[iat_j, 0, :]
                    diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :] - D_fp_mat_jat[iat_j, 1, :]
                    diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :] - D_fp_mat_jat[iat_j, 2, :]
                else:
                    diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :]
                    diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :]
                    diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :]
                
                # Kahan sum implementation
                '''
                diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                temp_del_fp[0] = accum_error[0] + np.real( np.matmul( diff_fp.T,  diff_D_fp_x ) )
                temp_del_fp[1] = accum_error[1] + np.real( np.matmul( diff_fp.T,  diff_D_fp_y ) )
                temp_del_fp[2] = accum_error[2] + np.real( np.matmul( diff_fp.T,  diff_D_fp_z ) )
                temp_sum[0] = del_fp[0] + temp_del_fp[0]
                temp_sum[1] = del_fp[1] + temp_del_fp[1]
                temp_sum[2] = del_fp[2] + temp_del_fp[2]
                accum_error[0] = temp_del_fp[0] - (temp_sum[0] - del_fp[0])
                accum_error[1] = temp_del_fp[1] - (temp_sum[1] - del_fp[1])
                accum_error[2] = temp_del_fp[2] - (temp_sum[2] - del_fp[2])
                del_fp[0] = temp_sum[0]
                del_fp[1] = temp_sum[1]
                del_fp[2] = temp_sum[2]
                
                fpdist_temp_num = fpdist_error + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)
                fpdist_temp_sum = fp_dist + fpdist_temp_num
                fpdist_error = fpdist_temp_num - (fpdist_temp_sum - fp_dist)
                fp_dist = fpdist_temp_sum
                '''
                
                
                diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                del_fp[i_atom][0] = del_fp[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_x ) )
                del_fp[i_atom][1] = del_fp[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_y ) )
                del_fp[i_atom][2] = del_fp[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_z ) )
                fp_dist = fp_dist + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)
                
                
                # Debugging
                '''
                print ("diff_D_fp_x = \n{0:s}".\
                      format(np.array_str(diff_D_fp_x, precision=6, suppress_small=False)))
                print ("diff_D_fp_y = \n{0:s}".\
                      format(np.array_str(diff_D_fp_y, precision=6, suppress_small=False)))
                print ("diff_D_fp_z = \n{0:s}".\
                      format(np.array_str(diff_D_fp_x, precision=6, suppress_small=False), \
                             np.array_str(diff_D_fp_y, precision=6, suppress_small=False), \
                             np.array_str(diff_D_fp_z, precision=6, suppress_small=False)))
                
                print ( "diff_fp = \n{0:s}".\
                      format(np.array_str(diff_fp, precision=6, suppress_small=False)) )
                print ( "del_fp = [{0:.6e}, {1:.6e}, {2:.6e}]".\
                      format(del_fp[i_atom][0], del_fp[i_atom][1], del_fp[i_atom][2]) )
                '''
                
                
                
                
                
                
                # Previous iat in j_sphere routine
                '''
                for i_common in range(common_count):
                    diff_D_fp_x = D_fp_mat_iat[0, :, i_rxyz_sphere_1[i_common]] - \
                                  D_fp_mat_jat[0, :, i_rxyz_sphere_2[i_common]]
                    diff_D_fp_y = D_fp_mat_iat[1, :, i_rxyz_sphere_1[i_common]] - \
                                  D_fp_mat_jat[1, :, i_rxyz_sphere_2[i_common]]
                    diff_D_fp_z = D_fp_mat_iat[2, :, i_rxyz_sphere_1[i_common]] - \
                                  D_fp_mat_jat[2, :, i_rxyz_sphere_2[i_common]]
                    del_fp[0] = del_fp[0] + np.matmul( diff_fp.T,  diff_D_fp_x )
                    del_fp[1] = del_fp[1] + np.matmul( diff_fp.T,  diff_D_fp_y )
                    del_fp[2] = del_fp[2] + np.matmul( diff_fp.T,  diff_D_fp_z )
                # if iat_in_j_sphere:
                #     diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :] - D_fp_mat_jat[iat_j, 0, :]
                #     diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :] - D_fp_mat_jat[iat_j, 1, :]
                #     diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :] - D_fp_mat_jat[iat_j, 2, :]
                # else:
                #     diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :]
                #     diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :]
                #     diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :]
                # diff_D_fp_x = D_fp_mat[0, :, i_atom] - D_fp_mat[0, :, j_atom]
                # diff_D_fp_y = D_fp_mat[1, :, i_atom] - D_fp_mat[1, :, j_atom]
                # diff_D_fp_z = D_fp_mat[2, :, i_atom] - D_fp_mat[2, :, j_atom]
                # del_fp[0] = del_fp[0] + 2.0*np.matmul( diff_fp.T,  diff_D_fp_x )
                # del_fp[1] = del_fp[1] + 2.0*np.matmul( diff_fp.T,  diff_D_fp_y )
                # del_fp[2] = del_fp[2] + 2.0*np.matmul( diff_fp.T,  diff_D_fp_z )
                '''
                
                
                # print("del_fp = ", del_fp)
                # rxyz[i_atom] = rxyz[i_atom] - step_size*del_fp
                '''
                if max(del_fp) < atol:
                    print ("i_iter = {0:d} \nrxyz_final = \n{1:s}".\
                          format(i_iter+1, np.array_str(rxyz, precision=6, suppress_small=False)))
                    return
                    # with np.printoptions(precision=3, suppress=True):
                    # sys.exit("Reached user setting tolerance, program ended")
                else:
                    print ("i_iter = {0:d} \nrxyz = \n{1:s}".\
                          format(i_iter, np.array_str(rxyz, precision=6, suppress_small=False)))
                '''
            
            # rxyz_new[i_atom] = rxyz_new[i_atom] - step_size*del_fp/np.linalg.norm(del_fp)
            
        sum_del_fp = np.sum(del_fp, axis=0)
        for ii_atom in range(len(rxyz_new)):
            del_fp[ii_atom, :] = del_fp[ii_atom, :] - sum_del_fp/len(rxyz_new)
            rxyz_new[ii_atom] = rxyz_new[ii_atom] - \
                                step_size*del_fp[ii_atom, :]/np.linalg.norm(del_fp[ii_atom, :])
        print ( "i_iter = {0:d} \nrxyz_final = \n{1:s}".\
              format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
        print ( "Forces = \n{0:s}".\
              format(np.array_str(del_fp, precision=6, suppress_small=False)) )
        print ( "Finger print energy difference = {0:s}".\
              format(np.array_str(fp_dist, precision=6, suppress_small=False)) )
            
    
    
    '''
    for x in range(3):
        for iat in range(len(rxyz)):
            del_fp = np.zeros(3)
            for jat in range(len(rxyz)):
                # amp, n_sphere, rxyz_sphere, rcov_sphere = \
                # fplib_GD.get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
                # print ("amp", amp)
                # print ("n_sphere", n_sphere)
                # print ("rxyz_sphere", rxyz_sphere)
                # print ("rcov_sphere", rcov_sphere)
                # D_n_i = x*iat
                # D_n_j = x*jat
                fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, iat)
                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, jat)
                D_fp_mat_iat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff)
                D_fp_mat_jat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff)
                diff_fp = fp_iat-fp_jat
                diff_D_fp = D_fp_mat_iat[:, D_n_i] - D_fp_mat_jat[:, D_n_j]
                del_fp[x] = del_fp[x] + np.dot( diff_fp,  diff_D_fp )
                print ("force in x direction", x, -del_fp[x])
                if np.dot( diff_fp,  diff_D_fp ) < atol:
                    print("rxyz_final = \n", rxyz)
                    print("Reached user setting tolerance, program ended")
            # print ("1 rxyz", rxyz)
        # print ("2 rxyz", rxyz)
        rxyz[iat][x] = rxyz[iat][x] - step_size*del_fp[x]
        # print ("3 rxyz", rxyz)
    
    # print ("n_sphere", n_sphere)
    # print ("length of amp", len(amp))
    # print ("length of rcov_sphere", len(rcov_sphere))
    # print ("size of rxyz_sphere", rxyz_sphere.shape)
    
    return rxyz
    '''

# Calculate crystal atomic finger print energy
def test2_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    fp_dist = 0.0
    fpdist_error = 0.0
    temp_num = 0.0
    temp_sum = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        for i_atom in range(len(rxyz)):
            if types[i_atom] == itype:
                for j_atom in range(len(rxyz)):
                    if types[j_atom] == itype:
                        fp_iat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, i_atom)
                        fp_jat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, j_atom)
                        temp_num = fpdist_error + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)
                        temp_sum = fp_dist + temp_num
                        accum_error = temp_num - (temp_sum - fp_dist)
                        fp_dist = temp_sum

    
    print ( "Finger print energy = {0:s}".format(np.array_str(fp_dist, \
                                               precision=6, suppress_small=False)) )
    # return fp_dist


# Compare Simpson's numerical integration with Fingerprint energy difference
def test3_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 4
    atol = 1.0e-6
    step_size = 1e-4
    const_factor = 1.0e+31
    del_fp_dist = 0.0
    rxyz_left = rxyz.copy()
    rxyz_new = rxyz.copy()
    rxyz_right = rxyz.copy()
    rxyz_delta = np.zeros_like(rxyz)
    for i_iter in range(iter_max):
        del_fp_left = np.zeros((len(rxyz_new), 3))
        del_fp = np.zeros((len(rxyz_new), 3))
        del_fp_right = np.zeros((len(rxyz_new), 3))
        sum_del_fp_left = np.zeros(3)
        sum_del_fp = np.zeros(3)
        sum_del_fp_right = np.zeros(3)
        fp_dist_0 = 0.0
        fp_dist_new = 0.0
        fp_dist_del = 0.0
        rxyz_delta = step_size*fplib_GD.get_rxyz_delta(rxyz)
        rxyz_left = rxyz_new.copy()
        rxyz_new = np.add(rxyz_new, rxyz_delta)
        rxyz_right = np.add(rxyz_new, rxyz_delta)
        for i_atom in range(len(rxyz)):
            # del_fp = np.zeros(3)
            for j_atom in range(len(rxyz)):
                fp_iat_0 = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, i_atom)
                fp_jat_0 = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, j_atom)
                fp_iat_left = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, i_atom)
                fp_jat_left = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, j_atom)
                fp_iat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                fp_jat = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                fp_iat_right = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, i_atom)
                fp_jat_right = \
                fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, j_atom)
                D_fp_mat_iat_left = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, i_atom)
                D_fp_mat_jat_left = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_left, types, znucl, cutoff, j_atom)
                D_fp_mat_iat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, i_atom)
                D_fp_mat_jat = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, j_atom)
                D_fp_mat_iat_right = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, i_atom)
                D_fp_mat_jat_right = \
                fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                          rxyz_right, types, znucl, cutoff, j_atom)
                diff_fp_left = fp_iat_left - fp_jat_left
                diff_fp = fp_iat - fp_jat
                diff_fp_right = fp_iat_right - fp_jat_right
                # common_count, i_rxyz_sphere_1, i_rxyz_sphere_2 = \
                # fplib_GD.get_common_sphere(ntyp, nx, lmax, lat, rxyz, types, \
                #                                 znucl, cutoff, i_atom, j_atom)
                iat_in_j_sphere_left, iat_j_left = fplib_GD.get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_left, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere_left:
                    diff_D_fp_x_left = D_fp_mat_iat_left[i_atom, 0, :] - \
                                       D_fp_mat_jat_left[iat_j_left, 0, :]
                    diff_D_fp_y_left = D_fp_mat_iat_left[i_atom, 1, :] - \
                                       D_fp_mat_jat_left[iat_j_left, 1, :]
                    diff_D_fp_z_left = D_fp_mat_iat_left[i_atom, 2, :] - \
                                       D_fp_mat_jat_left[iat_j_left, 2, :]
                else:
                    diff_D_fp_x_left = D_fp_mat_iat_left[i_atom, 0, :]
                    diff_D_fp_y_left = D_fp_mat_iat_left[i_atom, 1, :]
                    diff_D_fp_z_left = D_fp_mat_iat_left[i_atom, 2, :]
                
                iat_in_j_sphere, iat_j = fplib_GD.get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_new, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere:
                    diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :] - D_fp_mat_jat[iat_j, 0, :]
                    diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :] - D_fp_mat_jat[iat_j, 1, :]
                    diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :] - D_fp_mat_jat[iat_j, 2, :]
                else:
                    diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :]
                    diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :]
                    diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :]
                
                iat_in_j_sphere_right, iat_j_right = fplib_GD.get_common_sphere(ntyp, \
                              nx, lmax, lat, rxyz_right, types, znucl, cutoff, i_atom, j_atom)
                if iat_in_j_sphere_right:
                    diff_D_fp_x_right = D_fp_mat_iat_right[i_atom, 0, :] - \
                                        D_fp_mat_jat_right[iat_j_right, 0, :]
                    diff_D_fp_y_right = D_fp_mat_iat_right[i_atom, 1, :] - \
                                        D_fp_mat_jat_right[iat_j_right, 1, :]
                    diff_D_fp_z_right = D_fp_mat_iat_right[i_atom, 2, :] - \
                                        D_fp_mat_jat_right[iat_j_right, 2, :]
                else:
                    diff_D_fp_x_right = D_fp_mat_iat_right[i_atom, 0, :]
                    diff_D_fp_y_right = D_fp_mat_iat_right[i_atom, 1, :]
                    diff_D_fp_z_right = D_fp_mat_iat_right[i_atom, 2, :]
                
                diff_D_fp_x_left = np.vstack( (np.array(diff_D_fp_x_left)[::-1], ) ).T
                diff_D_fp_y_left = np.vstack( (np.array(diff_D_fp_y_left)[::-1], ) ).T
                diff_D_fp_z_left = np.vstack( (np.array(diff_D_fp_z_left)[::-1], ) ).T
                diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                diff_D_fp_x_right = np.vstack( (np.array(diff_D_fp_x_right)[::-1], ) ).T
                diff_D_fp_y_right = np.vstack( (np.array(diff_D_fp_y_right)[::-1], ) ).T
                diff_D_fp_z_right = np.vstack( (np.array(diff_D_fp_z_right)[::-1], ) ).T
                
                del_fp_left[i_atom][0] = del_fp_left[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp_left.T, diff_D_fp_x_left ) )
                del_fp_left[i_atom][1] = del_fp_left[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp_left.T, diff_D_fp_y_left ) )
                del_fp_left[i_atom][2] = del_fp_left[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp_left.T, diff_D_fp_z_left ) )
                del_fp[i_atom][0] = del_fp[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_x ) )
                del_fp[i_atom][1] = del_fp[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_y ) )
                del_fp[i_atom][2] = del_fp[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_z ) )
                del_fp_right[i_atom][0] = del_fp_right[i_atom][0] + \
                                    2.0*np.real( np.matmul( diff_fp_right.T, diff_D_fp_x_right ) )
                del_fp_right[i_atom][1] = del_fp_right[i_atom][1] + \
                                    2.0*np.real( np.matmul( diff_fp_right.T, diff_D_fp_y_right ) )
                del_fp_right[i_atom][2] = del_fp_right[i_atom][2] + \
                                    2.0*np.real( np.matmul( diff_fp_right.T, diff_D_fp_z_right ) )
                
                fp_dist_0 = fp_dist_0 + fplib_GD.get_fpdist(ntyp, types, fp_iat_0, fp_jat_0)
                fp_dist_new = fp_dist_new + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)
                fp_dist_del = fp_dist_new - fp_dist_0
                del_fp_dist = del_fp_dist + np.vdot(rxyz_delta[i_atom], del_fp[i_atom])
                
                '''
                print ("diff_D_fp_x = \n{0:s}".\
                      format(np.array_str(diff_D_fp_x, precision=6, suppress_small=False)))
                print ("diff_D_fp_y = \n{0:s}".\
                      format(np.array_str(diff_D_fp_y, precision=6, suppress_small=False)))
                print ("diff_D_fp_z = \n{0:s}".\
                      format(np.array_str(diff_D_fp_x, precision=6, suppress_small=False), \
                             np.array_str(diff_D_fp_y, precision=6, suppress_small=False), \
                             np.array_str(diff_D_fp_z, precision=6, suppress_small=False)))
                
                print ( "diff_fp = \n{0:s}".\
                      format(np.array_str(diff_fp, precision=6, suppress_small=False)) )
                print ( "del_fp = [{0:.6e}, {1:.6e}, {2:.6e}]".\
                      format(del_fp[i_atom][0], del_fp[i_atom][1], del_fp[i_atom][2]) )
                '''
                
            # del_fp_dist = del_fp_dist + np.vdot(rxyz_delta[i_atom], del_fp[i_atom])
        
        sum_del_fp_left = np.sum(del_fp_left, axis=0)
        sum_del_fp = np.sum(del_fp, axis=0)
        sum_del_fp_right = np.sum(del_fp_right, axis=0)
        
        for ii_atom in range(len(rxyz_new)):
            del_fp_left[ii_atom, :] = del_fp_left[ii_atom, :] - sum_del_fp_left/len(rxyz_new)
            del_fp[ii_atom, :] = del_fp[ii_atom, :] - sum_del_fp/len(rxyz_new)
            del_fp_right[ii_atom, :] = del_fp_right[ii_atom, :] - sum_del_fp_right/len(rxyz_new)
            del_fp_dist = del_fp_dist + \
                          ( np.absolute( np.dot(rxyz_delta[ii_atom], del_fp_left[ii_atom]) ) + \
                        4.0*np.absolute( np.dot(rxyz_delta[ii_atom], del_fp[ii_atom]) ) + \
                            np.absolute( np.dot(rxyz_delta[ii_atom], del_fp_right[ii_atom]) ) )/3.0
        
        print ( "i_iter = {0:d} \nrxyz_new = \n{1:s}".\
              format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
        print ( "Numerical integral = {0:.6e}".format(del_fp_dist) )
        print ( "Forces = \n{0:s}".\
              format(np.array_str(del_fp, precision=6, suppress_small=False)) )
        # print ( "Finger print energy difference = {0:s}".\
        #       format(np.array_str(fp_dist_del, precision=6, suppress_small=False)) )
    
    '''
    fp_dist_0 = 0.0
    fp_dist_new = 0.0
    fp_dist_del = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        for iat in range(len(rxyz)):
            if types[iat] == itype:
                for jat in range(len(rxyz)):
                    if types[jat] == itype:
                        fp_iat_0 = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, iat)
                        fp_jat_0 = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz, types, znucl, cutoff, jat)
                        fp_iat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, iat)
                        fp_jat = \
                        fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                          rxyz_new, types, znucl, cutoff, jat)
                        fp_dist_0 = fp_dist_0 + fplib_GD.get_fpdist(ntyp, types, fp_iat_0, fp_jat_0)
                        fp_dist_new = fp_dist_new + fplib_GD.get_fpdist(ntyp, types, fp_iat, fp_jat)
    
    fp_dist_del = fp_dist_new - fp_dist_0
    print ( "Finger print energy difference = {0:s}".\
          format(np.array_str(fp_dist_del, precision=6, suppress_small=False)) )
    '''
    
    
    # return fp_dist
    # return rxyz
    
    
    
    
# Compare del_fp with finite difference
def test4_CG(v1):
    ntyp = 1
    nx = 300
    lmax = 0
    cutoff = 6.5
    znucl = np.array([3], int)
    lat, rxyz, types = fplib_GD.readvasp(v1)
    contract = False
    i_iter = 0
    iter_max = 1
    atol = 1.0e-6
    step_size = 1e-4
    const_factor = 1.0e+31
    del_fp_dist = 0.0
    rxyz_left = rxyz.copy()
    rxyz_new = rxyz.copy()
    rxyz_right = rxyz.copy()
    rxyz_delta = np.zeros_like(rxyz)
    for i_iter in range(iter_max):
        del_fp = np.zeros((len(rxyz_new), 3))
        finite_diff = np.zeros((len(rxyz_new), 3))
        sum_del_fp = np.zeros(3)
        fp_dist_0 = 0.0
        fp_dist_new = 0.0
        fp_dist_del = 0.0
        rxyz_delta = step_size*fplib_GD.get_rxyz_delta(rxyz)
        rxyz_new = np.add(rxyz_new, rxyz_delta)
        for i_atom in range(len(rxyz)):
            for j_atom in range(len(rxyz)):
                for k in range(3):
                    h = rxyz_delta[i_atom][k]
                    # rxyz_left[i_atom][k] = rxyz_left[i_atom][k] - 2.0*h
                    rxyz_right[i_atom][k] = rxyz_right[i_atom][k] + 2.0*h
                    
                    fp_iat_left = \
                    fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_left, types, znucl, cutoff, i_atom)
                    fp_jat_left = \
                    fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_left, types, znucl, cutoff, j_atom)
                    fp_iat = \
                    fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_new, types, znucl, cutoff, i_atom)
                    fp_jat = \
                    fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_new, types, znucl, cutoff, j_atom)
                    fp_iat_right = \
                    fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_right, types, znucl, cutoff, i_atom)
                    fp_jat_right = \
                    fplib_GD.get_fp(contract, ntyp, nx, lmax, lat, \
                                              rxyz_right, types, znucl, cutoff, j_atom)
                    
                    fp_dist_left = fplib_GD.get_fpdist(ntyp, types, fp_iat_left, fp_jat_left)
                    fp_dist_right = fplib_GD.get_fpdist(ntyp, types, fp_iat_right, fp_jat_right)
                    finite_diff[i_atom][k] = (fp_dist_right - fp_dist_left)/(2.0*h)
                    
                    D_fp_mat_iat = \
                    fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                              rxyz_new, types, znucl, cutoff, i_atom)
                    D_fp_mat_jat = \
                    fplib_GD.get_D_fp_mat(contract, ntyp, nx, lmax, lat, \
                                              rxyz_new, types, znucl, cutoff, j_atom)
                    
                    diff_fp = fp_iat - fp_jat
                    
                    iat_in_j_sphere, iat_j = fplib_GD.get_common_sphere(ntyp, \
                                  nx, lmax, lat, rxyz_new, types, znucl, cutoff, i_atom, j_atom)
                    if iat_in_j_sphere:
                        diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :] - D_fp_mat_jat[iat_j, 0, :]
                        diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :] - D_fp_mat_jat[iat_j, 1, :]
                        diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :] - D_fp_mat_jat[iat_j, 2, :]
                    else:
                        diff_D_fp_x = D_fp_mat_iat[i_atom, 0, :]
                        diff_D_fp_y = D_fp_mat_iat[i_atom, 1, :]
                        diff_D_fp_z = D_fp_mat_iat[i_atom, 2, :]
                        
                    diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                    diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                    diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                
                    del_fp[i_atom][0] = del_fp[i_atom][0] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_x ) )
                    del_fp[i_atom][1] = del_fp[i_atom][1] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_y ) )
                    del_fp[i_atom][2] = del_fp[i_atom][2] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_z ) )
        
        
        
        
        
        sum_del_fp = np.sum(del_fp, axis=0)
        sum_finite_diff = np.sum(finite_diff, axis=0)
        for ii_atom in range(len(rxyz_new)):
            del_fp[ii_atom, :] = del_fp[ii_atom, :] - sum_del_fp/len(rxyz_new)
            finite_diff[ii_atom, :] = finite_diff[ii_atom, :] - sum_finite_diff/len(rxyz)
        
        print ( "i_iter = {0:d} \nrxyz_new = \n{1:s}".\
              format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
        print ( "Forces = \n{0:s}".\
              format(np.array_str(del_fp, precision=6, suppress_small=False)) )
        print ( "Finite difference = \n{0:s}".\
              format(np.array_str(finite_diff, precision=6, suppress_small=False)) )
    
    
    # return fp_dist
    # return rxyz

    

if __name__ == "__main__":
    args = sys.argv
    v1 = args[1]
    test1_CG(v1)
    # test2_CG(v1)
    # test3_CG(v1)
    # test4_CG(v1)
