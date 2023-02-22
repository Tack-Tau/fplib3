#!/usr/bin/env python
import os
import numpy as np

def OPT_2_CONTCAR(caldir):
    try:
        os.system('cp ' + str(caldir) + '/opt.vasp ' + str(caldir) + '/CONTCAR ')
    except:
        os.system('bash traj2optvasp.sh')
        os.system('cp ' + str(caldir) + '/opt.vasp ' + str(caldir) + '/CONTCAR ')

def write_OUTCAR(caldir):
    natom = os.popen('''grep "Number of atoms:" ''' + caldir + '''/slurm-* | awk '{print $4 }' ''')
    fe_atom = os.popen('''grep -A1 "Final energy per atom is" ''' + caldir + '''/slurm-* | tail -1 | awk '{print $1 }' ''')
    energies = os.popen('''grep "FIRE:" ''' + caldir + '''/slurm-* | tail -1 | awk '{printf "%.6f", $4 }' ''')
    f_max = os.popen('''grep "fmax" ''' + caldir + '''/slurm-* | tail -1 | awk '{printf "%.4f", $5 }' ''')
    eV_o_A3_2_kB = 6.24150912E-4
    stress_tmp = os.popen('''grep scalar_pressure ''' + caldir + '''/*_test.py | awk -F ',' '{print $2 }' | awk -F ')' '{print $1 }' | awk '{print $3}' ''')
    stress_kB = float( stress_tmp * eV_o_A3_2_kB / 3.0 )
    
    if fe_atom is not None:
        energy = fe_atom
    else:
        energy = float(energies / natom)
    
    with open("OUTCAR", "w") as f:
        f.write("  enthalpy is  TOTEN    =      " + str(energy) + " eV")
        f.write("  in kB" + ("      " + str(stress_kB))*3 + "      0.00000"*3)
        f.write("  external pressure =        " + str(stress_kB) + " kB  " + "Pullay stress =        0.00 kB" )
        f.write("POSITION                                       TOTAL-FORCE (eV/Angst)")
        f.write("---------------------------------------------------------------------")
        f.write("      0.00000"*3 + ("     " + str(f_max))*3)
        f.write("---------------------------------------------------------------------")

def write_OSZICAR(caldir):
    with open("OSZICAR", "w") as f:
        f.write("       N       E                     dE             d eps       ncg     rms              rms(c)")
        f.write("DAV:   1    -0.198615932504E+02   -0.32495E-08   -0.20110E-10  8888   0.171E-05    0")
        f.write("   1 F= -.15117390E+02 E0= -.15117100E+02  d E =-.151174E+02")
        
if __name__ == '__main__':
    caldir = './'
    OPT_2_CONTCAR(caldir)
    write_OUTCAR(caldir)
    write_OSZICAR(caldir)