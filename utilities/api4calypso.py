#!/usr/bin/env python
import os
import subprocess
import numpy as np

def OPT_2_CONTCAR(caldir):
    try:
        subprocess.run(["cp", caldir + "/opt.vasp", caldir + "/CONTCAR"], check = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        subprocess.run(["bash", "traj2optvasp.sh"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        subprocess.run(["cp", caldir + "/opt.vasp", caldir + "/CONTCAR"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)

def write_OUTCAR(caldir):
    natom = os.popen('''grep "Number of atoms:" ''' + caldir + '''/log | awk '{print $4 }' ''').read()
    natom = int(natom)
    try:
        fe_atom = os.popen('''grep -A1 "Final energy per atom is" ''' + caldir + '''/log | tail -1 | awk '{print $1 }' ''').read()
        fe_atom = float(fe_atom)
    except:
        fe_atom = None
    energies = os.popen('''grep "FIRE:" ''' + caldir + '''/log | tail -1 | awk '{printf "%.6f", $4 }' ''').read()
    energies = float(energies)
    f_max = os.popen('''grep "FIRE:" ''' + caldir + '''/log | tail -1 | awk '{printf "%.4f", $5 }' ''').read()
    f_max = float(f_max)
    eV_o_A3_2_kB = 6.24150912E-4
    stress_tmp = os.popen('''grep scalar_pressure ''' + caldir + '''/*_test.py | tail -1 | awk -F ',' '{print $2 }' | awk -F ')' '{print $1 }' | awk '{print $3}' ''').read()
    stress_tmp = float(stress_tmp)
    stress_kB = float( stress_tmp * eV_o_A3_2_kB / 3.0 )
    
    if fe_atom is not None:
        energy = fe_atom
    else:
        energy = float( energies / natom)
    
    with open("OUTCAR", "w") as f:
        f.write("  enthalpy is  TOTEN    =      " + str(energy) + " eV" + "\n")
        f.write("  in kB" + ("      " + str(stress_kB))*3 + "      0.00000"*3 + "\n")
        f.write("  external pressure =        " + str(stress_kB) + " kB  " + "Pullay stress =        0.00 kB" + "\n")
        f.write("POSITION                                       TOTAL-FORCE (eV/Angst)"+ "\n")
        f.write("------------------------------------------------------------------------"+ "\n")
        for i in range(natom):
            f.write("      0.00000"*3 + ("     " + str(f_max))*3+ "\n")
        f.write("------------------------------------------------------------------------"+ "\n")

def write_OSZICAR(caldir):
    with open("OSZICAR", "w") as f:
        f.write("       N       E                     dE             d eps       ncg     rms              rms(c)"+ "\n")
        f.write("DAV:   1    -0.198615932504E+02   -0.32495E-08   -0.20110E-10  8888   0.171E-05    0"+ "\n")
        f.write("   1 F= -.15117390E+02 E0= -.15117100E+02  d E =-.151174E+02"+ "\n")
        
if __name__ == '__main__':
    caldir = './'
    OPT_2_CONTCAR(caldir)
    write_OUTCAR(caldir)
    write_OSZICAR(caldir)
