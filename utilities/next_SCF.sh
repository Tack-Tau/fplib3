#!/bin/bash
input="./redo_SCF_dir" ; while IFS= read -r line ; do cd fVfV/$line ; rm slurm-* opt.* KPOINTS POSCAR CONTCAR OUTCAR WAVECAR ; cd ../.. ; done < "$input" 2> /dev/null
input="./redo_SCF_dir" ; while IFS= read -r line ; do cp fp/$line/opt.vasp fVfV/$line/POSCAR ; done < "$input"
input="./redo_SCF_dir" ; while IFS= read -r line ; do cd fVfV/$line ; sbatch ase_fp_VASP_job.sh ; cd ../.. ; done < "$input"
