#!/bin/bash
# Begin LSF Directives
#BSUB -P CSC401
#BSUB -W 1:00
#BSUB -nnodes 2
#BSUB -alloc_flags gpumps
#BSUB -J RunSim123
#BSUB -o RunSim123.%J
#BSUB -e RunSim123.%J

jsrun -n 2 -c 18 -a 1 -g 1 ./pp_cuda_aware
