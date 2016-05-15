#!/bin/bash
#
# debug.sh - Run dnmf_exec on cluster.
#
# EXAMPLE:  qsub -l nodes=2:ppn=11 -v "N_PROC=21, N_ROWS=1000, N_COLS=1000, N_COMP=10, N_ITER=100, COMP_ERR=10" batch.sh
#
# Author: Olavur Mortensen <olavurmortensen@gmail.com>, 2016.
#
# #PBS -N dnmf_s103261
# Keep the log files local.
#PBS -k oe
#PBS -l walltime=1:00:00
# XeonE5-2680 processor with 20 cores.
#PBS -l feature=XeonE5-2680

# change into work directory
cd $PBS_O_WORKDIR

# load the right MPI support module
module load mpi/gcc

mpiexec -n $N_PROC ./dnmf_exec $N_ROWS $N_COLS $N_COMP $N_ITER $COMP_ERR

