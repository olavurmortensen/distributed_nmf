#!/bin/bash
#
# debug.sh - Run dnmf_exec on cluster.
#
# EXAMPLE:  qsub [options] batch.sh
#
# Author: Olavur Mortensen <olavurmortensen@gmail.com>, 2016.
#
# name of the job
# #PBS -N dnmf_s103261
# we need to forward the DISPLAY, and the name of our program to debug
#PBS -v DISPLAY,PROG,COMP
# keep the log files local
#PBS -k oe
#PBS -l walltime=1:00:00
#PBS -l nodes=2:ppn=20
#PBS -l feature=XeonE5-2680

# change into work directory
cd $PBS_O_WORKDIR


# load the right MPI support module
module load mpi/gcc

mpiexec -n 40 ./dnmf_exec 1000 1000 100 100 10

