#!/bin/bash
## script to run small test MPI jobs in CSD3 login node

module load rhel7/default-peta4
module load intel/bundles/complib/2019.3

## load elpa module
module load elpa/2019.05.001-intel-17.0.4

## run 6 MPI processes (-np 6), with lowest priority (nice -19)
mpirun -np 6 -l nice -19 diag__matrix.x mat_input > mat_output
