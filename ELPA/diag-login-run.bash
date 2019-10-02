#!/bin/bash
module load rhel7/default-peta4
module load intel/bundles/complib/2019.3
module load elpa/2019.05.001-intel-17.0.4
mpirun -np 4 -l nice -19 diag__elpa.x > diag-elpa.out
