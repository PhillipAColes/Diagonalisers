#### makefile for DiRAC@Cambridge cluster CSD3 (CPUs) or Wilkes2 (Gpus) ####

PLAT   = __matrix
#FOR    = mpif90
FOR    = mpiifort


#### for CPUs (CSD3) ####
prefix=/usr/local/software/elpa/2019.05.001-intel-17.0.4

## for GPUs (Wilkes2) ####
#prefix=/usr/local/software/elpa/2019.05.001-intel-17.0.4-cuda


exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include


#### for CPUs (CSD3) ####
#CFLAGS = -I${includedir}/elpa_openmp-2019.05.001 -fopenmp
FCFLAGS = -O3 -ipo -xHost -traceback -g -I${includedir}/elpa_openmp-2019.05.001/modules
LIB =  -L${libdir} -lelpa_openmp  -L/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm


#### for GPUs (Wilkes2) ####
##CFLAGS= -I${includedir}/elpa-2019.05.001 -qopenmp
#FCFLAGS= -I${includedir}/elpa-2019.05.001/modules
#LIB= -L${libdir} -lelpa -lcudart -lcublas -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm 


default:   diag$(PLAT).x

diag$(PLAT).x:  $(OBJ) diag.o
	$(FOR) $(FCFLAGS) -o diag$(PLAT).x diag.o $(LIB)

diag.o:  diag.f90 $(OBJ)
	$(FOR) -c diag.f90 $(FCFLAGS)

clean:
	rm $(OBJ) *.mod diag.o *.x