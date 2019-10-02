!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! This program sets up blacs environment and calls the ELPA eigensolver to solve
! the eigenvector and eigenvalue problem for a dense double-real symmetric matrix.
! There are two algorithms implemented by ELPA: 1stage and 2stage. The latter is
! faster on CPUs and the former is faster using GPUS.
! For GPU version use the command  'call e%set("gpu",1,info)'. To use CPUs use
! 'call e%set("gpu",0,info)'. CPUs are default so this can also be commented out.
! In order to use the 2stage ELPA solver on GPUs the kernel must be set by
! 'call e%set("real_kernel",elpa_2stage_real_gpu,info)'. Check to see what kernels
! have been built in by running the 'elpa2_print_kernels_openmp' executable
! 
! KNOWN BUGS: 
! 1) currently elpa1 yields crazy eigenvalues when >20 nodes are used
!    for a matrix of dimension greater than 200,000. If the number of
!    nodes is REDUCED then elpa1 works fine. elpa2 always seems to work.
! 2) elpa1 yields crazy eigenvalues IF the matrix has off-diagonal elements.
!    If the matrix is already diagonal then the diagonaliser works fine (!?)
! 3) not really a bug... but currently the elpa2 GPU kernel has not been built
!    so elpa2 cannot be used on GPUs.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program diag_elpa

use iso_c_binding
use elpa
use mpi

implicit none

!relating to MPI and process grid
integer:: mpierr,mypnum,nprocs,nprow,npcol,myprow,mypcol

!relating to the global matrix
integer:: dimen_s !dimension of A, the matrix to be diagonalised
integer:: neigenvals
double precision, allocatable:: eigenvals(:)

!relating to distributed matrix
integer:: loc_r,loc_c,llda !row and column dimensions of local matrix
integer:: desca(9),descz(9)
double precision, allocatable:: a(:,:),z(:,:)!local matrices, a = input, z = eigenvectors
integer::i_loc,j_loc !row and column index of element in local array
integer::rsrc,csrc !grid coordinates of the process that owns the matrix element with index i,j in the global matrix
integer::nb !block size

!relating to elpa
integer:: elpa_version
integer:: ctxt
class(elpa_t), pointer::e
character(200)::errormsg

!relating to blacs
integer, external:: numroc

!general
integer::info,i,j
integer::iseed(1000000)
double precision::t1,t2

!Initialise MPI communicator
   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world,mypnum,mpierr)
   call mpi_comm_size(mpi_comm_world,nprocs,mpierr)

   if(mpierr/=0)then
   print*,'mpierr returned nonzero value... stopping'
   stop
   end if

!Initialise ELPA, most recent elpa version = 20190524
   if (elpa_init(20190524)/= elpa_ok)then
    print*,"ELPA API not supported"
    stop
   end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! here we set up the process grid and stuff !!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   do npcol = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,npcol) == 0 ) exit
   enddo

   nprow = nprocs/npcol

   if(mypnum==0)print '(3(a,i0))','Number of processor rows=',nprow,', cols=',npcol,', total=',nprocs

   ctxt = mpi_comm_world
   call blacs_gridinit(ctxt,'r',nprow,npcol)
   call blacs_gridinfo(ctxt,nprow,npcol,myprow,mypcol)

   write(*,"(/'mpi communicator: ', i10,', number of processes: ', i5,', process number: ',i5,',at coordinate (nprow,npcol) = (',i4,',',i4,')')")mpi_comm_world,nprocs,mypnum,myprow,mypcol
   
   dimen_s = 64 !global matrix dimension
   nb = 32 !block number
   neigenvals = dimen_s !number of eigenvalues we want to compute

   loc_r = numroc(dimen_s,nb,myprow,0,nprow)!essentially number of rows of submatrix (i.e. number of rows of local array owned by process)
   loc_c = numroc(dimen_s,nb,mypcol,0,npcol)!essentially number of columns of submatrix
   llda = max(1,loc_r)!local leading dimension
   !
   !initialise the array descriptor of the distributed matrix, presumably ctxt provides info of the grid setup
   call descinit(desca,dimen_s,dimen_s,nb,nb,0,0,ctxt,llda,info)!info = 0 for successful exit
   call descinit(descz,dimen_s,dimen_s,nb,nb,0,0,ctxt,llda,info)!info = 0 for successful exit

   call blacs_barrier(ctxt,'A')
   
   !allocate the distributed matrix (a) and matrix of eigenvectors (z)
   allocate(a(loc_r,loc_c),z(loc_r,loc_c))
   !allocate array of eigenvalues
   allocate(eigenvals(dimen_s))

   call mpi_barrier(mpi_comm_world,mpierr)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! Begin generating matrix !!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   t1 = MPI_Wtime()

   if (mypnum == 0)write(*,"(/'Begin generating matrix')")

! double real pseudo-random symmetric matrix

!   do i = 1, dimen_s
!     iseed(i) = 123456789-i-1;
!     iseed(i) = mod((1103515245 * iseed(i) + 12345),1099511627776);
!   enddo

!   do j = 1,dimen_s
!     do i = 1,dimen_s
!       call infog2l(i,j,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc
!       if (myprow == rsrc .and. mypcol == csrc) then
!         if(i>=j) a(i_loc,j_loc) = dble(iseed(i))/dble(1099511627776.0)
!         if(i < j) a(i_loc,j_loc) = dble(iseed(j))/dble(1099511627776.0)
!         if(i==j) a(i_loc,j_loc) = a(i_loc,j_loc) + dble(10.0) + dble(j)
!       endif
!     enddo
!   enddo

! small, simple, double-real positive symmetric test matrix

   do i = 1,dimen_s
     do j = i,dimen_s

       call infog2l(i,j,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc

       if (myprow == rsrc .and. mypcol == csrc) then
          a(i_loc,j_loc) = 0.00001d0 * dble(j)
       endif
       
       call infog2l(j,i,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc
       
       if (myprow == rsrc .and. mypcol == csrc) then
          a(i_loc,j_loc) = 0.00001d0 * dble(j)
       endif
       
       call infog2l(i,j,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc
       
       if (myprow == rsrc .and. mypcol == csrc) then
         if(i==j) a(i_loc,j_loc) = a(i_loc,j_loc) + dble(10.0) + dble(j)
       endif
       
     enddo
   enddo

   call mpi_barrier(mpi_comm_world,mpierr)

! print the simple test matrix as a sanity check
do i=1,nprocs

   if(mypnum==i)then
   write(*,"(/'Process number: ',i5,',at coordinate (nprow,npcol) = (',i4,',',i4,')')")mypnum,myprow,mypcol
    do j=1,loc_r
    write(*,'(32(2x,f11.6))')a(j,:)!print first 32 matrix elements (nb=32)
    enddo
   endif

   call mpi_barrier(mpi_comm_world,mpierr)
   
enddo

   call mpi_barrier(mpi_comm_world,mpierr)

   t2 = MPI_Wtime()
   if(mypnum == 0)write(*,"(/'Time taken to generate matrix was ', f12.6,' secs')")t2-t1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! Now setup ELPA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   e => elpa_allocate()

   call e%set("na",dimen_s,info)
   call e%set("nev",neigenvals,info)
   call e%set("local_nrows",loc_r,info)
   call e%set("local_ncols",loc_c,info)
   call e%set("nblk",nb,info)
   call e%set("mpi_comm_parent",mpi_comm_world,info)
   call e%set("process_row",myprow,info)
   call e%set("process_col",mypcol,info)
   
!are we using GPUs? 1 = yes, 0 = no
   call e%set("gpu",0,info)
   if(info/=elpa_ok)then
     errormsg = elpa_strerr(info)
     if(mypnum==0)print*,'setting up ELPA GPU, errormsg = ',errormsg
     print*,'no gpu support, stopping...'
     stop
   end if


!if we want to use the 2stage solver on GPUs we need to set the  appropriate kernel

!check the necessary kernel for ELPA2 GPU
!   call e%get("real_kernel",18,info)
!   if(mypnum==0)print*,elpa_int_value_to_string("real_kernel",18,info)

!   if(mypnum==0)print*,'now attempted to setup gpu kernel...'
!   call e%set("real_kernel",elpa_2stage_real_gpu,info)
!   errormsg = elpa_strerr(info)
!   if(mypnum==0)print*,'errormsg = ',errormsg


   info = e%setup()
   if(mypnum==0)print*,'attempted to setup elpa... info = ',info


!the solver type
   call e%set("solver",elpa_solver_1stage,info)!1stage
!   call e%set("solver",elpa_solver_2stage,info)!2stage
   


   call mpi_barrier(mpi_comm_world,mpierr)
   t1 = MPI_Wtime()


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!! DIAGONALISE THAT MATRIX !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   if(mypnum==0)print*,'Diagonalising matrix...'
   call e%eigenvectors(a,eigenvals,z,info)
   if(mypnum==0)print*,'... finished'
   
   if(info/=elpa_ok)then
     print*,'something went wrong, info = ',info,'stopping...'
     stop
   end if

   call mpi_barrier(mpi_comm_world,mpierr)
   t2 = MPI_Wtime()
   if(mypnum == 0)write(*,"(/'Time taken to diagonalise matrix is ', f12.6,' secs')")t2-t1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!! Print eigenvalues and clean up blacs !!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   if(mypnum==0)print*,"Printing eigenvalues..."
   if(mypnum==0)then
      do i=1,dimen_s
      write(*,*)i,eigenvals(i)
      end do
   end if
   
   call elpa_deallocate(e)
   call elpa_uninit()
   
   call blacs_gridexit(ctxt)
   call mpi_finalize(mpierr)
   

end program