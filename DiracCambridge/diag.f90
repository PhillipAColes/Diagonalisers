!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! This program sets up blacs environment and calls the pdsyev/pdsyevd/pdsyevr/pdsyevx/ELPA 
! eigensolver to solve the eigenvector and eigenvalue problem for a dense double- precision 
! real symmetric matrix.
! There are two algorithms implemented by ELPA: 1stage and 2stage. The latter is
! faster on CPUs and the former is faster using GPUS.
! For GPU version use the command  'call e%set("gpu",1,info)'. To use CPUs use
! 'call e%set("gpu",0,info)'. CPUs are default so this can also be commented out.
! In order to use the 2stage ELPA solver on GPUs the kernel must be set by
! 'call e%set("real_kernel",elpa_2stage_real_gpu,info)'. Check to see what kernels
! have been built in by running the 'elpa2_print_kernels_openmp' executable.
!
! Notes:
! 1) exact workspace requirements for ELPA unknown. Requesting 8*(4*n*n)/1024/1024/1024 Gb 
!    (where n = matrix dimension) total memory should be sufficient for ALL solvers except pdsyevx.
! 2) to use ELPA2 GPU, kernel must be set AFTER solver
! 
! KNOWN BUGS: 
! 1) currently elpa1 produces incorrect eigenvalues when >20 nodes are used
!    for a matrix of dimension greater than 200,000. If the number of
!    nodes is REDUCED then elpa1 works fine... reasons unknown. No such problem with elpa2.
! 2) pdsyevx generates incorrect eigenvalues for large matrices. Reasons unknown but
!    it is connected with nroots
! 3) pdsyevr generates incorrect eigenvalues for large matrices when most or all roots
!    are requested.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program diagmatrix

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!comment out if not using ELPA !!!!
use iso_c_binding
use elpa
use mpi
!use test_scalapack
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

implicit none

character(100)::buf!buffer
integer::iounit=20
character(100)::inpfile,outfile,filename
integer::dimen_s!matrix dimension
integer::imatrix!1 for matrix to be read, 2 for matrix to be generated
integer::mpierr

!relating to processor grid
integer::mypnum, nprocs!process index (from 0 to nprocs-1), total number of processes
integer::ctxt!grid context i.e. a (non-unique) process grid used for a specific operation
integer::nprow,npcol!number of process rows and columns to use
integer::myprow,mypcol!row and column coordinate of the calling process

!relating to matrix description
integer::desca(9),descz(9)
integer::nb!block number (dimension that constitutes a matrix block, we are free to define this)
integer::loc_c,loc_r,llda!number of columns and rows of (sub)matrix owned by particular process, local leading dimension
integer::i_loc,j_loc!row and column index of element in local array
integer::rsrc,csrc!grid coordinates of the process that owns the matrix element with index i,j in the global matrix

!relating to diagonalisation
integer::trilwmin,lel1,lel2!all used to calculate lwork
double precision, allocatable::work(:)!work array, of size lwork
integer,allocatable::iwork(:)!integer work array, of size liwork
integer::lwork,liwork
double precision, allocatable::w(:)!array for storing eigenvalues
double precision, allocatable::zloc(:,:)!local array of eigenvectors
integer::lwork_,liwork_!used for checking lwork and liwork
double precision::work_(10)!forst element is the estimated value of lwork
integer::iwork_(10)!the first element is the estimated value of liwork
logical::vects_write = .false.
double precision, allocatable::local_vecs(:)
double precision::maxcontrib!maximum coefficient in the variational calculation
integer::maxterm!index of the maximum contributing coefficient

!only necessary for pdsyevx/pdsyevr
double precision::vl,vu!the lower and upper ranges of eigenvalues to be found
integer::il,iu!index of smallest and largest eigenvalue to be returned
double precision::abstol,orfac!convergance threshold, re-orthoganlisation threshold
integer::neig!number of eigenvalues requested
integer::nn,np0,mq0!necessary to calculate optimal lwork
integer::clustersize!number of eigenvalues in a cluster, see mkl manual for definition
integer::nnp!necessary for calculation of liwork
integer,allocatable::ifail(:)!indices of eigenvalues that failed to converge
integer,allocatable::icluster(:)!indices of eigenvectors that are not orthogonal, see mkl documentation
double precision,allocatable::gap(:)
integer::nvals=0,nvects!number of eigenvalues and eigenvectors to be computed
real::frac

!relating to ELPA
integer:: elpa_version
class(elpa_t), pointer::e
character(200)::errormsg

!for matrix norm
integer::normlwork
double precision::matnorm
double precision,allocatable::normwork(:)

!relating to physical system
double precision::zpe=0!zero point energy
double precision::emax=100000.0!energy threshold for eigenvectors

!relating to matrix reading
double precision, allocatable::a_temp(:)!temporary storage for rows of A
double precision, allocatable::aloc(:,:)!submatrix a, which is formed of blocks from the global matrix A

!for matrix generation
integer::seed(1000000)

!external Scalapack functions
integer,external::numroc,iceil
double precision,external::pdlamch,pdlange

!external MPI functions
!double precision,external::MPI_Wtime

!general
integer::i,j,info,ios,incount,idimen,jdimen
integer::diagonaliser
double precision::t2,t1!timer

!input file
character(100)::param,pval1,pval2,matfile
integer::iunit

!debugging
double precision,allocatable::mat(:,:)
integer::chkptIO=30,e_chkptIO=40

!Initialise MPI communicator
   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world,mypnum,mpierr)
   call mpi_comm_size(mpi_comm_world,nprocs,mpierr)

!get system info on number of processes, and current process index
!   call blacs_pinfo(mypnum, nprocs)

!   if(nprocs<1)then
!   call blacs_setup( mypnum, nprocs)
!   end if

!setup default system context
   call blacs_get(0,0,ctxt)

!determine number of process rows and columns to use
   do i=1,int( sqrt( dble(nprocs) ) + 1 )
      if(mod(nprocs,i) .eq. 0) nprow = i
   end do
   npcol = nprocs/nprow

!setup process grid in 'r'ow or 'c'olumn major order
   call blacs_gridinit(ctxt,'r',nprow,npcol)

!holds up processes in the current context, until all of them have caught up
   call blacs_barrier(ctxt,'a')

!returns information on the grid tied to the context ctxt
   call blacs_gridinfo(ctxt,nprow,npcol,myprow,mypcol)

   write(*,"(/'Process = ',i4,':',i4,' Grid-coord (',i4,',',i4,') PROW = ',i4,':',i4,' PCOL = ',i4,':',i4)") mypnum,nprocs,myprow,mypcol,myprow,nprow,mypcol,npcol


!open input file and read input parameters
   iunit = 20
   call getarg(1,inpfile)
   open(unit=iunit,file=trim(inpfile))
   
   do
      read(iunit,'(a100)',iostat=ios)buf
      !if(mypnum==0)print*,buf
      if(ios<0)exit
      if(trim(buf)=="")cycle
      read(buf,*,iostat=ios)param,pval1,pval2
      if(trim(pval1)=="")then
      if(mypnum==0)print*,"No input for ",trim(param)," given. Stopping"
      stop
      end if
      !
      select case(trim(param))
      !
      case("diagonaliser","diagonalizer","DIAGONALISER","DIAGONALIZER")
         !
         select case(trim(pval1))
         !
         case("ELPA","ELPA2","elpa","elpa2")
         if(mypnum==0)print*,"user has specified ",trim(pval1)," to be used"
         !
         diagonaliser=5
         !
         case("PDSYEVD","pdsyevd")
         if(mypnum==0)print*,"user has specified ",trim(pval1)," to be used"
         !
         diagonaliser=1
         !
         case("PDSYEV","pdsyev")
         if(mypnum==0)print*,"user has specified ",trim(pval1)," to be used"
         !
         diagonaliser=2
         !
         case("PDSYEVX","pdsyevx")
         if(mypnum==0)print*,"user has specified ",trim(pval1)," to be used"
         !
         diagonaliser=3
         !
         case("PDSYEVR","pdsyevr")
         if(mypnum==0)print*,"user has specified ",trim(pval1)," to be used"
         !
         diagonaliser=4
         !
         case default
         !
         if(mypnum==0)print*,"No diagonaliser recognised. Stopping."
         stop
         !
         end select
         !
      case("matrix","mat","MATRIX","MAT")
         !
         select case(trim(pval1))
         !
         case("gen","generate","GEN","GENERATE")
         !
         read(pval2,*)dimen_s
         !
         if(dimen_s==0)then
         if(mypnum==0)print*,"No matrix dimension supplied. Stopping"
         stop
         end if
         !
         if(mypnum==0)print*,"user has specified a matrix of dimension ",dimen_s," be generated"
         !
         imatrix = 2
         !
         case("read","READ")
         !
         matfile=pval2
         !
         if(trim(matfile)=="")then
         if(mypnum==0)print*,"No matrix file supplied. Stopping"
         stop
         end if
         !
         imatrix = 1
         !
         if(mypnum==0)print*,"The matrix file ",trim(matfile)," will be read and diagonalised"
         !
         case default
         !
         if(mypnum==0)print*,"User needs to specify whether matrix will be read or generated. Stopping"
         stop
         !
         end select
         !
      case("zpe","ZPE")
         !
         read(pval1,*)zpe
         !
      case("nvals","NVALS","neigenvals","NEIGENVALS","neigen","NEIGEN")
         !
         read(pval1,*)nvals
         !
      case("eigenvects","EIGENVECTS")
         !
         vects_write = .true.
         !
      case("enercut","ENERCUT","enermax","ENERMAX")
         !
         read(pval1,*)emax
         !
      case default
         !
         if(mypnum==0)print*,"No input params supplied. Stopping"
         stop
         !
      end select
   end do



!select whether matrix is to be read or pseudo-randomly generated
   select case(imatrix)
      !
      case(1)
      !
   !open matrix file and check header
      open(chkptIO,form='unformatted',action='read',position='rewind',status='old',file=trim(matfile))
      !
      read(chkptIO) dimen_s
      if (mypnum == 0)write(*,"(/'Global matrix has dimension',i5)")dimen_s
      !
      read(chkptIO) buf(1:12)
      if (buf(1:12)/='Start matrix') then
        write (*,"(' matrix ',a,' has bogus header: ',a,' should be _Start matrix_')")'matrix2_2.chk',buf(1:12)
        stop 'bogus file format'
      end if
      !
      nb = 64!block size
      loc_r = numroc(dimen_s,nb,myprow,0,nprow)!essentially number of rows of submatrix (i.e. number of rows of local array owned by process)
      loc_c = numroc(dimen_s,nb,mypcol,0,npcol)!essentially number of columns of submatrix
      llda = max(1,loc_r)!local leading dimension
      !
   !initialise the array descriptor of the distributed matrix, presumably ctxt provides info of the grid setup
      call descinit(desca,dimen_s,dimen_s,nb,nb,0,0,ctxt,llda,info)!info = 0 for successful exit
      call descinit(descz,dimen_s,dimen_s,nb,nb,0,0,ctxt,llda,info)!info = 0 for successful exit
      !
      write(*,"(/'Allocating local matrix a, with dimension (',i5,'x',i5,') Belonging to process ',i4,' at coord (',i4,',',i4')')") loc_r,loc_c,mypnum,myprow,mypcol
      allocate(aloc(loc_r,loc_c),a_temp(dimen_s))
      allocate(zloc(loc_r,loc_c),w(dimen_s))
      !
      !
   !start timer for reading in matrix
      call blacs_barrier(ctxt, 'a')
      t1 = MPI_Wtime()
      !
   !now we read matrix into local array
      do i=1,dimen_s
      read(chkptIO)a_temp
        do j=1,dimen_s
          call infog2l(i,j,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)
          if(myprow==rsrc .and. mypcol==csrc)aloc(i_loc,j_loc)=a_temp(j)
        end do
      end do
      !
      read(chkptIO) buf(1:10)
      !
      if (buf(1:10)/='End matrix') then
        write (*,"(' matrix ',a,' has bogus header: ',a)")'matrix2_2.chk',buf(1:10)
        stop 'bogus file format'
      end if
       !
   !stop timer for reading inmatrix
      call blacs_barrier(ctxt, 'a')
      t2 = MPI_Wtime()
      if(mypnum == 0)write(*,"(/'Time taken to read matrix is ', f12.6,' secs')")t2-t1
      !
      !
      close(chkptIO)
      !
      case(2)
      !
      !
      call blacs_barrier(ctxt, 'a')
      if (mypnum == 0)write(*,"(/'User has selected to generate pseudo-random matrix with dimension',i7)")dimen_s
      !
      nb = 64!block number (dimension that constitute a block, e.g., for nb=64, a block is 64x64 elements)
      loc_r = numroc(dimen_s,nb,myprow,0,nprow)!essentially number of rows of submatrix (i.e. number of rows of local array owned by process)
      loc_c = numroc(dimen_s,nb,mypcol,0,npcol)!essentially number of columns of submatrix
      llda = max(1,loc_r)!local leading dimension
      !
   !initialise the array descriptor of the distributed matrix, presumably ctxt provides info of the grid setup
      call descinit(desca,dimen_s,dimen_s,nb,nb,0,0,ctxt,llda,info)!info = 0 for successful exit
      call descinit(descz,dimen_s,dimen_s,nb,nb,0,0,ctxt,llda,info)!info = 0 for successful exit
      !
      write(*,"(/'Allocating local matrix a, with dimension (',i5,'x',i5,') Belonging to process ',i4,' at coord (',i4,',',i4')')") loc_r,loc_c,mypnum,myprow,mypcol
      allocate(aloc(loc_r,loc_c),a_temp(dimen_s))
      allocate(zloc(loc_r,loc_c),w(dimen_s))
      !
      !
      call blacs_barrier(ctxt, 'a')
      t1 = MPI_Wtime()
      !
   !create array of pseudo-random numbers with which we use to fill matrix
      do i = 1, dimen_s
        seed(i) = 123456789-i-1;
        seed(i) = mod((1103515245 * seed(i) + 12345),1099511627776);
      enddo
      !
      call blacs_barrier(ctxt, 'a')
      if (mypnum == 0)write(*,"(/'Begin generating matrix')")
      !
      do j = 1,dimen_s
        do i = 1,dimen_s
          call infog2l(i,j,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc
          if (myprow == rsrc .and. mypcol == csrc) then
            if(i>=j) aloc(i_loc,j_loc) = dble(seed(i))/dble(1099511627776.0)
            if(i < j) aloc(i_loc,j_loc) = dble(seed(j))/dble(1099511627776.0)
            if(i==j) aloc(i_loc,j_loc) = aloc(i_loc,j_loc) + dble(10.0) + dble(j)
          endif
        enddo
      enddo
      !
      !
      !
   !small, simple, double-real positive symmetric test matrix
!   do i = 1,dimen_s
!     do j = i,dimen_s
!       call infog2l(i,j,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc
!       if (myprow == rsrc .and. mypcol == csrc) then
!          aloc(i_loc,j_loc) = 0.00001d0 * dble(j)
!       endif
!       call infog2l(j,i,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc
!       if (myprow == rsrc .and. mypcol == csrc) then
!          aloc(i_loc,j_loc) = 0.00001d0 * dble(j)
!       endif
!       call infog2l(i,j,desca,nprow,npcol,myprow,mypcol,i_loc,j_loc,rsrc,csrc)!returns i_loc,j_loc,rsrc,csrc
!       if (myprow == rsrc .and. mypcol == csrc) then
!         if(i==j) aloc(i_loc,j_loc) = aloc(i_loc,j_loc) + dble(10.0) + dble(j)
!       endif
!     enddo
!   enddo
      !
      !
      !
      call blacs_barrier(ctxt, 'a')
      t2 = MPI_Wtime()
      if (mypnum == 0)write(*,"(/'Matrix generation completed')")
      if(mypnum == 0)write(*,"(/'Time taken to generate matrix is ', f12.6,' secs')")t2-t1
      !
      !
   case default
   !
   write(*, '(/a)') 'error: no matrix specified'
   stop
   !
   !
end select

!if user doesn't request a specific number of eigenpairs, generate all eigenpairs
if(nvals==0)nvals=dimen_s

!select which diagonaliser is to be used
!NB: generally elpa2 or pdsyevd should be used. pdsyev is slow and pdsyevx is unstable at high matrix dimension
select case(diagonaliser)
   !
   !!!!!!!!!!!!!!! PDSYEVD !!!!!!!!!!!!!!!!!!!!
   case(1)
   !
 !pdsyevd computes all eigenvalues
   nvals = dimen_s
   !
   trilwmin = 3*dimen_s + max( nb*( loc_r+1 ), 3*nb)
   lwork = max( 1 + 6*dimen_s + 2*loc_r*loc_c, trilwmin) + 2*dimen_s
   liwork = 7*dimen_s + 8*npcol + 2
   !
   allocate(work(lwork),iwork(liwork),stat=info)
   !
   if (mypnum == 0) then
   write(*,"(/'lwork =  ', i16, ' liwork = ', i16, ' llda = ', i16, ' loc_r = ', i16, ' loc_c = ', i16)") lwork, liwork, llda, loc_r, loc_c
   endif
   !
   call blacs_barrier(ctxt, 'a')
   t1 = MPI_Wtime()
   !
   if (mypnum == 0)write(*,'(/a)')'Calling PDSYEVD...'
   call pdsyevd('V','L',dimen_s,aloc,1,1,desca,w,zloc,1,1,descz,work,lwork,iwork,liwork,info)
   if (mypnum == 0)write(*,'(/a)')'... done!'
   !
   call blacs_barrier(ctxt, 'a')
   t2 = MPI_Wtime()
   !
   if(mypnum == 0)write(*,"(/'PDSYED finished, INFO =  ', i16)") info
   if(mypnum == 0)write(*,"(/'Time taken to diagonalise matrix is ', f12.6,' secs')")t2-t1
   !
   !
   !!!!!!!!!!!!!!! PDSYEV !!!!!!!!!!!!!!!!!!!!!
   case(2)
   !
 !pdsyev computes all eigenvalues
   nvals = dimen_s
   !
   lel1 = nb**2 * (dimen_s/(nb*nprow)+dimen_s/(nb*npcol)+3)
   lel2 = nb*((dimen_s-1)/(nb*nprow*npcol)+1)
   lwork = dimen_s*5 + max(2*dimen_s,lel1) +dimen_s*lel2 + 1
   !
   allocate(work(lwork), stat=info)
   !
   if (mypnum == 0) then
     write(*,"(/'lwork =  ', i16, ' lda = ', i16, ' loc_r = ', i16, ' loc_c = ', i16)") lwork, llda, loc_r, loc_c
   endif
   !
   call blacs_barrier(ctxt, 'a')
   t1 = MPI_Wtime()
   !
   if (mypnum == 0)write(*,'(/a)')'Calling PDSYEV...'
   call pdsyev('V','L',dimen_s,aloc,1,1,desca,w,zloc,1,1,descz,work,lwork,info)
   if (mypnum == 0)write(*,'(/a)')'... done!'
   !
   call blacs_barrier(ctxt, 'a')
   t2 = MPI_Wtime()
   !
   if(mypnum == 0)write(*,"(/'PDSYEV finished, INFO =  ', i16)") info
   if(mypnum == 0)write(*,"(/'Time taken to diagonalise matrix is ', f12.6,' secs')")t2-t1
   !
   !
   !!!!!!!!!!!!!!! PDSYEVX !!!!!!!!!!!!!!!!!!!!
   case(3)
   !
   !
   vl = 0.0
   vu = 100000.0
   il = 1
   iu = nvals
   nvects = nvals
   neig = nvals
   !
   abstol = PDLAMCH(ctxt,'U')!absolute error tolerance for eigenvalues
!   abstol = 2.0*PDLAMCH(ctxt,'S')
   !
   orfac = 1.0e-7!1.0e-6
   !
   nn = max(dimen_s, nb, 2)
   np0 = numroc(nn,nb,0,0,nprow)
   mq0 = numroc(max(neig,nb,2),nb,0,0,npcol)
   lwork = 5*dimen_s + max(5*nn , (np0*mq0 + 2*nb*nb)) + iceil(neig , nprow*npcol)*nn
   !
   !
   normlwork = 1e6 !A guess
   allocate(normwork(normlwork),stat=info)
   matnorm = pdlange('1',loc_r,loc_c,aloc,1,1,desca,normwork)
   deallocate(normwork)
   !
   if(mypnum == 0)write(*,*)'matrix Frobenius-norm = ',matnorm
   if(mypnum == 0)write(*,"(/'A cluster is defined by eigenvalues within orfac*2*norm(A) =  ', f12.6)")orfac*2*matnorm
   !
   clustersize = 10!an estimate
   lwork = lwork + clustersize*dimen_s
   !
   liwork = liwork*1.5
   lwork = lwork*1.5!increase work array
   !
   nnp = max(dimen_s , nprow*npcol + 1 , 4)
   liwork = 6*nnp
   !
   !
   lwork_=-1
   liwork_=-1
   !
   call pdsyevx('V','A','L',dimen_s,aloc,1,1,desca,vl,vu,il,iu,abstol,nvals,nvects,w,orfac,zloc,1,1,descz,work_,lwork_,iwork_,liwork_,ifail,icluster,gap,info)
   !
   if (mypnum == 0)write(*,"(/'Scalapack estimated lwork = ', f18.1,' and liwork = ',i16)")work_(1),iwork_(1)
   !
   allocate(work(lwork),iwork(liwork))
   allocate(ifail(dimen_s),icluster(2*nprow*npcol),gap(nprow*npcol))
   !
   if (mypnum == 0) then
     write(*,"(/'lwork =  ', i16, ' lda = ', i16, ' loc_r = ', i16, ' loc_c = ', i16)") lwork, llda, loc_r, loc_c
   endif
   !
   call blacs_barrier(ctxt, 'a')
   t1 = MPI_Wtime()
   !
   if (mypnum == 0)write(*,'(/a)')'Calling PDSYEVX...'
   call pdsyevx('V','A','L',dimen_s,aloc,1,1,desca,vl,vu,il,iu,abstol,nvals,nvects,w,orfac,zloc,1,1,descz,work,lwork,iwork,liwork,ifail,icluster,gap,info)
   if (mypnum == 0)write(*,'(/a)')'...done!'
   !
   call blacs_barrier(ctxt, 'a')
   t2 = MPI_Wtime()
   !
   if(mypnum == 0)write(*,"(/'PDSYEX finished, INFO =  ', i16)") info
   if(mypnum == 0)write(*,"(/'Time taken to diagonalise matrix is ', f12.6,' secs')")t2-t1
   !
   !
   if(mod(info,2)/=0)then
   write(*,"(/'One or more eigenvalues failed to converge, their indices are')")
     do i=1,dimen_s
     if(ifail(i)/=0)write(*,'(i16)')ifail(i)
     end do
   end if
   !
   if(mod(info/2,2)/=0)then
   write(*,"(/'Eigenvalues corresponding to one or more clusters failed to orthogonalise, their indices are')")
     do i=1,2*nprow*npcol
     if(icluster(i)/=0)write(*,'(i16,2x,i16)')i,icluster(i)
     end do
   end if
   !
   do i=1,dimen_s
     if(ifail(i)/=0)write(*,"(/'Eigenvector ', i16, ' failed to converge')")i
   end do
   !
   do i=1,nprow*npcol
     if(gap(i)/=0)write(*,"(/'Gap ', i16, ' is',i16)")i,gap(i)
   end do
   !
   call blacs_barrier(ctxt, 'a')
   !
   !
   !!!!!!!!!!!!!!! PDSYEVR !!!!!!!!!!!!!!!!!!!!
   case(4)
   !
   !
   vl = 0.0
   vu = 100000.0
   !
 !for computing a subset of eigenpairs
   il = 1 !index of the smallest eigenvalue to be returned
   iu = nvals !index of the largest eigenvalue to be returned
   nvects = nvals !number of eigenvectors to be computed
   neig = nvals !number of eigenvectors requested
   !
   nn = max(dimen_s, nb, 2)
   np0 = numroc(nn,nb,0,0,nprow)
   mq0 = numroc(max(neig,nb,2),nb,0,0,npcol)
   lwork = 2 + 5*dimen_s + max(18*nn , (np0*mq0 + 2*nb*nb)) + ( 2 + iceil(neig , nprow*npcol))*nn
   !
   nnp = max(dimen_s , nprow*npcol + 1 , 4)
   liwork = 12*nnp + 2*dimen_s
   !
   !
   allocate(work(lwork),iwork(liwork))
   !
   if (mypnum == 0) then
     write(*,"(/'lwork =  ', i16, ' lda = ', i16, ' loc_r = ', i16, ' loc_c = ', i16)") lwork, llda, loc_r, loc_c
   endif
   !
   call blacs_barrier(ctxt, 'a')
   t1 = MPI_Wtime()
   !
   if (mypnum == 0)write(*,'(/a)')'Calling PDSYEVR...'
   !'I' - all eigenvalues within interval [il,iu] will be found
   call pdsyevr('V','I','L',dimen_s,aloc,1,1,desca,vl,vu,il,iu,nvals,nvects,w,zloc,1,1,descz,work,lwork,iwork,liwork,info)
   if (mypnum == 0)write(*,'(/a)')'...done!'
   !
   call blacs_barrier(ctxt, 'a')
   t2 = MPI_Wtime()
   !
   if(mypnum == 0)write(*,"(/'PDSYER finished, INFO =  ', i16)") info
   if(mypnum == 0)write(*,"(/'Time taken to diagonalise matrix is ', f12.6,' secs')")t2-t1
   !
   !
   call blacs_barrier(ctxt, 'a')
   !
   !
   !!!!!!!!!!!!!!! ELPA !!!!!!!!!!!!!!!!!!!!
   case(5)
   !
!Initialise ELPA, most recent elpa version = 20190524
   if (elpa_init(20190524)/= elpa_ok)then
    print*,"ELPA API not supported"
    stop
   end if
   !
   !
   e => elpa_allocate()
   !
   call e%set("na",dimen_s,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   call e%set("nev",nvals,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   call e%set("local_nrows",loc_r,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   call e%set("local_ncols",loc_c,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   call e%set("nblk",nb,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   call e%set("mpi_comm_parent",mpi_comm_world,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   call e%set("process_row",myprow,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   call e%set("process_col",mypcol,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   !
   call e%set("blacs_context",ctxt,info)
   if(mypnum==0)print*,'info=',elpa_strerr(info)
   !
!are we using GPUs? 1 = yes, 0 = no
   call e%set("gpu",0,info)
   if(info/=elpa_ok)then
     errormsg = elpa_strerr(info)
     if(mypnum==0)print*,'setting up ELPA GPU, errormsg = ',errormsg
     print*,'no gpu support, stopping...'
     stop
   end if
   !
!if we want to use the 2stage solver on GPUs we need to set the  appropriate kernel
   !
!check the necessary kernel for ELPA2 GPU
!   call e%get("real_kernel",18,info)
!   if(mypnum==0)print*,elpa_int_value_to_string("real_kernel",18,info)
   !
!   if(mypnum==0)print*,'now attempted to setup gpu kernel...'
!   call e%set("real_kernel",elpa_2stage_real_gpu,info)
!   errormsg = elpa_strerr(info)
!   if(mypnum==0)print*,'errormsg = ',errormsg
   !
   info = e%setup()
   if(mypnum==0)print*,'attempted to setup elpa... info = ',info
   !
!the solver type
!   call e%set("solver",elpa_solver_1stage,info)!1stage
   call e%set("solver",elpa_solver_2stage,info)!2stage
   !
!kernel type
!   call e%set("real_kernel",elpa_2stage_real_avx512_block2,info)
   !
   call blacs_barrier(ctxt, 'a')
   t1 = MPI_Wtime()
   !
   if (mypnum == 0)write(*,'(/a)')'Calling ELPA eigensolver...'
   call e%eigenvectors(aloc,w,zloc,info)
   if(info/=elpa_ok)then
     print*,'something went wrong, info = ',info,'stopping...'
     stop
   end if
   !
   if(mypnum==0)print*,'... done!'
   !
   call blacs_barrier(ctxt, 'a')
   t2 = MPI_Wtime()
   !
   if(mypnum == 0)write(*,"(/'ELPA finished, INFO =  ', i16)") info
   if(mypnum == 0)write(*,"(/'Time taken to diagonalise matrix is ', f12.6,' secs')")t2-t1
   !
   case default
   !
   write(*,'(/a)')'Unknown diagonaliser'
   stop
   !
end select

call blacs_barrier(ctxt, 'a')


if(mypnum==0)then
   write(*,"(/'Printing eigenvalues')")
   do i=1,nvals
     write(*,*)i,w(i)-zpe
   end do
end if


!Write eigenvectors if specified
if(vects_write)then

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!! Ahmed's procedure for writing vectors and energies.chk files !!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    call blacs_barrier(ctxt, 'a')
    !
    filename = 'energies.chk'
    !Allocate space for vectors
    allocate(local_vecs(dimen_s), stat=info)
    !
    if(myprow==0 .and. mypcol==0) then
        !
        write(*, '(/1x,a,1x,a,1x,a,1x,i3)') 'write eigenvectors into file=', trim(filename), 'iounit=', chkptIO
        !
        open(e_chkptIO, form='unformatted', action='write', status='unknown', position='rewind', file=filename, buffered='yes')
        !
        if (info/=0) then
          write(*, '(/a,1x,a)') 'error while opening file=', trim(filename)
          stop
        endif
        !
        !nvals = 0
        !do idimen=1, dimen_s
        !  if (abs(w(idimen)-zpe)<=emax) then
        !     nvals = nvals + 1
        !  else
        !    exit
        !  endif
        !enddo
        !
        write(e_chkptIO) 'Start energies'
        write(e_chkptIO) dimen_s, nvals
        write(e_chkptIO) w(1:nvals)
        write(e_chkptIO) 'Start contrib'
        !
        filename = 'j0eigen_vectors.chk'
        if(mypnum==0)  open(chkptIO, form='unformatted',access='STREAM',action='write',position='rewind',status='unknown', file=filename)
    endif
    !
    !
    if( (myprow.eq.0) .and. (mypcol.eq.0) ) then
       call igebs2d(ctxt, 'all', 'i-ring', 1, 1, nvals, 1 )
    else
       call igebr2d(ctxt, 'all', 'i-ring', 1, 1, nvals, 1, 0, 0 )
    endif
    !
    !
    !
    if(mypnum==0)print*,nvals,' out of ',dimen_s,'eigevectors will be written'
    !
    do jdimen=1, nvals
       !
     !Clear the local_vector
       local_vecs=0
       !
       do idimen=1, dimen_s
          !
          call infog2l(idimen, jdimen, descz, nprow, npcol, myprow, mypcol, i, j, rsrc, csrc)
          !
          if (myprow==rsrc .and. mypcol==csrc) then
          !
          local_vecs(idimen) = zloc(i,j)
          !
          endif
          !
       enddo
      !
      !Gather all vectors
      if(mypnum==0) write(*,'(/a,i)') 'writing column ',jdimen
      call dgsum2d(ctxt, 'all', ' ', dimen_s, 1, local_vecs, -1, -1, 0 )
      !
      if(mypnum==0) then
         write(chkptIO) local_vecs
         maxcontrib=0
         do i=1,dimen_s
            if (abs(local_vecs(i))>=maxcontrib) then 
              maxcontrib = abs(local_vecs(i))
              maxterm = i
            endif
         enddo
         write(e_chkptIO) maxterm,maxcontrib
      endif
      !
    enddo
    !
  !write(chkptIO) 'End vectors'
    !
    call blacs_barrier(ctxt, 'a')
    !
    t2 = MPI_Wtime()
    !
    if (mypnum == 0) then
      write(*,'(/a,f12.6,a)') 'Time to write vectors: ',t2-t1,' sec'
    endif
    !
    if(mypnum==0) close(chkptIO)
    if(mypnum==0) close(e_chkptIO)
    !
!    write(*, '(1x,a,1x,i3)') 'done for iounit=', chkptIO
    !
    call blacs_barrier(ctxt, 'a')
    !
end if


call blacs_gridexit(ctxt)
call blacs_exit(0)


end program