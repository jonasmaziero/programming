!###################################################################################################################################
!                                             Random density matrix generators - RDMG
!###################################################################################################################################
subroutine rdm_std(optg, d, rdm) ! Generates a random density matrix using the standard method: rdm = \sum_j p_j U|c_j><c_j|U^†
! Ref: Maziero, J. (2015). Random sampling of quantum states: A survey of methods. Braz. J. Phys. 45, 575.
implicit none
integer :: d  ! Dimension of the random density matrix
complex(8) :: rdm(1:d,1:d)  ! Random density matrix 
complex(8), allocatable :: ru(:,:)  ! Random unitary matrix
real(8), allocatable :: rpv(:)  ! Random probability vector
integer :: j, k, l  ! Auxiliary variable for counters
character(10), dimension(5) :: optg  ! Options for the generators

allocate ( ru(1:d,1:d), rpv(1:d) ) ;   call rpvg(optg, d, rpv)
call rug(optg, d, ru)  ! Allocate memory for and get these random variables

rdm = (0.d0,0.d0)  ! Generates the rdm
do j = 1, d ;   do k = 1, d  ;  do l = 1, d ;   rdm(j,k) = rdm(j,k) + rpv(l)*ru(j,l)*conjg(ru(k,l)) ;   enddo ;   enddo ;   enddo

deallocate ( ru, rpv )

end
!-----------------------------------------------------------------------------------------------------------------------------------
subroutine rdm_ginibre(optg, d, rdm) ! Generates a random density matrix normalizing G*G^†, with G being a Ginibre matrix
! Ref:  \.{Z}yczkowski, K., and Sommers, H.-J. (2001). Induced measures in the space of mixed quantum states. 
!       J. Phys. A: Math. Gen. 34, 7111.
implicit none
integer :: d  ! Dimension of the random density matrix
complex(8) :: rdm(1:d,1:d)  ! Random density matrix 
complex(8), allocatable :: G(:,:), GGd(:,:)  ! For the Ginibre matrix and its product with its adjoint
real(8) :: norm_hs  ! For the Hilbert-Schmidt norm function
character(10), dimension(5) :: optg  ! Options for the generators

allocate ( G(1:d,1:d), GGd(1:d,1:d) ) ;   call ginibre(optg, d, d, G) ;   call  matmul_AAd(d, d, G, GGd) 
rdm = GGd/((norm_hs(d, d, G))**2.d0)  ! Defines the density matrix
deallocate ( G, GGd )

end
!-----------------------------------------------------------------------------------------------------------------------------------
subroutine rdm_bures(optg, d, rdm) ! Generates a random density matrix normalizing (id+U)G*G^†(id+U^†), with G being a 
                                   ! Ginibre matrix and U a random unitary
! Ref: Al Osipov, V., Sommers, H.-J., and \.{Z}yczkowski, K. (2010). Random Bures mixed states and the distribution of their purity. 
!      J. Phys. A: Math. Theor. 43, 055302.
implicit none
integer :: d  ! Dimension of the matrices
complex(8) :: rdm(1:d,1:d)  ! Random density matrix 
complex(8), allocatable :: G(:,:)  ! For the Ginibre matrix
complex(8), allocatable :: U(:,:)  ! For the random unitary matrix
!complex(8), allocatable :: id(:,:)  ! For the indentity matrix
complex(8), allocatable :: A(:,:), AAd(:,:)  ! Auxiliary matrices
real(8) :: norm_hs  ! For the Hilbert-Schmidt norm function
integer :: j  ! Auxiliary variable for counters
character(10), dimension(5) :: optg  ! Options for the generators

allocate ( G(1:d,1:d), U(1:d,1:d), A(1:d,1:d), AAd(1:d,1:d) )
call ginibre(optg, d, d, G) ;   call rug(optg, d, U) ;   forall ( j = 1:d ) U(j,j) = U(j,j) + 1.d0  ! U -> id+U
A = matmul(U,G) ;   call  matmul_AAd(d, d, A, AAd)
rdm = AAd/((norm_hs(d, d, A))**2.d0)  ! Defines the density matrix
deallocate ( G, U, A, AAd )

end
!-----------------------------------------------------------------------------------------------------------------------------------
subroutine rdm_ptrace(optg, d, rdm) ! Generates a random density matrix via partial tracing over a random state vector
! Ref: Mej\'ia, J., Zapata, C., and Botero, A. (2015). The difference between two random mixed quantum states: Exact and asymptotic 
!      spectral analysis. arXiv:1511.07278.
implicit none
integer :: d  ! Dimension of the density matrix
complex(8) :: rdm(1:d,1:d)  ! Random density matrix 
complex(8), allocatable :: rsv(:)  ! For the random state vector
complex(8), allocatable :: proj(:,:)  ! For the projector
character(10), dimension(5) :: optg  ! Options for the generators

allocate ( rsv(1:d*d), proj(1:d*d,1:d*d) )
call rsvg(optg, d*d, rsv) ;   call projector(rsv, d*d, proj)
call partial_trace_a_he(proj, d, d, rdm)
deallocate ( rsv, proj )

end
!-----------------------------------------------------------------------------------------------------------------------------------
subroutine rdm_bds(rrho)  ! Returns a random Bell-diagonal state
implicit none
real(8) :: rpv(1:4)  ! The random probability vector
character(10), dimension(5) :: optg ! Options for the generators
real(8) :: c11, c22, c33  ! Correlation functions
complex(8) :: rrho(1:4,1:4)  ! The random state

 optg = 'std' ;   call rpvg(optg, 4, rpv) 
 c11 = 2.d0*(rpv(1)+rpv(2)) - 1.d0 ;   c22 = 2.d0*(rpv(2)+rpv(3)) - 1.d0 ;   c33 = 2.d0*(rpv(1)+rpv(3)) - 1.d0
 call rho_bds(c11, c22, c33, rrho) 

end
!-----------------------------------------------------------------------------------------------------------------------------------
subroutine rdm_x(rrho)  ! Returns a random X state (in standard form)
implicit none
real(8) :: rn(1:5)  ! Vector of random numbers
character(10), dimension(5) :: optg ! Options for the generators
complex(8) :: rrho(1:4,1:4)  ! The random state
real(8) :: c11, c22, c33, a3, b3, p1, p2, p3, p4  !  Correlation functions, polarizations, and probabilities

optg = 'std'
do
  call rng_unif(optg, 5, -1.d0, 1.d0, rn)
  c11 = rn(1) ; c22 = rn(2) ; c33 = rn(3) ; a3 = rn(4) ; b3 = rn(5)
  p1 = 0.25d0*(1.d0-c33-sqrt((c11+c22)**2.d0 + (a3-b3)**2.d0))
  p2 = 0.25d0*(1.d0-c33+sqrt((c11+c22)**2.d0 + (a3-b3)**2.d0))
  p3 = 0.25d0*(1.d0+c33-sqrt((c11-c22)**2.d0 + (a3+b3)**2.d0))
  p4 = 0.25d0*(1.d0+c33+sqrt((c11-c22)**2.d0 + (a3+b3)**2.d0))
  if ((p1 >= 0.d0) .and. (p2 >= 0.d0) .and. (p3 >= 0.d0) .and. (p4 >= 0.d0)) then
    call rho_x(c11, c22, c33, a3, b3, rrho) ;   exit 
  endif
enddo

end
!-----------------------------------------------------------------------------------------------------------------------------------
subroutine rdm_xs(rrho)  ! Returns a random X state (in standard form), & symmetric under particle exchange
implicit none
real(8) :: rn(1:4)  ! Vector of random numbers
character(10), dimension(5) :: optg ! Options for the generators
complex(8) :: rrho(1:4,1:4)  ! The random state
real(8) :: c11, c22, c33, a3, b3, p1, p2, p3, p4  !  Correlation functions, polarizations, and probabilities

optg = 'std'
do
  call rng_unif(optg, 4, -1.d0, 1.d0, rn)
  c11 = rn(1) ;   c22 = rn(2) ;   c33 = rn(3) ;   a3 = rn(4) ;   b3 = a3
  p1 = 0.25d0*(1.d0-c33-sqrt((c11+c22)**2.d0))
  p2 = 0.25d0*(1.d0-c33+sqrt((c11+c22)**2.d0))
  p3 = 0.25d0*(1.d0+c33-sqrt((c11-c22)**2.d0 + 4.d0*(a3**2.d0)))
  p4 = 0.25d0*(1.d0+c33+sqrt((c11-c22)**2.d0 + 4.d0*(a3**2.d0)))
  if ((p1 >= 0.d0) .and. (p2 >= 0.d0) .and. (p3 >= 0.d0) .and. (p4 >= 0.d0)) then
    call rho_x(c11, c22, c33, a3, b3, rrho) ;   exit 
  endif
enddo

end
!###################################################################################################################################
!                                                   Calling subroutines - RDMG
!###################################################################################################################################
subroutine rdmg(optg, d, rdm)  ! Calls the choosed random density matrix generator
implicit none
integer :: d  ! Dimension of the random density matrix
complex(8) :: rdm(1:d,1:d)  ! The random density matrix
character(10), dimension(5) :: optg  ! Options for the generators

     if ( optg(5) == "std" ) then ;   call rdm_std(optg, d, rdm)
else if ( optg(5) == "ginibre" ) then ;   call rdm_ginibre(optg, d, rdm)
else if ( optg(5) == "bures" ) then ;   call rdm_bures(optg, d, rdm)
else if ( optg(5) == "ptrace" ) then ;   call rdm_ptrace(optg, d, rdm)
     endif

end
!###################################################################################################################################