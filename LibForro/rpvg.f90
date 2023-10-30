!###################################################################################################################################
!                                          Random probability vector generators
!###################################################################################################################################
subroutine rpv_zhsl(optg, d, rpv) ! Generates a random probability vector of dimension d using the Zyczkowski-Horodecki-Sanpera-Lewenstein method
! Zyczkowski, K., Horodecki, P., Sanpera, A., and Lewenstein, M. (1998). Volume of the set of separable states, Phys. Rev. A 58, 883. 
implicit none
integer :: d   ! Dimension of the probability vector
real(8) :: rpv(1:d)  ! Random probability vector 
real(8), allocatable :: rn(:)  ! Vector of random numbers
real(8) :: norm ! Normalization for the prv (auxiliary variable)
integer :: j  ! Auxiliary variable, counter for do
character(10), dimension(5) :: optg  ! Options for the generators

allocate( rn(1:d-1) ) ! Allocate memory for the vector of random numbers
call rng(optg, d-1, rn) ! Call the choosed random number generator
 
rpv(1) = 1.d0 - (rn(1))**(1.d0/(dble(d-1))) ;  norm = rpv(1) ! These lines implement the ZHSL method
if ( d >= 3 ) then
  do j = 2, d-1 ;   rpv(j) = (1.d0 - norm)*(1.d0 - (rn(j))**(1.d0/(dble(d-j)))) ;   norm = norm + rpv(j) ;   enddo
endif
rpv(d) = 1.d0 - norm

deallocate( rn ) ! Deallocate the memory used with the vector of random numbers

end
!------------------------------------------------------------------------------------------------------------------------------------
subroutine rpv_norm(optg, d, rpv)  ! Generates a random-unbiased probability vector of dimension d using the normalization method & shuffling
! Ref: Maziero, J. (2015). Generating pseudo-random discrete probability distributions. Braz. J. Phys. 45, 377.
implicit none
integer :: d   ! Dimension of probability vector
real(8) :: rpv(1:d)  ! Random-unbiased probability vector
real(8), allocatable :: rn(:)  ! Vector of random numbers
real(8) :: norm  ! Normalization (auxiliary variable)
integer :: j  ! Counter for do (auxiliary variable)
integer, allocatable :: rperm(:)  ! vector for the random permutation of {1,2,...,d}
character(10), dimension(5) :: optg  ! Options for the generators

allocate( rn(1:d-1) )  ! Allocate memory for the vector of random numbers
call rng(optg, d-1, rn)  ! Call the choosed random number generator

allocate( rperm(1:d) )  ! Allocate memory for the vector of random permutation of {1,2,...,d}
call rand_perm(optg, d, rperm)  ! Gets the random permutation 

rpv(rperm(1)) = rn(1) ;  norm = rpv(rperm(1))  ! These lines implement the normilization method with shuffling
if ( d >= 3 ) then
  do j = 2, d-1
    rpv(rperm(j)) = (1.d0 - norm)*rn(j) ;  norm = norm + rpv(rperm(j))
  enddo
endif
rpv(rperm(d)) = 1.d0 - norm

deallocate( rn, rperm ) ! Deallocate the memory used with these two variables

end
!------------------------------------------------------------------------------------------------------------------------------------
subroutine rpv_trig_b(optg, d, rpv) ! Generates a random probability vector of dimension d using the trigonometric method
! Ref: Maziero, J. (2015). Generating pseudo-random discrete probability distributions. Braz. J. Phys. 45, 377.
implicit none
integer :: d   ! Dimension of the random probability vector
real(8) :: rpv(1:d)  ! Random probability vector
real(8), allocatable :: rn(:)  ! Vector of random numbers
real(8), allocatable :: vec_theta(:)  ! Vector of angles used in this method
real(8) :: prod_cos  ! Store the product of the squared cossines
integer :: j, k  ! Auxiliary variables
character(10), dimension(5) :: optg  ! Options for the generators

allocate( rn(1:d-1), vec_theta(1:d-1) ) ! Allocate memory for the vector of random numbers and for the vector of angles
call rng(optg, d-1, rn) ! Call the choosed random number generator

prod_cos = 1.d0  ! These lines implement the trigonometric method
do k = d-1, 1, -1
  vec_theta(k) = acos(sqrt(rn(d-k)))
  rpv(k+1) = ((sin(vec_theta(k)))**2.d0)*prod_cos
  prod_cos = prod_cos*((cos(vec_theta(k)))**2.d0)
enddo
rpv(1) = prod_cos

deallocate( rn, vec_theta )  ! Liberate the memory used by these variables

end
!------------------------------------------------------------------------------------------------------------------------------------
subroutine rpv_trig(optg, d, rpv) ! Generates a random-unbiased probability vector of dimension d using the trigonometric method & shuffling
! Ref: Maziero, J. (2015). Generating pseudo-random discrete probability distributions. Braz. J. Phys. 45, 377.
implicit none
integer :: d   ! Dimension of the random(-unbiased) probability vector
real(8), allocatable :: rpv_(:)  ! Random probability vector
real(8) :: rpv(1:d)  ! Random-unbiased probability vector 
integer :: j  ! Auxiliary variable
integer, allocatable :: rperm(:)  ! Vector for the random permutation of {1,2,...,d}
character(10), dimension(5) :: optg  ! Options for the generators

allocate( rpv_(1:d) )  ! Allocates memory for the random probability vector
call rpv_trig_b(optg, d, rpv_)  ! Gets the random probability vector via the trigonometric method

allocate( rperm(1:d) )  ! Allocates memory for the random permutation of {1,2,...,d}
call rand_perm(optg, d, rperm)  ! Gets the random permutation 

forall( j = 1:d )  ! Shuffles the components of the random probability vector to avoid biasing
  rpv(j) = rpv_(rperm(j))
end forall

deallocate( rpv_, rperm ) ! Deallocates the memory used by these variables

end
!------------------------------------------------------------------------------------------------------------------------------------
subroutine rpv_iid(optg, d, rpv) ! This subroutine generates and random probability vector of dimension d using the iid method
! Ref: Maziero, J. (2015). Generating pseudo-random discrete probability distributions. Braz. J. Phys. 45, 377.
implicit none
integer :: d  ! Dimension of the random probability vector
real(8) :: rpv(1:d)  ! Random probability vector
real(8), allocatable :: rn(:)  ! Vector of random numbers
character(10), dimension(5) :: optg  ! Options for the generators

allocate( rn(1:d) ) ;   call rng(optg, d, rn)  ! Allocate memory for and get the uniformly distributed random numbers
rpv = rn/sum(rn)   ! Divides each of the d independent random numbers by their sum
deallocate( rn )  ! Frees the memory used with this variable

end
!------------------------------------------------------------------------------------------------------------------------------------
subroutine rpv_devroye(optg, d, rpv) ! Generates a random probability vector of dimension d using Devroye's method
! Ref: Devroye, L. (1986). Non-Uniform Random Variate Generation. New York: Springer.
implicit none
integer :: d  ! Dimension of the random probability vector
real(8) :: rpv(1:d)  ! Random probability vector
real(8), allocatable :: ern(:)  ! Vector of exponentially distributed random numbers
character(10), dimension(5) :: optg  ! Options for the generators

allocate( ern(1:d) ) ;  call rng_exp(optg, d, ern)  ! Allocate memory for and get the exponentially distributed random numbers
rpv = ern/sum(ern)  ! This line implements Devroye's method
deallocate( ern )  ! Frees the memory used by this variable

end
!------------------------------------------------------------------------------------------------------------------------------------
subroutine rpv_kraemer(optg, d, rpv) ! Generates a random probability vector of dimension d using Kraemer's method 
! (in contrast to the others, this method uses sorting)
! Ref: Kraemer, H. (1999). Post on MathForum on December 20. Topic: Sampling uniformly from the n-simplex.
!use qsort_mod  ! Uses the modules which implements the Quicksort algorithm
implicit none
integer :: d  ! Dimension of the random probability vector
real(8) :: rpv(1:d)  ! Random probability vector 
real(8), allocatable :: rn(:)  ! Vector of random numbers
integer :: j  ! Auxiliary variable for counters
character(10), dimension(5) :: optg  ! Options for the generators

allocate( rn(1:d-1) ) ;   call rng(optg, d-1, rn)  ! Allocates memory for and gets the vector of random numbers
call qsort(rn, d-1)  ! Sort the random numbers in non-decreasing order
rpv(1) = rn(1) ;  rpv(d) = 1.d0 - rn(d-1)  ! These two lines implement the Kraemer's method
forall ( j = 2:(d-1) ) rpv(j) = rn(j) - rn(j-1)

deallocate( rn ) ! Frees the memory used by this variable

end
!###################################################################################################################################
!                                              Calling subroutine for the RPVG
!###################################################################################################################################
subroutine rpvg(optg, d, rpv) ! Calls the choosed random probability vector generator
implicit none
integer :: d  ! Dimension of the vector
real(8) :: rpv(1:d)  ! The random probability vector
character(10), dimension(5) :: optg  ! Options for the generators

     if ( (optg(2) == "std") .or. (optg(2) == "zhsl") ) then ;   call rpv_zhsl(optg, d, rpv)
else if ( optg(2) == "norm" ) then ;   call rpv_norm(optg, d, rpv)
else if ( optg(2) == "trig" ) then ;   call rpv_trig(optg, d, rpv)
else if ( optg(2) == "iid" ) then ;   call rpv_iid(optg, d, rpv)
else if ( optg(2) == "devroye" ) then ;   call rpv_devroye(optg, d, rpv)
else if ( optg(2) == "kraemer" ) then ;   call rpv_kraemer(optg, d, rpv)
     endif

end
!###################################################################################################################################
!                                                       Tests - RPVG
!###################################################################################################################################
subroutine rpvg_tests(optg, d, ns, ni) ! Some simple tests for the random probability vector generators
implicit none
integer, parameter :: d != 3  ! Dimension of the random probability vectors
integer, parameter :: ns != 10**6  ! No. of samples for the averages and probability distributions
integer, parameter :: ni != 100  ! No. of intervals for the domain of the RPV components
real(8) :: delta != 1.d0/dble(ni) ! Step for the domain of the RPV components
real(8) :: rpv(1:d), avg_rpv(1:d)  ! Random probability vector and its average
real(8) :: t1, t2  ! Times
integer :: j, k, l  ! Auxiliary variables for counters
integer :: ct(1:ni,1:d) = 0  ! Counter for the probability densities
character(10), dimension(5) :: optg  ! Options for the generators

call cpu_time(t1)

delta = 1.d0/dble(ni)

write(*,*) "## Performing some tests for the random probability vector generators"

! Sets the methods to be used by the generators
!optg(1) = RNG ;   optg(2) = RPVG ;  optg(3) =  RUG ;    optg(4) = RSVG ;    optg(5) = RDMG
!optg(1) = 'std' ;  optg(2) = 'std' ; optg(3) = 'std' ;   optg(4) = 'std' ;   optg(5) = 'std'


write(*,*) "# Generating a sampling for 2d scatter plots"
open(unit = 11, file = 'rpv_2d.dat', status = 'unknown') ! Gnuplot commands to see the results: plot 'rpv_2d.dat'
do j = 1, 5*10**3 ;   call rpvg(optg, d, rpv) ;  write(11,*) (rpv(k) , k=1,2) ;   enddo
!do j = 1, 5*10**3 ;   call rpvg(optg, d, rpv) ;  write(11,*) ((1.d0 - rpv(k)) , k=1,2) ; !enddo  ! To put the points above the 'diagonal'
close(11)


write(*,*) "# Computing the probability distribution for the components of the RPVs"
open(unit = 12, file = 'rpv_pd.dat', status = 'unknown') ! Gnuplot commands to see the results: plot 'rpv_pd.dat'
avg_rpv = 0.d0
do j = 1, ns
  call rpvg(optg, d, rpv) ;   avg_rpv = avg_rpv + rpv
  do k = 1, d ;   if (rpv(k) == 1.d0) rpv(k) = 1.d0 - 1.d-10
    do l = 1, ni ;   if ( (rpv(k) >= (dble(l)-1.d0)*delta) .and. (rpv(k) < dble(l)*delta)) ct(l,k) = ct(l,k) + 1 ;   enddo
  enddo
enddo
! Writes  on the screen the average of the (up to) first 5 components of the RPV
if ( d <= 5 ) write(*,*) 'avg_rpv = ', avg_rpv/dble(ns)
do l = 1, ni
  write(12,*) (dble(l)-0.5)*delta, dble(ct(l,1))/dble(ns)  ! Writes the probability density for the first component of the RPV on a file
  !write(12,*) (dble(l)-0.5)*delta, (dble(ct(l,k))/dble(ns), k=1,d)  ! Writes the probability density for all components of the RPV on a file
enddo
close(12)

call cpu_time(t2) ;   write(*,*) 'time=',t2-t1,'seconds'  ! Writes the time taken by this subroutine

end
!###################################################################################################################################
