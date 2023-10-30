!-----------------------------------------------------------------------------------------------------------------------------------
subroutine rand_perm(optg, d, rperm)  ! Returns a random permutation of {1,2,...,d-1,d}
! Used e.g. in the normalization and trigonometric methods for rpvg
implicit none
integer :: d  ! Dimension of the random permutation vector
integer :: rperm(1:d)  ! Random permutation vector
integer :: j  ! Counter for do (auxiliary variable)
integer, allocatable :: counter(:)  ! Counter for the no. of times a component of rand_perm is randomly choosed (auxiliary variable)
integer :: ind  ! Identify the component of rand_perm choosed (auxiliary variable)
real(8) :: rn(1)  ! Random numbers
character(10), dimension(5) :: optg  ! Options for the generators

allocate( counter(1:d) )  ! Allocate memory for the counter for the componets of rand_perm
 counter = 0
 
do j = 1, d
  do
    call rng(optg, 1, rn)  ! Returns one random number using the method choosed via option op_rng 
    if ( rn(1) <= 0.d0 ) rn(1) = 1.d-10  ! These two lines are a precaution for the determination of ind (avoid rn = 0 and rn = 1) 
    if ( rn(1) >= 1.d0 ) rn(1) = 1.d0 - 1.d-10
    ind = aint( dble(d)*rn(1) + 1.d0 )  ! aint(x) returns the largest integer smaller than x (i.e., ind is in [1,d])
    counter(ind) = counter(ind) + 1  ! No. of times the value ind appeared in rperm
    if ( counter(ind) >= 2 ) cycle  ! cycle returns the pointer to the top of the do
    if ( counter(ind) == 1 ) then
      rperm(j) = ind ;   exit  ! exit the do going to the next j, i.e., the next component of rand_perm
    endif
  enddo
enddo
 
deallocate( counter )  ! Deallocate the memory used with the counter

end
!-----------------------------------------------------------------------------------------------------------------------------------