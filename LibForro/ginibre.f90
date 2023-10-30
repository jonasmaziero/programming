!-----------------------------------------------------------------------------------------------------------------------------------
subroutine ginibre(optg, m, n, G)  ! Returns a m x n complex matrix from the Ginibre ensemble
implicit none
integer :: m, n  ! No. of rows and columns of G
complex(8) :: G(1:m,1:n)  ! The Ginibre matrix
real(8), allocatable :: grn(:)  ! Vector of gaussianily distributed random numbers
integer :: j, k  ! Auxiliary variables for counters
character(10), dimension(5) :: optg  ! Options for the generators

allocate( grn(1:2*m) )
do j = 1, n
  call rng_gauss(optg, 2*m, grn) ;   forall( k = 1:m ) G(k,j) = grn(k) + (0.d0,1.d0)*grn(m+k)  ! Generates G column by column
enddo
deallocate( grn )

end
!-----------------------------------------------------------------------------------------------------------------------------------