subroutine read_qrng()
  implicit none
  integer :: j, rn(9492100)
  real(4) :: rnr(9492100)
  open(unit=13, file = "/home/jonasmaziero/qRNG/QN.dat")
  open(unit=14, file = "/home/jonasmaziero/qRNG/QNr.dat")
  do j = 1, 9492100
    read(13,*) rn(j)
    rnr(j) = rn(j)/65535.d0
    write(14,*) rnr(j)
  enddo
end
