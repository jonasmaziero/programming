 reset
 set terminal postscript enhanced 'Helvetica' 24
 set output 'planck.eps'
 plot 'planck.dat' u 1:2 w l, '' u 1:3 w l, '' u 1:4 w l
