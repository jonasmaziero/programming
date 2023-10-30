//------------------------------------------------------------------------------
#include <stdio.h>
//#include <stdlib.h>
#include <time.h>
//------------------------------------------------------------------------------
int main()
{
  // dt is the cpu time, not physical time
  int n = 42, F, fibonacci(), FR, fibonacciR();
  clock_t t1, t2;
  double dt;
  t1 = clock();
  F = fibonacci(n);
  t2 = clock();
  printf("Fib(%i) = %i \n", n, F);
  dt = (double)(t2-t1)/CLOCKS_PER_SEC;
  printf("dt = %lf \n", dt);

  t1 = clock();
  FR = fibonacciR(n);
  t2 = clock();
  printf("Fib(%i) = %i \n", n, FR);
  dt = (double)(t2-t1)/CLOCKS_PER_SEC;
  printf("dtR = %lf \n", dt);

  return 0;
}
//------------------------------------------------------------------------------
