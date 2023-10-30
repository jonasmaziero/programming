//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
//------------------------------------------------------------------------------
int main() {
  //void forCycle();  forCycle();
  void While();  While();
  void doWhile();  doWhile();
  return 0;
}
//------------------------------------------------------------------------------
void forCycle() {
  int j, k, N = 2, M = 3;
  for (j = 0; j < N; j++) {
    printf("j = %i \t", j);
    for (k = 0; k < M; k++) {
      if (k > 1) continue;
      printf("k = %i \t", k);
    }
    printf("\n");
  }
}
//------------------------------------------------------------------------------
void While() {
  int idx[5] = {0,0,0,0,1};
  int j = -1;
  while (idx[j+1] == 0) {
    j += 1;
  }
  printf("%i \n", j);
}
//------------------------------------------------------------------------------
void doWhile() {
  int idx[5] = {0,0,0,0,1};
  int j = -1;
  do {
    j += 1;
  } while(idx[j+1] == 0);
  printf("%i \n", j);
}
//------------------------------------------------------------------------------

// variable declaration
//int j, jm = 4;
//int k, km = 5;



// nested loops
/*
int k, km = 5;
// one does not need to declare variables at the top of the function body
for(j = 0; j < jm; j++){
  printf("j = %i \n", j);
  for(k = 0; k < km; k++){
    printf("k = %i \n", k);
  }
}
*/

/*// cycling a loop
for(j = 0; j < jm; j++){
  if(j == 2)continue;
  printf("j = %i \n", j);
}
*/

// cycling nested loops
// it seems it's not possible to cycle a particular for (in contrast to Fortran)
/*int j, l, m, o, da = 2, dc = 3;
for (j = 0; j < da; j++) {
for (l = 0; l < dc; l++) {
  for (m = 0; m < da; m++) {
    for (o = 0; o < dc; o++) {
      printf("%i %i %i %i %i %i \n", j, l, m, o, j*dc+l, m*dc+o);
    }
  }
}
}*/

// single while loop
/*
j = 2;
while(j < jm){j++;  printf("j = %i \n", j);}
// one can write several statements in the same line
*/

// the do while command
/*
j = 2;
do{
  j++;  printf("j = %i \n", j);
} while(j < jm);
*/
//------------------------------------------------------------------------------
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//------------------------------------------------------------------------------
/*int main()
{
  // factorial
  //int n, ft, ftR, factorial(), factorialR();
  for (n = 0; n < 20; n++) {
    ft = factorial(n); ftR = factorialR(n);
    printf("n = %i, n! = %i, n! = %i \n", n, ft, ftR);
  }// ok
  //factorialR(5);

  // fibonacci
  int n, F, fibonacci(), FR, fibonacciR();
  F = fibonacci(40);  printf("%i \n", F);
  FR = fibonacciR(40);  printf("%i \n", FR);
  for (n = 0; n <= 12; n++) {
    F = fibonacci(n);  FR = fibonacciR(n); printf("%i, %i \n", F, FR);
  } // ok!
  return 0;
}
*/
//------------------------------------------------------------------------------
int factorial(int n) {
  if (n == 0 || n == 1) { return 1; }
  int j = 0;
  int ft = 1;
  while (j < n) {
    j += 1;
    ft *= j;
  }
  return ft;
}
//------------------------------------------------------------------------------
int factorialR(int n) {
  //printf("I'm calculating %i! \n", n);
  if (n == 0) { return 1; }
  return n*factorialR(n-1);
  //int fct = n*factorialR(n-1);
  //printf("I'm done calculating. Result = %i \n", fct);
  //return fct;
}
//------------------------------------------------------------------------------
int fibonacci(n) {
  if (n < 2) { return n; }
  int j, F, F1, F2;
  F1 = 0;
  F2 = 1;
  for (j = 2; j <= n; j++)
  {
    F = F1 + F2;
    F1 = F2;
    F2 = F;
  }
  return F;
}
//------------------------------------------------------------------------------
int fibonacciR(n) { // bad timing because of multiple evals of same value
  if (n < 2) { return n; }
  return fibonacciR(n-2)+fibonacciR(n-1);
}
//------------------------------------------------------------------------------
