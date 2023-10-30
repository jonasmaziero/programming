//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
//------------------------------------------------------------------------------
int main() {
  int dec = 6;
  int n = 5;
  int *bin;
  size_t szi = sizeof(int);
  bin = (int *)calloc(n, szi);
  //size_t szi = n*sizeof(int);
  //bin = (int *)malloc(szi);
  int j;
  /**(bin+0) = 1;  *(bin+1) = 0;  *(bin+2) = 1;
  int bin2dec(); dec = bin2dec(&n, bin);
  printf("%i \n", dec);
  return 0;*/

  void dec2bin(); dec2bin(&dec, &n, bin);
  for (j = 0; j < n; j++){
    printf("%i", *(bin+j));
  }
  printf("\n");

  return 0;
}
//------------------------------------------------------------------------------
int bin2dec(int *n, int *bin) {
  // n is the number of digits of the binary number
  int j;
  int dec = 0;
  for (j = 0; j < (*n); j++) {
    dec += (*(bin+j))*pow(2,((*n)-1-j));
  }
  return dec;
}
//------------------------------------------------------------------------------
void dec2bin(int *dec, int *n, int *bin) {
  int j = (*n)-1;
  *(bin+j) = (*dec)%2;
  int deca = (*dec)/2;
  while (deca != 0) {
    j -= 1;
    *(bin+j) = deca%2;
    deca /= 2;
  }
}
//------------------------------------------------------------------------------
