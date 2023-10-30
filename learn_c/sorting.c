//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
//#include <time.h>
#include <limits.h>
//------------------------------------------------------------------------------
int main()
{
  int d = 10;
  int A[10] = {1,25,5,2,30,8,50,43,6,77};
  int j;
  for (j = 0; j < d; j++) {
    printf("%i \t", A[j]);
  }
  printf("\n");
  //void selectionSortI();  selectionSortI(&d, A);
  void selectionSortIopt();  selectionSortIopt(&d, A);
  for (j = 0; j < d; j++) {
    printf("%i \t", A[j]);
  }
  printf("\n");
  return 0;
}
//------------------------------------------------------------------------------
void selectionSortI(int *d, int *A)
{
  int maxi = INT_MAX;
  int *B;
  size_t sz = (*d)*sizeof(int);
  B = (int *)malloc(sz);
  int j, k, kmin;
  for (j = 0; j < (*d); j++) {
    kmin = 0;
    for (k = 1; k < (*d); k++) {
      if (*(A+kmin) > *(A+k)) { kmin = k; }
    }
    *(B+j) = *(A+kmin);
    *(A+kmin) = maxi;
  }
  for (j = 0; j < (*d); j++) {
    *(A+j) = *(B+j);
  }
  free(B);
}
//------------------------------------------------------------------------------
void selectionSortIopt(int *d, int *A)  // CC ~ O(d^2)
{
  int j, k, jMin, temp;
  for (j = 0; j < (*d)-1; j++) {
    jMin = j;
    for (k = j+1; k < (*d); k++) {
      if (*(A+jMin) > *(A+k)) { jMin = k; }
    }
    temp = *(A+j);
    *(A+j) = *(A+jMin);
    *(A+jMin) = temp;
  }
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
