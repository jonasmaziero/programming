//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
//------------------------------------------------------------------------------
/* stack memory (SM)
* Stack memory is allocated during runtime
* Last called function are on the top of the stack memory
* If too much functions are called sequentilly and the SM is totally occupied,
* we'll have a STACK-OVERFLOW
*/
//------------------------------------------------------------------------------
// heap memory (HM) is dynamically allocated, and not size-limited
//------------------------------------------------------------------------------
void memAlloc(){
  //void heapMalloc();  heapMalloc();
  //void heapCalloc();  heapCalloc();
  //void heapRealloc();  heapRealloc();
  void realloc2Dc();  realloc2Dc();
}
//------------------------------------------------------------------------------
void heapMalloc(){
  int a = 10; // goes on stack
  int *p;  // p also goes on stack
  p = (int *)malloc(sizeof(int));
  // the size sent to malloc must be an unsigned integer variable (size_t)
  // malloc returns a void pointer. So, (int *) type cast it to point to integer
  // The only way to use memory on heap is through reference
  *p = a;
  printf("%p %i \n", p, *p);
  free(p);  // free the memory previously allocated for p

  int N = 10;
  size_t sz = N*sizeof(int);  // size_t is an unsigned integer variable
  p = (int *)malloc(sz);
  int j;
  for (j = 0; j < N; j++){
    *(p+j) = j;  // changing the values stored in the memory
    //printf("%p %i \n", p+j, *(p+j));  // p[j] = *(p+j)
    printf("%p %i \n", p+j, p[j]); // accessing the values as with arrays
  }
  free(p);  // always remember to free the memory allocated
}
//------------------------------------------------------------------------------
void heapCalloc(){
  int j;
  size_t N = 10;
  size_t sz = sizeof(int);
  int *p;
  p = (int *)calloc(N,sz);  // calloc initializes the memory values to zero
                            // both arguments of malloc are of type size_t
                            // 1st arg is the No. of elements & 2nd is the type
  for (j = 0; j < N; j++){
    printf("%p %i \n", p+j, *(p+j));
  }
  printf("\n");
  for (j = 0; j < N; j++){
    p[j] = j;
    printf("%p %i \n", p+j, p[j]);
  }
  free(p);
}
//------------------------------------------------------------------------------
void heapRealloc(){
  int j;
  int N = 5, M = 3, P = 4;
  size_t szi = sizeof(int);
  int *p;
  p = (int *)malloc(N*szi);
  for (j = 0; j < N; j++){
    *(p+j) = j;
    printf("%p %i \n", p+j, *(p+j));
  }

  p = (int *)realloc(p, M*szi);
  printf("after 1st realloc \n");
  for (j = 0; j < M; j++){
    *(p+j) = j;
    printf("%p %i \n", p+j, *(p+j));
  }

  p = (int *)realloc(p, P*szi);
  printf("after 2nd realloc \n");
  for (j = 0; j < P; j++){
    *(p+j) = j;
    printf("%p %i \n", p+j, *(p+j));
  }
  free(p);

  // realloc(NULL, size_t sz) == malloc(size_t sz)
  // realloc(p, 0) == free(p)
}
//------------------------------------------------------------------------------
void realloc2Dc(){
  int j, k;
  int xd = 2, yd = 3;
  size_t sz = xd*yd*sizeof(double _Complex);
  //printf("%i \n", sz);
  double _Complex *A = (double _Complex *)malloc(sz);
  //printf("%p \n", A);
  for (j = 0; j < xd; j++) {
    for (k = 0; k < yd; k++) {
      *(A+k+j*yd) = j + I*k;
       //printf("%lf + I*%lf\n", creal(*(A+k+j*yd)), cimag(*(A+k+j*yd)));
    }
  }
  void array2DisplayC();  array2DisplayC(&xd, &yd, A);
}
//------------------------------------------------------------------------------
void array2DisplayC(int *xd, int *yd, double _Complex *A){
  // I sent only the first memory address of the "2D" array
  int j, k;
  printf("Real part of the array) \n");
  for (j = 0; j < (*xd); j++) {
    for (k = 0; k < (*yd); k++) {
      printf("%lf \t", creal(*(A+k+j*(*yd))));//best dereferencing for 2D array
    }
    printf("\n");
  }
  printf("Imaginary part of the array \n");
  for (j = 0; j < (*xd); j++) {
    for (k = 0; k < (*yd); k++) {
      printf("%lf \t", cimag(*(A+k+j*(*yd))));
    }
    printf("\n");
  }
}
//------------------------------------------------------------------------------
