//------------------------------------------------------------------------------
#include <stdio.h>
#include <complex.h>
//------------------------------------------------------------------------------
void pointers() {
  // for each data type we have a pointer type, because of the dereferencing

  //void pointersSimple();  pointersSimple();
  //void pointer2pointer();  pointer2pointer();
  //void callByRef();  callByRef();
  //void callSumOelem();  callSumOelem();
  //void pointers1D();  pointers1D();
  //void pointers2D();  pointers2D();
  void pointers3D();  pointers3D();
}
//------------------------------------------------------------------------------
void pointer2pointer(){
  int a = 0;
  int *pa = &a;
  int **ppa = &pa;  // declare a pointer to pointer
  int ***pppa = &ppa;  // declare a pointer to pointer to pointer
  printf("%p \n", &a);
  printf("%p \n", pa);
  printf("%p \n", *ppa);
  printf("%p \n", *(*pppa));
  /*printf("%p \n", &pa);
  printf("%p \n", ppa);
  printf("%p \n", *pppa);*/
}
//------------------------------------------------------------------------------
void callByRef(){
  int a;
  a = 10;
  void increment1();  increment1(&a);
  printf("%p \n", &a);
  printf("%i \n", a);
  void increment2();  increment2(a);
}
void increment1(int *a){  // we send the address of a
  *a += 1;  // change the value of a via its pointer
  printf("%p \n", a);
  printf("%i \n", *a);
}
void increment2(int a){ // the a is local to this function
  a += 1;
  printf("%p \n", &a);
  printf("%i \n", a);
} // when the function ends its local variables disappears
//------------------------------------------------------------------------------
void pointersSimple(){
  //

  printf("integer variables \n");
  int a;  // declares an integer variable
  int *pa;  // declares a pointer to an integer variable
  pa = &a;  // the pointer receives the address of the variable a
  a = 10;
  printf("address of the pointer = %p \n", &pa);
  printf("%p \n", pa);
  printf("%p \n", &a);
  printf("%i \n", *pa);
  printf("%i \n", a);

  printf("double variables \n");
  double b;
  double *pb;
  pb = &b;
  b = 1.0;
  printf("address of the pointer = %p \n", &pb);
  printf("%p \n", pb);
  printf("%p \n", &b);
  printf("%f \n", *pb);
  printf("%f \n", b);
  *pb = 2.0;  // changing the value of a variable via its pointer
  printf("%f \n", *pb);
  printf("%f \n", b);

  printf("complex variables \n");
  double _Complex c;
  double _Complex *pc;
  pc = &c;
  c = a + I*b;
  printf("address of the pointer = %p \n", &pc);
  printf("%p \n", pc);
  printf("%p \n", &c);
  printf("%f + I*%f \n", creal(*pc), cimag(*pc));
  printf("%f + I*%f \n", creal(c), cimag(c));

}
//------------------------------------------------------------------------------
void callSumOelem(){// C only send addresses of arrays to functions
  int A[5] = {0,1,2,3,4};
  int size = sizeof(A)/sizeof(A[0]);
  printf("size of the array in the calling function = %i \n", size);
  int sum, sumOelem();
  sum = sumOelem(A, &size);
  printf("%i \n", sum);
}
int sumOelem(int *A, int *size){ // the size of the array must be given as input
  //int sumOelem(int A[], int *size){
  int si = sizeof(A)/sizeof(A[0]);
  printf("size of the array in the called function = %i \n", si);
  int j, sum = 0;
  for (j = 0; j < (*size); j++) {
    sum += A[j];
  }
  return sum;
}
//------------------------------------------------------------------------------
void pointers1D(){
  int A[5] = {0,1,2,3,4};  // declaring and initializing a 1D array
  int *pA;
  pA = A; // the pointer pA receives the address (of start point) of A
  //A = pA;  // we cannot change the address of A in this way
  printf("%p \n", A);  // 3 ways to get the same memory address
  printf("%p \n", pA);
  printf("%p \n", &A[0]);
  printf("%p \n", pA+1);  // using pointer arithmetic
  printf("%p \n", &A[1]);
  printf("%i \n", *pA);
  printf("%i \n", A[0]);
  printf("%i \n", *(pA+1));  // using pointer arithmetic
  printf("%i \n", A[1]);
  int j;
  int B[5] = {5,6,7,8,9};
  int *pB;
  pB = B;
  //pB = pA;  // pB would point to A
  //B = A;  // not allowed
  for (j = 0; j < 5; j++){
    //B[j] = A[j];  // is one way to pass values from one array to another
    *(pB+j) = *(pA+j);  // atribute the value of A to B via their pointers
    printf("B[j] = %i \n", *(pB+j));
    printf("B[j] = %i \n", B[j]);
  }
}
//------------------------------------------------------------------------------
void pointers2D(){
  int A[2][3] = {{0,1,2},{3,4,5}};  // 2D array = array of 1D arrays
  //int *pA = A; // gives an error, A is a pointer to a nc=3 integer array A[0]
  //int (*pA)[3] = A; // the right way to define the pointer ...
  printf("%p \n", A);
  printf("%i \n", *(*A+1));  // A is a pointer to A[0], which points to A[0][0]
  printf("%p \n", A[0]); // A[0] is a pointer to A[0][0]
  printf("%p \n", A[1]); // A[1] is a pointer to A[1][0]
  /*printf("&A[0][0] = %p \n", &A[0][0]);
  printf("&A[0][1] = %p \n", &A[0][1]);
  printf("&A[0][2] = %p \n", &A[0][2]);
  printf("&A[1][0] = %p \n", &A[1][0]);
  printf("&A[1][1] = %p \n", &A[1][1]);
  printf("&A[1][2] = %p \n", &A[1][2]);*/
  int ne = sizeof(A)/sizeof(A[0][0]); // No. of elements of A
  int nc = sizeof(A[0])/sizeof(A[0][0]);  // No. of columns of A
  int nr = ne/nc;  // No. of rows of A
  void array2DisplayI();  array2DisplayI(&nr, &nc, A);
}
//------------------------------------------------------------------------------
void array2DisplayI(int *nr, int *nc, int A[][*nc]){
  int j, k;
  for (j = 0; j < (*nr); j++) {
    for (k = 0; k < (*nc); k++) {
        //A[j][k] = *(*A + k + j*(*nc)) = *(*(A+j) + k)
        printf("%i \t", *(*A + k + j*(*nc)));
        //printf("%i \t", *(*(A+j) + k));
        //printf("%i \t", A[j][k]);
    }
    printf("\n");
  }
}
//------------------------------------------------------------------------------
void pointers3D(){
  int C[4][3][2] = {{{0,1},{2,3},{4,5}},{{6,7},{8,9},{10,11}},
                    {{12,13},{14,15},{16,17}},{{18,19},{20,21},{22,23}}};
  printf("sizeof(C[0][0][0]) = %lu bytes \n", sizeof(C[0][0][0]));
  printf("sizeof(C[0][0]) = %lu bytes \n", sizeof(C[0][0]));
  printf("sizeof(C[0]) = %lu bytes \n", sizeof(C[0]));
  printf("sizeof(C) = %lu bytes \n", sizeof(C));
  int zd = sizeof(C[0][0])/sizeof(C[0][0][0]);
  int yd = sizeof(C[0])/sizeof(C[0][0]);
  int xd = sizeof(C)/sizeof(C[0]);
  printf("xd = %i, yd = %i, zd = %i \n", xd, yd, zd);
  void array3DisplayI();  array3DisplayI(&xd, &yd, &zd, C);
  printf("C = %p, *C = %p, C[0] = %p,  &C[0][0] = %p \n", C, *C, C[0], C[0][0]);
  printf("*(C[1]+1) = %p \n", *(C[1]+1));
  printf("*(C[0][1]+1) = %p, C[0][1][1] = %i \n", *(C[1]+1), C[0][1][1]);
}
//------------------------------------------------------------------------------
void array3DisplayI(int *xd, int *yd, int *zd, int A[][*yd][*zd]){
  int j, k, l;
  for (j = 0; j < (*xd); j++) {
    printf("A[j][k][l] for j = %i \n", j);
    for (k = 0; k < (*yd); k++) {
      for (l = 0; l < (*zd); l++) {
         //printf("%i \t", A[j][k][l]);
         printf("%i \t", *(*(*(A+j)+k)+l));
      }
      printf("\n");
    }
  }
}
//------------------------------------------------------------------------------
