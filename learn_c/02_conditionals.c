//------------------------------------------------------------------------------
#include <stdio.h>  // compila com:
#include <stdlib.h>   // gcc 02_conditionals.c -lm
#include <math.h>   // O -lm é para ter acesso às funções de math.h
#include <complex.h>
//------------------------------------------------------------------------------
int main(){
  void bascara(); bascara();
}
//------------------------------------------------------------------------------
void bascara(){
 double a, b, c, Delta;
 double _Complex x1, x2;

 printf("Forneça os coeficientes de ax^2+bx+c=0 \n");
 printf("Digite a \n");
 scanf("%lf",&a); // scanf receives the pointer to the variable we want to read
 if (a == 0) {
   printf("%s \n","a não pode ser nulo");
   return;
 }
 printf("Digite b \n");
 scanf("%lf",&b);
 printf("Digite c \n");
 scanf("%lf",&c);
 Delta = pow(b,2.0) - 4.0*a*c;  // pow(x,y)=x**y
 printf("Delta = %lf \n", Delta);
 printf("%s \n","Raízes");
 if (Delta == 0) {
   x1 = -b/(2.0*a);
   x2 = x1;
   printf("x1 = %lf, x2 = %lf \n", creal(x1), creal(x2));
 } else if (Delta > 0) {
   x1 = (-b + sqrt(Delta))/(2.0*a);
   x2 = (-b - sqrt(Delta))/(2.0*a);
   printf("x1 = %lf, x2 = %lf \n", creal(x1), creal(x2));
 } else if (Delta < 0.0) {
   x1 = (-b/(2.0*a)) + I*(sqrt(abs(Delta))/(2.0*a));
   x2 = (-b/(2.0*a)) - I*(sqrt(abs(Delta))/(2.0*a));
   printf("Re(x1) = %lf, Im(x1) = %lf \n", creal(x1), cimag(x1));
   printf("Re(x2) = %lf, Im(x2) = %lf \n", creal(x2), cimag(x2));
 }

}
//------------------------------------------------------------------------------
