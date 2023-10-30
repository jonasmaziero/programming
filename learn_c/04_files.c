#include <stdio.h>
#include <complex.h>
#include <math.h>

int main(){
  int j, k;
  int d = 4;
  double rhoR[4][4], rhoI[4][4];
  double _Complex rho[4][4];
  FILE *flR = fopen("rhoR", "r");
  FILE *flI = fopen("rhoI", "r");
  for(j = 0; j < 4; j++){
    for(k = 0; k < 4; k++){
      fscanf(flR,"%lf \t", *(rhoR+j*4+k));
      fscanf(flI,"%lf \t", *(rhoI+j*4+k));
      *(rho+j*4+k) = *(rhoR+j*4+k) + (*(rhoI+j*4+k))*I
    }
    fscanf(fl,"\n");
  }
  fclose(fl);
  printf("%lf \n", concurrence(&d, rho))
  /*for(j = 0; j < 4; j++){
    for(k = 0; k < 4; k++){
      printf("%lf \t", *(rhoR+j*4+k));
    }
    printf("\n");
  }*/
  return 0;
}
/*int j;
FILE *fp;  char output[] = "output.txt";  fp = fopen(output, "w+");
double genrand64_real1(), a, b;
int ns = pow(10,4);  //printf("%d \n", ns);
for (j = 0; j < ns; j++){
  a = genrand64_real1();  b = genrand64_real1();
  fprintf(fp, "%f \t %f \n", a, b);  //printf("%f \t %f \n", a, b);
}
for (j = 0; j < 10; j++){
  fscanf(fp, "%f \t %f \n", genrand64_real1(), genrand64_real1());
  printf("%f \t %f \n", genrand64_real1(), genrand64_real1());
}
fclose(fp);*/
