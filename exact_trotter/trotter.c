#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

unsigned int N;
unsigned long int D;

int RX(int k, double tau, double complex *psi, double complex *tmp){

  unsigned long int mask = 1<<k;

  for(unsigned long int j=0; j<D; j++){
    unsigned long int l = j^mask;
    tmp[j] = cos(tau/2)*psi[j] - 1*I*sin(tau/2)*psi[l];
  }

  for(unsigned long int j=0; j<D; j++){
    psi[j] = tmp[j];
  }

}

int CRZ(int k1, int k2, double tau, double complex *psi, double complex *tmp){

  unsigned long int m1 = 1<<k1;
  unsigned long int m2 = 1<<k2;

  for(unsigned long int j=0; j<D; j++){

    unsigned long int b1 = (j&m1)>>k1;
    unsigned long int b2 = (j&m2)>>k2;
    int sign = 1;

    if(b1!=b2){
      sign = -1;
    }

    psi[j] = (cos(tau/2) - sign*1*I*sin(tau/2))*psi[j];
  }

}

int trotter_step(double tau, double h, double complex *psi, double complex *tmp){

  for(int k=0; k<(N-1); k++){
    CRZ(k,k+1,-2*tau,psi,tmp);
  }
  CRZ(N-1,0,-2*tau,psi,tmp);

  for(int k=0; k<N; k++){
    RX(k,-2*tau*h,psi,tmp);
  }

}

double measure_sx(double complex *psi){

  double sx = 0;

  for(unsigned long int j=0; j<D; j++){
    sx += creal(conj(psi[j])*psi[j^1]);
  }

  return sx;
}

void print_psi(double complex *psi){

  for(unsigned long int j=0; j<D; j++){
    printf("%i %.10f %.10f\n",j,creal(psi[j]),cimag(psi[j]));
  }

}

int main(int argc, char const *argv[]) {

  // init global variables
  N = 25;
  D = 1<<N;

  // alloc memory for wavefunction
  double complex *psi = malloc(D*sizeof(double complex));
  double complex *tmp = malloc(D*sizeof(double complex));

  // init wavefunction
  for(unsigned long int j=0; j<D; j++){
    psi[j] = 1.0/sqrt(D);
  }

  // Trotter evolution
  double tau = .005;
  int steps = 200;

  double h=2;

  printf("# t sx\n");
  for(int s=0; s<steps; s++){
    double sx = measure_sx(psi);
    printf("%f %f\n", s*tau, sx);
    trotter_step(tau,h,psi,tmp);
  }

  // free memory for wavefunction
  free(psi);
  free(tmp);

  return 0;
}
