#include <stdio.h>
#include <stdlib.h>

double * allocVectorOnHost(int const SZ);
void printMatrix(double *v, int const M, int const N);
double * matrixMulHost(double *M, double *N, int const SZ);
double calcDiff(double *P, double *Q, int const SZ);
