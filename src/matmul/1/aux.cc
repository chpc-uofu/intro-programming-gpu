#include <math.h>
#include <stdio.h>
#include <aux.h>

// Allocate Vectors on Host (Size SZ)
double * allocVectorOnHost(int const SZ)
{
    double *v;
    v=(double*)malloc(sizeof(double)*SZ);       
    if(v==NULL)
    {
       printf("  Error allocating vector on HOST\n");
       exit(-1);
    }
    return v;     
}


// Matrix multiplication of P=MxN (each SZ*SZ matrices)
//   return P (as vector)
double * matrixMulHost(double *M, double *N, int const SZ){

    double *P;
    int const SZ2=SZ*SZ;

    P=allocVectorOnHost(SZ2);
    for(int i=0; i<SZ; i++){
        for(int j=0; j<SZ; j++){
            for(int k=0; k<SZ; k++)
                P[i*SZ+j] += M[i*SZ+k]*N[k*SZ+j];
        }
    }
    return P;
}


// Calculate the Frob. Norm of (P-Q) 
//    P & Q: SZ x SZ matrices
double calcDiff(double *P, double *Q, const int SZ){

    int const SZ2=SZ*SZ;
    double dx, norm=0.0;

    for(int i=0; i<SZ*SZ; i++){ 
        dx = P[i] - Q[i];
        norm += dx*dx;    
    }
    return sqrt(norm);
}


// Print Matrix v (MxN matrix)
void printMatrix(double *v, int const M, int const N)
{
    int k=0;
    for(int i=0; i<M; i++)
    {
        printf("    ");
        for(int j=0; j<N; j++)
        {
            printf("%8.1lf   ", v[k]);
            k++;
        }
        printf("\n");
    }
    return;
}
