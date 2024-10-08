#include <mul.h>

__global__ void MatrixMulKernel1(double *M_d, double *N_d, 
                                 double *P_d, int const SZ)
{
    double Pval=0;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    for(int k=0; k<SZ; k++)
        Pval+=M_d[ty*SZ +k]*N_d[tx+k*SZ];
    P_d[tx+ty*SZ]=Pval;    
    return;
}

