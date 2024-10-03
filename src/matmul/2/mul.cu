#include <mul.h>

__global__ void MatrixMulKernel2(double *M_d, double *N_d,
                                 double *P_d, int const SZ)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if((tx<SZ) && (ty<SZ))
    { 
       double Pval=0.0;
       for(int k=0; k<SZ; k++)
           Pval += M_d[tx*SZ+k]*N_d[k*SZ+ty];
       P_d[tx*SZ+ty]=Pval;   
    }
    return;
}
