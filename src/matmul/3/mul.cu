#include <mul.h>
#define WIDTH 16
__global__ void MatrixMulKernel3(double *M_d, double *N_d, double *P_d, int const SZ)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ double M_s[WIDTH][WIDTH];
    __shared__ double N_s[WIDTH][WIDTH];

    double Pval=0.0;
    int nslices=(SZ%WIDTH==0)?(SZ/WIDTH):(SZ/WIDTH+1);   
    for(int islice=0; islice<nslices; islice++)
    {
        M_s[threadIdx.x][threadIdx.y]=M_d[tx*SZ + islice*WIDTH + threadIdx.y]; 
        N_s[threadIdx.x][threadIdx.y]=N_d[islice*WIDTH*SZ + threadIdx.x*SZ +ty ];
        __syncthreads();

        for(int k=0; k<WIDTH; k++)
            Pval += M_s[threadIdx.x][k]*N_s[k][threadIdx.y];
        __syncthreads();  
    }

    if(tx<SZ && ty<SZ)
       P_d[tx*SZ+ty]=Pval;
    return;
}
