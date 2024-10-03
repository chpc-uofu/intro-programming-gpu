#include "aux.h"
#include "mul.h"

int main(void)
{
    double *M_h, *N_h, *P_h;  // Pointers (host)
    double *M_d, *N_d, *P_d;  // Pointers (device)
    int const SZ=512;
    int const SZ2=SZ*SZ;

    // Allocate the matrices M, N & P on the host
    M_h=allocVectorOnHost(SZ2);
    N_h=allocVectorOnHost(SZ2);
    P_h=allocVectorOnHost(SZ2);

    // Initialize the matrices M, N (host)
    for(int k=0; k<SZ2; k++)
        M_h[k]=(double)k;
    for(int k=0; k<SZ2; k++)
        N_h[k]=k+1.;

    // Allocate M, N and P on the device
    if(cudaMalloc(&M_d,sizeof(double)*SZ2) != cudaSuccess){
       printf(" ERROR: alloc vector M on DEVICE\n");
       return 1;}

    if(cudaMalloc(&N_d,sizeof(double)*SZ2) != cudaSuccess){
       printf(" ERROR: alloc vector N on DEVICE\n");
       return 1;}
    
    if(cudaMalloc(&P_d,sizeof(double)*SZ2) != cudaSuccess){
       printf(" ERROR: alloc vector P on DEVICE \n");
       return 1;}


    // Copy M,N from Host onto Device
    if(cudaMemcpy(M_d,M_h,sizeof(double)*SZ2,cudaMemcpyHostToDevice) != cudaSuccess){
       printf(" ERROR: copy vector M: HOST -> DEVICE\n");
       return 1;}
    
    if(cudaMemcpy(N_d,N_h,sizeof(double)*SZ2,cudaMemcpyHostToDevice) != cudaSuccess){
       printf(" ERROR: copy vector N: HOST -> DEVICE\n");
       return 1;}


    // KERNEL: P=M*N on DEVICE 
    int const THREADX=16;
    int const THREADY=16;
    dim3 dimBlock(THREADX,THREADY,1);
    int numBlocksX=(SZ%THREADX==0 ? SZ/THREADX : SZ/THREADX +1);
    int numBlocksY=(SZ%THREADY==0 ? SZ/THREADY : SZ/THREADY +1);
    dim3 dimGrid(numBlocksX,numBlocksY,1);

    printf(" Calling Kernel ...\n");
    MatrixMulKernel3<<<dimGrid,dimBlock>>>(M_d,N_d,P_d,SZ);
    printf(" Kernel Call Finished ...\n");

    if(cudaSuccess != cudaGetLastError()){
       printf(" ERROR: MatrixMulKernel\n");
       return 1;}


    // Copy P (result) from the Device to the Host
    if(cudaMemcpy(P_h,P_d,sizeof(double)*SZ2,cudaMemcpyDeviceToHost) != cudaSuccess){
       printf(" ERROR: copy vector N: DEVICE -> HOST\n");
       return 1;}


    #ifdef check
       // Calc. P=M*N on the Host
       double *P = matrixMulHost(M_h,N_h,SZ);

       // Calc. Frob. Norm of (P-P_h)
       double frobnorm = calcDiff(P,P_h,SZ);
       printf("\n Frob. Norm(P-P_h):%16.10lf\n\n", frobnorm);

       // Deallocate P
       free(P);
    #endif


    // Deallocate matrices on the Host (M_h, N_h, P_h)
    free(M_h);
    free(N_h);
    free(P_h);


    // Deallocate matrices on the Device (M_d, N_d, P_d)
    if(cudaFree(M_d) != cudaSuccess){
       printf(" ERROR: unable to deallocate M_d (DEVICE)\n");
       return 1;}

    if(cudaFree(N_d) != cudaSuccess){
       printf(" ERROR: unable to deallocate N_d (DEVICE)\n");
       return 1;}

    if(cudaFree(P_d) != cudaSuccess){
       printf(" ERROR: unable to deallocate P_d (DEVICE)\n");
       return 1;}

    return 0;
}
