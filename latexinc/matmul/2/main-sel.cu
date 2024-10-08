int main(void)
{
    int const SZ=500;

    // ..
    int const THREADX=16;
    int const THREADY=16;
    dim3 dimBlock(THREADX,THREADY,1);
    int numBlocksX=(SZ%THREADX==0 ? SZ/THREADX : SZ/THREADX +1);
    int numBlocksY=(SZ%THREADY==0 ? SZ/THREADY : SZ/THREADY +1);
    dim3 dimGrid(numBlocksX,numBlocksY,1);

    MatrixMulKernel2<<<dimGrid,dimBlock>>>(M_d,N_d,P_d,SZ);    
    //..
}    
