int main(void)
{
    int const SZ=16;
    // ...
    // Invoke Kernel to generate P=MxN
    dim3 dimBlock(SZ,SZ,1);
    dim3 dimGrid(1,1,1);
    MatrixMulKernel1<<<dimGrid,dimBlock>>>(M_d,N_d,P_d,SZ);
    //..
}    
