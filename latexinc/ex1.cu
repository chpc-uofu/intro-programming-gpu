double *M_d, *N_d, *P_d;  // Pointers (device)
int const SZ=16;
int const SZ2=SZ*SZ;

// Allocate M on device (M_d)
if(cudaMalloc(&M_d,sizeof(double)*SZ2) != cudaSuccess)
{
   printf(" ERROR: alloc vector M on DEVICE \n");
   return 1;
}

// Deallocate matrix M_d
if(cudaFree(M_d) != cudaSuccess)
{
   printf(" ERROR: unable to deallocate M_d (DEVICE)\n");
   return 1;
}
