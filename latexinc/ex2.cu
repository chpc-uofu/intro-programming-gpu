// Copy M from Host onto Device
if(cudaMemcpy(M_d, M_h, sizeof(double)*SZ2,
	      cudaMemcpyHostToDevice) != cudaSuccess)
{
   printf(" ERROR: copy vector M: HOST -> DEVICE\n");
   return 1;
}


// Copy P (result) from the Device to the Host
if(cudaMemcpy(P_h, P_d, sizeof(double)*SZ2,
	      cudaMemcpyDeviceToHost) != cudaSuccess)
{
   printf(" ERROR: copy vector N: DEVICE -> HOST\n");
   return 1;
}
