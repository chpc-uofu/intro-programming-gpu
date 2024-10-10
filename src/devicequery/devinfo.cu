// Code written by Wim R.M. Cardoen
// Date: 10/10/2024

#include <stdio.h>
#include <unistd.h>
#include <cuda.h>

#define MAXSZ 512
#define kToG   1E6
#define ToG   1E9

static void getDeviceInfo(cudaDeviceProp p){
     
    // Hardware
    printf("    Device Name : %s\n", p.name);
    printf("    ECC Enabled             : %d\n", p.ECCEnabled);
    printf("    Compute Capability      : %d%d\n", p.major, p.minor);
    printf("    Compute Mode            : %d\n", p.computeMode); 
    printf("    #SM on device           : %d\n", p.multiProcessorCount);
    printf("    Device Clock Rate (GHz) : %8.4E\n", double(p.clockRate)/kToG); 
    printf("    Peak Memory clock frequency (GHz): %8.4E\n", double(p.memoryClockRate)/kToG); 
    printf("    L2 Cache Size (bytes)            : %d\n", p.l2CacheSize);
    printf("    Warp Size in Threads             : %d\n\n", p.warpSize);

    // Software
    printf("    Max. #Blocks/SM                  : %d\n", p.maxBlocksPerMultiProcessor);
    printf("    Max. Size of each dim. of a Grid : (%d,%d,%d)\n",
                p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf("    Max. Size of each dim. of a Block: (%d,%d,%d)\n",
                p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("    Max. #Threads/Block              : %d\n", p.maxThreadsPerBlock);
    printf("    Max. Resident Threads/SM         : %d\n", p.maxThreadsPerMultiProcessor);
    printf("    Global Mem. available on device (bytes)  : %ld\n", p.totalGlobalMem);
    printf("    Constant Mem. available on device (bytes): %ld\n", p.totalConstMem);
    printf("    #32-bit Registers available per SM       : %d\n", p.regsPerMultiprocessor); 
    printf("    #32-bit Registers available per Block    : %d\n", p.regsPerBlock);
    printf("    Shared Mem. available per Block (bytes)  : %d\n", p.sharedMemPerBlock);
    printf("    Shared Mem. available per SM (bytes)     : %d\n\n", p.sharedMemPerMultiprocessor);
    return;
}


int main(int argc, char **argv){

    int numDevices;
    char hostname[MAXSZ]; 

    // Retrieve hostname
    hostname[MAXSZ-1]='\0'; 
    gethostname(hostname, MAXSZ-1);
    printf("Node: %s\n", hostname);
   
    // Find #Devices
    cudaGetDeviceCount(&numDevices);
    printf("#Devices detected: %d\n\n",numDevices); 

    // Loop over the devices
    for(int indDev=0; indDev < numDevices; indDev++){
        printf("\n  Device:%d\n", indDev);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, indDev);
        getDeviceInfo(devProp); 
    }
    return 0;
}
