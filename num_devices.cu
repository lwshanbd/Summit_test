#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main()
{
    int NumOfDevices;

    cudaErrorCheck( cudaGetDeviceCount(&NumOfDevices));

    printf("number of devices is %d\n", NumOfDevices);

    return 0;
    
}