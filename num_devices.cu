#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
	cudaError_t cuErr = call;                                                             \
	if(cudaSuccess != cuErr){                                                             \
		printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
		exit(0);                                                                            \
	}                                                                                     \
}while(0)

int main()
{
    int NumOfDevices;

    cudaErrorCheck( cudaGetDeviceCount(&NumOfDevices));

    printf("number of devices is %d\n", NumOfDevices);

    return 0;

}