#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// -I$(OMPI_DIR)/include -L$(OMPI_DIR)/lib -lmpi_ibm

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
	cudaError_t cuErr = call;                                                             \
	if(cudaSuccess != cuErr){                                                             \
		printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
		exit(0);                                                                            \
	}                                                                                     \
}while(0)


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    // Check that only 2 MPI processes are spawn
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if(comm_size != 2)
    {
        printf("This application is meant to be run with 2 MPI processes, not %d.\n", comm_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    // Get my rank
    int my_rank;
    int window_buffer = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    cudaErrorCheck( cudaGetDeviceCount(&window_buffer));

    MPI_Win window;
    MPI_Win_create(&window_buffer, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);
 
    int value_fetched;
    if(my_rank == 0)
    {
        for(int i = 1; i < comm_size; i++)
        {
            MPI_Get(&value_fetched, 1, MPI_INT, i, 0, 1, MPI_INT, window);
            window_buffer += value_fetched;
        }

    }
 
    // Wait for the MPI_Get issued to complete before going any further
    MPI_Win_fence(0, window);
 
    if(my_rank == 0)
    {
        printf("[MPI process 0] Value fetched from MPI process 1 window: %d.\n", value_fetched);
    }
 
    // Destroy the window
    MPI_Win_free(&window);
 
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}