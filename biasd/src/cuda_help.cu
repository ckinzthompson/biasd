#include <stdlib.h>
#include <cuda.h>

int device_count() {
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

int cuda_errors(int device){
	cudaSetDevice(device);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error : %s\n", cudaGetErrorString(err));
		return 1;
	}
	return 0;
}
