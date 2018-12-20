#ifndef CUDA_HELP_H_
#define CUDA_HELP_H_

// External calls from python
extern "C" int device_count();
extern "C" int cuda_errors(int);

__device__ double atomicAdd(double* address, double val);

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull =(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));

	// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

__global__ void cuda_parallel_sum(double *in, int num_elements, double *sum);
__global__ double parallel_sum(double * a_d, int N, int num_SMs);

int get_padding(int device, int N);
#endif
