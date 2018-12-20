#ifndef CUDA_HELP_H_
#define CUDA_HELP_H_

// External calls from python
extern "C" int device_count();
extern "C" int cuda_errors(int);

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val);
#endif

__global__ void cuda_parallel_sum(double *in, int num_elements, double *sum);
__global__ double parallel_sum(double * a_d, int N, int num_SMs);

int get_padding(int device, int N);
#endif
