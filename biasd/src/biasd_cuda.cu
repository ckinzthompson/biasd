#include <stdlib.h>
#include <cuda.h>
#include "biasd_cuda_integrate.h"
#include "biasd_cuda.h"
#include "cuda_help.h"

// ##########################################################
// ##########################################################

__global__ void kernel_loglikelihood(int N, double * d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < N) {
		double out, intval[2] = {0.,0.};
		theta t = {.ep1 = ep1, .ep2 = ep2, .sig1 = sigma1, .sig2 = sigma2, .k1 = k1, .k2 = k2, .tau = tau};

		// Calculate the state contributions
		out = k2/(k1+k2) / sigma1 * exp(-k1*tau - .5 * pow((d[idx]-ep1)/sigma1,2.)); // state 1
		out += k1/(k1+k2) / sigma2 * exp(-k2*tau - .5 * pow((d[idx]-ep2)/sigma2,2.)); // state 2

		// Perform the blurring integral
		adaptive_integrate(0.,1.,intval,epsilon,d[idx],&t);

		out += 2.*k1*k2/(k1+k2)*tau *intval[0]; // the blurring contribution
		out = log(out) - .5 * log(2.* M_PI); // Add prefactor
		ll[idx] = out; // transfer out result
	}
}

void load_data(int device, int N, double * d, void* d_d, void * ll_d){
	cudaSetDevice(device);
	int padding = get_padding(device,N);

	cudaMalloc((void**)&d_d,N*sizeof(double));
	cudaMalloc((void**)&ll_d,(N+padding)*sizeof(double)); // Make this bigger to pad to 1024

	cudaMemcpy(d_d,d,N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemset(ll_d, 0.0, (N+padding)*sizeof(double));
}

void free_data(void *d_d, void *ll_d){
	cudaFree(d_d);
	cudaFree(ll_d);
}

void log_likelihood(int device, int N, void *d_d, void *ll_d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll) {

	// Sanity checks from the model
	if ((ep1 < ep2) && (sigma1 > 0.) && (sigma2 > 0.) && (k1 > 0.) && (k2 > 0.) && (tau > 0.) && (epsilon > 0.)) {

		// Initialize CUDA things
		int threads = 256;//deviceProp.maxThreadsPerBlock/8;
		int blocks = (N+threads-1)/threads;
		// Evaluate integrand at f -> store in ll.
		kernel_loglikelihood<<<blocks,threads>>>(N,d_d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon,ll_d);
		cudaMemcpy(ll,ll_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	} else {
		int i;
		for (i=0;i<N;i++){ ll[i] = -INFINITY;}
	}
}

double sum_log_likelihood(int device, int N, void *d_d, void *ll_d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon) {

	double sum = 0.;

	if ((ep1 < ep2) && (sigma1 > 0.) && (sigma2 > 0.) && (k1 > 0.) && (k2 > 0.) && (tau > 0.) && (epsilon > 0.)) {
		int padding = get_padding(device,N);
		int nSM = get_num_SM(device);

		int threads = 256;//deviceProp.maxThreadsPerBlock/8;
		int blocks = (N+threads-1)/threads;

		kernel_loglikelihood<<<blocks,threads>>>(N,d_d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon,ll_d);
		sum = parallel_sum(ll_d,N+padding,nSM);
	} else {
		sum = -INFINITY;
	}
	return sum;
}
