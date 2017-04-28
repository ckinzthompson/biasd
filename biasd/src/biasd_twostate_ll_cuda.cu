#include <stdlib.h>
#include <cuda.h>
#include "biasd_twostate_integrate_cuda.h"
#include "biasd_twostate_ll_cuda.h"

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

void log_likelihood(int device, int N, double * d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll) {

	// Sanity checks from the model
	if ((ep1 < ep2) && (sigma1 > 0.) && (sigma2 > 0.) && (k1 > 0.) && (k2 > 0.) && (tau > 0.) && (epsilon > 0.)) {

		// Initialize CUDA things
		//get_cuda_errors();
		cudaSetDevice(device);
		//cudaDeviceProp deviceProp;
		//cudaGetDeviceProperties(&deviceProp, device);
		int threads = 256;//deviceProp.maxThreadsPerBlock/8;
		int blocks = (N+threads-1)/threads;

		double * d_d;
		double * ll_d;

		cudaMalloc((void**)&d_d,N*sizeof(double));
		cudaMalloc((void**)&ll_d,N*sizeof(double));

		cudaMemcpy(d_d,d,N*sizeof(double),cudaMemcpyHostToDevice);

		// Evaluate integrand at f -> store in y.

		kernel_loglikelihood<<<blocks,threads>>>(N,d_d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon,ll_d);

		cudaMemcpy(ll,ll_d,N*sizeof(double),cudaMemcpyDeviceToHost);

		cudaFree(d_d);
		cudaFree(ll_d);

		//get_cuda_errors();
	} else {
		int i;
		for (i=0;i<N;i++){ ll[i] = -INFINITY;}
	}
}

double sum_log_likelihood(int device, int N, double *d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon) {

	int i = 0;
	double sum = 0.;

	double * ll;
	ll = (double *) malloc(N*sizeof(double));

	log_likelihood(device,N,d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon,ll);

	for (i=0;i<N;i++) {
		sum += ll[i];
	}
	free(ll);
	return sum;
}
