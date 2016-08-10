#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>

#define n_integration_pts 751

/* Compile with:
nvcc -arch compute_50 cuda_biasd_simpson.c -o cuda_biasd_simpson
*/


extern "C" double * log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau);
__global__ void kernel_loglikelihood(int nf, double * f, int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double * ll);
__device__ double integrand(double f, double ep1, double ep2, double sigma, double k1, double k2, double tau);
__device__ double simpson(int N, double * x, double * y);

__device__ double integrand(double f, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau) {
	double y, out;
	
	y = 2.*tau*sqrt(k1*k2*f*(1.-f));
	
	out = exp(-(k1*f+k2*(1.-f))*tau)
		* exp(-.5*pow((d - (ep1*f + ep2*(1.-f)))/sigma,2.))
		* (cyl_bessel_i0(y) + (k2*f+k1*(1.-f))*tau * cyl_bessel_i1(y)/y);
		
	return out;
}

__device__ double simpson(int n, double * x, double * y){
	int i;
	double out = 0.;
	double h0,h1;
	
	// For all the sections with midpoints
	for (i = 0; i < n-2; i+=2) { 
		// Calculate left and right spacings
		h0 = x[i+1] - x[i];
		h1 = x[i+2] - x[i+1];
		// Make sure the divisions aren't too small.
		if (~ isnan(h0/h1)) {
		// Add this section to result (Simpson's rule)
			out += (h0+h1)/6. * (y[i]*(2.-h1/h0) 
				+ y[i+1]*(h0+h1)*(h0+h1)/h0/h1 + y[i+2]*(2.-h0/h1));
		}
	}
	return out;
}


__global__ void kernel_loglikelihood(int nf, double * f, int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double * ll) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (idx < N) {
		double out;
		out = k2/(k1+k2) * exp(-k1*tau - .5 * pow((d[idx]-ep1)/sigma,2.)); // state 1
		out += k1/(k1+k2) * exp(-k2*tau - .5 * pow((d[idx]-ep2)/sigma,2.)); // state 2
		
		double intval[n_integration_pts]; //cheat
		int i;
		for (i = 0; i < nf; i++) {
			intval[i] = integrand(f[i],d[idx],ep1,ep2,sigma,k1,k2,tau);
		}

		out += 2.*k1*k2/(k1+k2)*tau * simpson(nf,f,intval); // both

		out = log(out) - .5 * log(2.* M_PI) - log(sigma); // prefactor
		ll[idx] = out; // transfer 
	}
}


double * log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau) {

	double * ll;
	double * f;
	int i;
	int nf = n_integration_pts;
	
	ll = (double *) malloc(N*sizeof(double));
	f = (double *) malloc(nf*sizeof(double));

	// Set up fraction, x, to range from 0 to 1 on a logit scale (1e-30 to 1e30) to make sure we hit the edges in the integration
	double d1 = ((double)nf-1.) - 0.;
	double d2 = 30. - (-30.);
	for (i = 0; i < nf; i++){
		f[i] = (double)i /d1 * d2 - 30.0;
		f[i] = 1./(exp(-f[i]) + 1.);
	}
	
	
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int threads = deviceProp.maxThreadsPerBlock;
	int blocks = (N+threads-1)/threads;
	
	double * d_d;
	double * f_d;
	double * ll_d;
	cudaMalloc((void**)&d_d,N*sizeof(double));
	cudaMalloc((void**)&f_d,nf*sizeof(double));
	cudaMalloc((void**)&ll_d,N*sizeof(double));
	cudaMemcpy(d_d,d,N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(f_d,f,nf*sizeof(double),cudaMemcpyHostToDevice);
	
	// Evaluate integrand at f -> store in y.
	kernel_loglikelihood<<<blocks,threads>>>(nf,f_d,N,d_d,ep1,ep2,sigma,k1,k2,tau,ll_d);
	cudaMemcpy(ll,ll_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_d);
	cudaFree(f_d);
	cudaFree(ll_d);
	
	free(f);
	return ll;
}


