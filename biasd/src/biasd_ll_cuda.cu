#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include <time.h>

#include <cuda.h>

/* Compile with:
nvcc -arch compute_50 cuda_biasd_simpson.c -o cuda_biasd_simpson
*/

#define maxiter 2000

extern "C" double * log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon);

__global__ void kernel_loglikelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * ll);
__device__ double adaptive_quad(int idx, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon);
__device__ double integrand(double f, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau);

// ##########################################################
// ##########################################################

__device__ double integrand(double f, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau) {

	if (f == 0.){
		return exp(-k2 * tau) * (1.+ k1 * tau / 2.) * exp(-.5 * pow((d - ep2)/sigma,2.)); // Limits
	} else if (f == 1.){
		return exp(-k1 * tau) * (1.+ k2 * tau / 2.) * exp(-.5 * pow((d - ep1)/sigma,2.)); //Limits
	} else {
		double y = 2.*tau*sqrt(k1*k2*f*(1.-f));
		return exp(-(k1*f+k2*(1.-f))*tau)
			* exp(-.5*pow((d - (ep1 * f + ep2 * (1.-f)))/sigma,2.))
			* (cyl_bessel_i0(y) + (k2*f+k1*(1.-f))*tau
				* cyl_bessel_i1(y)/y); // Full Expression
	} 
	
}

__device__ double adaptive_quad(int idx, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon) {

	// Doing it this way to cache all of the non-midpoint integrand calculations in order to minimize the number of calculations that are performed.

	int i = 0;
	double ival = 0., ax = 0., ay = 0., mx = 0., my = 0., h = 0., s1 = 0., s2_left = 0., s2_right = 0.;
	double bx[maxiter], by[maxiter];
	bx[0] = 1.;
	
	// Calculate f(0), and f(1)
	ay = integrand(ax,d[idx],ep1,ep2,sigma,k1,k2,tau);
	by[0] = integrand(bx[0],d[idx],ep1,ep2,sigma,k1,k2,tau);
	
	while (i >= 0) {
		h = (bx[i] - ax)/2.;
		mx = ax + h;
		my = integrand(mx,d[idx],ep1,ep2,sigma,k1,k2,tau); // Calc f(mid-point)
		
		s1 = h/3. * (ay + 4.*my + by[i]);
		
		s2_left = h/6. * (ay + 4.*integrand(ax + h/2.,d[idx],ep1,ep2,sigma,k1,k2,tau) + my);
		s2_right = h/6. * (my + 4.*integrand(ax + h*3./2.,d[idx],ep1,ep2,sigma,k1,k2,tau) + by[i]);
		
		// Success
		if (fabs(s1 - s2_left - s2_right) < 15.*epsilon){
			ival += s2_left + s2_right;
			ax = bx[i];
			ay = by[i];
			i -= 1;
		} else { // Failure, so add midpoint to list
			i += 1;
			bx[i] = mx;
			by[i] = my;
		}
	}
	return ival;
}


__global__ void kernel_loglikelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * ll) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (idx < N) {
		double out;
		out = k2/(k1+k2) * exp(-k1*tau - .5 * pow((d[idx]-ep1)/sigma,2.)); // state 1
		out += k1/(k1+k2) * exp(-k2*tau - .5 * pow((d[idx]-ep2)/sigma,2.)); // state 2
		out += 2.*k1*k2/(k1+k2)*tau * adaptive_quad(idx,d,ep1,ep2,sigma,k1,k2,tau,epsilon); // both
		out = log(out) - .5 * log(2.* M_PI) - log(sigma); // prefactor
		ll[idx] = out; // transfer 
	}
}


double * log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon) {

	double * ll;
	ll = (double *) malloc(N*sizeof(double));

	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int threads = deviceProp.maxThreadsPerBlock/2;
	int blocks = (N+threads-1)/threads;
	
	double * d_d;
	double * ll_d;

	cudaMalloc((void**)&d_d,N*sizeof(double));
	cudaMalloc((void**)&ll_d,N*sizeof(double));
	
	cudaMemcpy(d_d,d,N*sizeof(double),cudaMemcpyHostToDevice);
	
	// Evaluate integrand at f -> store in y.
	kernel_loglikelihood<<<blocks,threads>>>(N,d_d,ep1,ep2,sigma,k1,k2,tau,epsilon,ll_d);
	
	cudaMemcpy(ll,ll_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_d);
	cudaFree(ll_d);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    	printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	return ll;
}


