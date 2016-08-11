#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include <time.h>

#include <cuda.h>

/* Compile with:
nvcc -arch compute_50 cuda_biasd_simpson.c -o cuda_biasd_simpson
*/

#define maxiter 2000

extern "C" void log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * ll);
extern "C" double sum_log_likelihood(int N, double *d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon);

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
	
	int iters = 0;
	while (i >= 0 && iters < maxiter) {
		iters++;
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


void log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * ll) {

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
}

double sum_log_likelihood(int N, double *d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon) {
	
	int i = 0;
	double sum = 0.;
	
	double * ll;
	ll = (double *) malloc(N*sizeof(double));
	
	log_likelihood(N,d,ep1,ep2,sigma,k1,k2,tau,epsilon,ll);
	
	for (i=0;i<N;i++) {
		sum += ll[i];
	}
	free(ll);
	return sum;
}


/*
int main(){
	double d[50] = {0.87755042,  0.90101722,  0.88297422,  0.90225072,  0.91185969,        0.88479424,  0.64257305,  0.23650566,  0.17532272,  0.24785572,        0.77647345,  0.12143413,  0.04994399,  0.19918067,  0.09625039,
        0.14283554,  0.30052487,  0.8937437 ,  0.90544194,  0.87350816,        0.62315481,  0.48258872,  0.77018322,  0.42989469,  0.69183523,        0.35556625,  0.90622313,  0.12529433,  0.74309849,  0.8860914 ,        0.8335358 ,  0.56208782,  0.45287218,  0.79373139,  0.42808399,        0.86643919,  0.70459052,  0.09161765,  0.53514735,  0.06578612,        0.09050594,  0.14923124,  0.8579178 ,  0.884698  ,  0.8745358 ,        0.89191605,  0.57743238,  0.80656044,  0.9069933 ,  0.65817311};
        
	double sum = 0;
	
	sum = sum_log_likelihood(50,d,0.,1.,.05,3.,8.,.1,1e-6);
	
	printf("%f\n",sum);       
}
*/


