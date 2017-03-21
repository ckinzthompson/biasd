#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include <time.h>
#include <cuda.h>

/* Compile with:
nvcc -arch compute_50 cuda_biasd_simpson.c -o cuda_biasd_simpson
*/

#define DBL_EPSILON 2.22045e-16
#define DBL_MIN 4.94066e-324

extern "C" void log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * ll);
extern "C" double sum_log_likelihood(int N, double *d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon);

__global__ void kernel_loglikelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * ll);
__device__ double integrand(double f, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau);
__device__ void qgauss(double a, double b, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double *out);
__device__ double integrate(double d, double ep1, double ep2, double sigma, double k1, double k2, double tau,double epsilon);


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


/*
	Parameters ripped from Quadpack dqk21.f on 3/20/2017:
	c gauss quadrature weights and kronron quadrature abscissae and weights
	c as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
	c bell labs, nov. 1981.
*/


__constant__ static double wg[5] = {
	0.066671344308688137593568809893332,
	0.149451349150580593145776339657697,
	0.219086362515982043995534934228163,
	0.269266719309996355091226921569469,
	0.295524224714752870173892994651338
};
__constant__ static double wgk[11] = {
	0.011694638867371874278064396062192,
	0.032558162307964727478818972459390,
	0.054755896574351996031381300244580,
	0.075039674810919952767043140916190,
	0.093125454583697605535065465083366,
	0.109387158802297641899210590325805,
	0.123491976262065851077958109831074,
	0.134709217311473325928054001771707,
	0.142775938577060080797094273138717,
	0.147739104901338491374841515972068,
	0.149445554002916905664936468389821
};
__constant__ static double xgk[11] = {
	0.995657163025808080735527280689003,
	0.973906528517171720077964012084452,
	0.930157491355708226001207180059508,
	0.865063366688984510732096688423493,
	0.780817726586416897063717578345042,
	0.679409568299024406234327365114874,
	0.562757134668604683339000099272694,
	0.433395394129247190799265943165784,
	0.294392862701460198131126603103866,
	0.148874338981631210884826001129720,
	0.000000000000000000000000000000000
};

__device__ void qgauss(double a, double b, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double *out){
	// translated from Fortran from Quadpack's dqk21.f

	double center = 0.5*(b+a);
	double halflength = 0.5*(b-a);

	double result10 = 0.;
	double result21 = 0.;

	int i;
	// Function values
	double fval1[10],fval2[10];
	for (i=0;i<10;i++){
		fval1[i] = integrand(center+halflength*xgk[i],d,ep1,ep2,sigma,k1,k2,tau);
		fval2[i] = integrand(center-halflength*xgk[i],d,ep1,ep2,sigma,k1,k2,tau);
	}

	// Evaluate gauss-10 and kronrod-21
	for (i=0;i<5;i++){
		result10 += wg[i]*(fval1[2*i+1] + fval2[2*i+1]);
		result21 += wgk[2*i+1]*(fval1[2*i+1] + fval2[2*i+1]);
		result21 += wgk[2*i]*(fval1[2*i] + fval2[2*i]);
	}

	double fc = integrand(center,d,ep1,ep2,sigma,k1,k2,tau);
	result21 += wgk[10]*fc;

	// // Removed error calculation.....
	// errors
	// double resabs = result21; // in this case resabs = result21 b/c it's all positive
	// double reskh = result21*.5;
	// double resasc = wgk[10]*fabs(fc-reskh);
	// for (i=0;i<10;i++){
		// resasc += wgk[i]*(fabs(fval1[i] - reskh) + fabs(fval2[i] - reskh));
	// }

	out[0] = result21*halflength;
	// resabs *= halflength;
	// resasc *= halflength;
	// out[1] = fabs(result10-result21)*halflength;
	// if (resasc != 0. && out[1] != 0.){
	// 	out[1] = resasc*fmin(1.,pow(200.*out[1]/resasc,1.5));
	// }
	// if (resabs > DBL_MIN/(50.*DBL_EPSILON)){
	// 	out[1] = fmax(resabs*(DBL_EPSILON*50.),out[1]);
	// }
}

__device__ double integrate(double d, double ep1, double ep2, double sigma, double k1, double k2, double tau,double epsilon){

	double out[2],integral = 0.,error = 0.;

	int i;
	for (i=0;i<20;i++){
		qgauss(0.05*(double)i,0.05*(double)(i+1),d,ep1,ep2,sigma,k1,k2,tau,out);
		integral += out[0];
		error += out[1];
	}
	return integral;
}


__global__ void kernel_loglikelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * ll) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < N) {
		double out;
		out = k2/(k1+k2) * exp(-k1*tau - .5 * pow((d[idx]-ep1)/sigma,2.)); // state 1
		out += k1/(k1+k2) * exp(-k2*tau - .5 * pow((d[idx]-ep2)/sigma,2.)); // state 2

		out += 2.*k1*k2/(k1+k2)*tau *integrate(d[idx],ep1,ep2,sigma,k1,k2,tau,epsilon); // both

		out = log(out) - .5 * log(2.* M_PI) - log(sigma); // prefactor
		ll[idx] = out; // transfer
	}
}

void get_cuda_errors(){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
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

	get_cuda_errors();
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


// This is for testing... the value is -45.312935111611729
// Compile with nvcc ./biasd_ll_cuda.cu -o test
// Run with $ ./test

// Python code to double check is:
/*
import biasd as b
b.likelihood.use_python_ll()
d = np.array((0.87755042,  0.90101722,  0.88297422,  0.90225072,  0.91185969,        0.88479424,  0.64257305,  0.23650566,  0.17532272,  0.24785572,        0.77647345,  0.12143413,  0.04994399,  0.19918067,  0.09625039,
        0.14283554,  0.30052487,  0.8937437 ,  0.90544194,  0.87350816,        0.62315481,  0.48258872,  0.77018322,  0.42989469,  0.69183523,        0.35556625,  0.90622313,  0.12529433,  0.74309849,  0.8860914 ,        0.8335358 ,  0.56208782,  0.45287218,  0.79373139,  0.42808399,        0.86643919,  0.70459052,  0.09161765,  0.53514735,  0.06578612,        0.09050594,  0.14923124,  0.8579178 ,  0.884698  ,  0.8745358 ,        0.89191605,  0.57743238,  0.80656044,  0.9069933 ,  0.65817311))
theta = np.array((0.,1.,.05,3.,8.))
tau = 0.1
print b.likelihood.log_likelihood(theta,d,tau)
*/


/*
int main(){
	double d[50] = {0.87755042,  0.90101722,  0.88297422,  0.90225072,  0.91185969,        0.88479424,  0.64257305,  0.23650566,  0.17532272,  0.24785572,        0.77647345,  0.12143413,  0.04994399,  0.19918067,  0.09625039,
        0.14283554,  0.30052487,  0.8937437 ,  0.90544194,  0.87350816,        0.62315481,  0.48258872,  0.77018322,  0.42989469,  0.69183523,        0.35556625,  0.90622313,  0.12529433,  0.74309849,  0.8860914 ,        0.8335358 ,  0.56208782,  0.45287218,  0.79373139,  0.42808399,        0.86643919,  0.70459052,  0.09161765,  0.53514735,  0.06578612,        0.09050594,  0.14923124,  0.8579178 ,  0.884698  ,  0.8745358 ,        0.89191605,  0.57743238,  0.80656044,  0.9069933 ,  0.65817311};

	double sum = 0;
	printf("Starting Test...\n");
	sum = sum_log_likelihood(50,d,0.,1.,.05,3.,8.,.1,1e-10);
	double truth = -45.312935111611729;

	printf("Real : %.10f\n",truth);
	printf("Calcd: %.10f\n",sum);
	printf("%%Diff: %.10f\n",abs((truth-sum)/truth)*100.);
}
*/
