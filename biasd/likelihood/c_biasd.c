#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "c_biasd.h"

/*
	Parameters ripped from Quadpack dqk21.f on 3/20/2017:
	c gauss quadrature weights and kronron quadrature abscissae and weights
	c as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
	c bell labs, nov. 1981.
*/
static double wg[5] = {
	0.066671344308688137593568809893332,
	0.149451349150580593145776339657697,
	0.219086362515982043995534934228163,
	0.269266719309996355091226921569469,
	0.295524224714752870173892994651338
};
static double wgk[11] = {
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
static double xgk[11] = {
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

// #############################################################################
// #############################################################################
// Bessels are from numerical recipies in C?
double bessel_i0(double x) {
	double ax,ans;
	double y;

	if ((ax=fabs(x)) < 3.75) {
		y = x/3.75;
		y *= y;
		ans = 1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
			+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
	} else {
		y=3.75/ax; ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
			+y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
			+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
			+y*0.392377e-2))))))));
	}
	return ans;
}

double bessel_i1(double x) {
	double ax,ans;
	double y;

	if ((ax=fabs(x)) < 3.75) {
		y = x/3.75;
		y *= y;
		ans = ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
			+y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
	} else {
		y = 3.75/ax;
		ans = 0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
			-y*0.420059e-2));
		ans = 0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
			+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
		ans *= (exp(ax)/sqrt(ax));
	}
	return x < 0.0 ? -ans : ans;
}

double integrand(double f, ip * p) {
	double y,var;

	y = 2. * p->tau * sqrt(p->k1 * p->k2 * f * (1.-f));
	var = p->sig1 * p->sig1 * f + p->sig2 * p->sig2 * (1.-f);

	return exp(-(p->k1 * f + p->k2*(1.-f)) * p->tau)/ sqrt(var)
		* exp(-.5*pow((p->d - (p->ep1 * f + p->ep2 * (1.-f))),2.)/var)
		* (bessel_i0(y) + (p->k2*f+p->k1*(1.-f))*p->tau
		* bessel_i1(y)/y); // Full Expression

}

// #############################################################################
// #############################################################################

void qg10k21(double a, double b, double *out, ip * p) {
	// Do quadrature with the Gauss-10, Kronrod-21 rule
	// translated from Fortran from Quadpack's dqk21.f...

	double center = 0.5*(b+a);
	double halflength = 0.5*(b-a);

	double result10 = 0.;
	double result21 = 0.;

	int i;
	// Pre-calculate function values
	double fval1[10],fval2[10];
	for (i=0;i<10;i++){
		fval1[i] = integrand(center+halflength*xgk[i],p);
		fval2[i] = integrand(center-halflength*xgk[i],p);
	}

	for (i=0;i<5;i++){
		// Evaluate Gauss-10
		result10 += wg[i]*(fval1[2*i+1] + fval2[2*i+1]);

		//Evaluate Kronrod-21
		result21 += wgk[2*i]*(fval1[2*i] + fval2[2*i]);
		result21 += wgk[2*i+1]*(fval1[2*i+1] + fval2[2*i+1]);
	}

	// Add 0 point to Kronrod-21
	double fc = integrand(center,p);
	result21 += wgk[10]*fc;

	// Scale results to the interval
	result10 *= fabs(halflength);
	result21 *= fabs(halflength);

	// Error calculation
	double avgg = result21/fabs(b-a), errors = 0.;
	for (i=0;i<5;i++){
		errors += wgk[2*i]*(fabs(fval1[2*i]-avgg)+fabs(fval2[2*i]-avgg));
		errors += wgk[2*i+1]*(fabs(fval1[2*i+1]-avgg)+fabs(fval2[2*i+1]-avgg));
	}
	errors += wgk[10]*(fabs(fc-avgg));
	errors *= fmin(1.,pow(200.*fabs(result21-result10)/errors,1.5));

	// Output results
	out[0] += result21;
	out[1] += errors;
}

void adaptive_integrate(double a0, double b0, double *out, double epsilon, ip * p){

	// a is the lowerbound of the integral
	// b is the upperbound of the integral
	// out[0] is the value of the integral
	// out[1] is the error of the integral
	// epsilon is the error limit we must reach in each sub-interval

	double current_a = a0, current_b = b0, temp_out[2];

	// This loops from the bottom to the top, subdividing and trying again on the lower half, then moving up once converged to epsilon
	while (current_a < b0){
		// Evaluate quadrature on current interval
		temp_out[0] = 0.;
		temp_out[1] = 0.;
		// Perform the quadrature
		qg10k21(current_a,current_b,temp_out,p);
		// Check convergence on this interval
		if (temp_out[1] < epsilon ){
			// keep the results
			out[0] += temp_out[0];
			out[1] += temp_out[1];
			// step the interval
			current_a = current_b;
			current_b = b0;
		} else { // if the interval failed to converge
			// sub-divide the interval
			current_b -= 0.5*(current_b-current_a);
		}
	}
}


// #############################################################################
// #############################################################################

void log_likelihood(int N, double * d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * out) {
	int i;

	if ((ep1 < ep2) && (sigma1 > 0.) && (sigma2 > 0.) && (k1 > 0.) && (k2 > 0.) && (tau > 0.) && (epsilon > 0.)) {

		ip p = {0.,ep1,ep2,sigma1,sigma2,k1,k2,tau};

		double lli, intval[2] = {0.,0.};

		for (i=0;i<N;i++){

			// Peak for state 1
			lli = k2/(k1+k2) / sigma1 * exp(-1. * k1 * tau - .5 * pow((d[i] - ep1) / sigma1,2.));
			// Peak for state 2
			lli += k1/(k1+k2) / sigma2 * exp(-1.* k2 * tau - .5 * pow((d[i] - ep2) / sigma2,2.));

			// Add in the contribution from the numerical integration
			p.d = d[i];
			intval[0] = 0.;
			intval[0] = 0.;
			adaptive_integrate(0,1,intval,epsilon,&p);
			lli += 2.*k1 * k2/(k1 + k2) * tau * intval[0];

			// Log and get the prefactor
			lli = log(lli) - .5 * log(2.* M_PI);
			out[i] = lli;
		}
	} else {
		for (i=0;i<N;i++) {
			out[i] = - INFINITY;
		}
	}
}

double sum_log_likelihood(int N, double *d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon) {

	int i = 0;
	double sum = 0.;

	double * ll;
	ll = (double *) malloc(N*sizeof(double));

	log_likelihood(N,d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon,ll);
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
