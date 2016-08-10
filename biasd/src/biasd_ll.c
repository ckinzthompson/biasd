#include <stdlib.h>
#include <math.h>
#include "biasd_ll.h"

/* Compile with:
gcc -O3 biasd_ll.c -o biasd_ll -lm
*/

#define n_integration_points 501

// Bessel Functions from NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING, section 6.6

//Returns the modified Bessel function I0(x) for any real x.
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

//Returns the modified Bessel function I1(x) for any real x. 
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

double integrand(double f, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau) {
	double y, out;
	
	y = 2.*tau*sqrt(k1*k2*f*(1.-f));
	
	out = exp(-(k1*f+k2*(1.-f))*tau)
		* exp(-.5*pow((d - (ep1*f + ep2*(1.-f)))/sigma,2.))
		* (bessel_i0(y) + (k2*f+k1*(1.-f))*tau * bessel_i1(y)/y);
		
	return out;
}

//Calculate Simpson's Rule Integration w/ Unequal Intervals
double simpson(int n, double * x, double * y){
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

double * log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau) {
	
	double *out;
	out = (double *) malloc(N*sizeof(double));

	double lli;
	double* f;
	double* integrand_vals;
	int i,j;
	int nf = n_integration_points; // kind of arbitrary... but DEFINITELY enough points
	
	// For numerical integration to marginalize fraction, f. Will be reused.
	f = (double *) malloc(nf*sizeof(double));
	integrand_vals = (double *) malloc(nf*sizeof(double));

	// Set up fraction, x, to range from 0 to 1 on a logit scale (1e-30 to 1e30) to make sure we hit the edges in the integration
	double d1 = ((double)nf-1.) - 0.;
	double d2 = 30. - (-30.);
	for (i = 0; i < nf; i++){
		f[i] = (double)i /d1 * d2 - 30.0;
		f[i] = 1./(exp(-f[i]) + 1.);
	}
	
	// Calculate Log-likelihood for each data point
	for (i=0;i < N;i++) {
		// Peak for state 1
		lli = k2/(k1+k2) * exp(-k1*tau - .5 * pow((d[i]-ep1)/sigma,2.));
		// Peak for state 2
		lli += k1/(k1+k2) * exp(-k2*tau - .5 * pow((d[i]-ep2)/sigma,2.));
		// calculate integrand at each fraction point, f. 
		for (j=0;j<nf;j++){
			integrand_vals[j] = integrand(f[j],d[i],ep1,ep2,sigma,k1,k2,tau);
		}
		// Add in the contribution from the numerical integration
		lli += 2.*k1*k2/(k1+k2)*tau * simpson(nf,f,integrand_vals);

		// Log and get the prefactor
		lli = log(lli) - .5 * log(2.* M_PI) - log(sigma); 
		out[i] = lli;
	}
	
	free(f);
	free(integrand_vals);
	
	return out;
}

/*
double sum_log_likelihood(int N, double *d, double ep1, double ep2, double sigma, double k1, double k2, double tau) {
	
	int i = 0;
	double sum = 0.;
	double *ll;
	ll = (double *) malloc(N*sizeof(double));
	
	log_likelihood(N,d,ll,ep1,ep2,sigma,k1,k2,tau);
	for (i=0;i<N;i++) {
		sum += ll[i];
	}
	free(ll);
	return sum;
}
*/
