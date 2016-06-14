#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

struct fxn_params {double d; double tau; double ep1; double ep2; double sigma; double k1; double k2;}; 

double integrand(double f, void * param_pointer) {
	struct fxn_params p = *(struct fxn_params *)param_pointer;
	double y, out;
	
	y = 2.*p.tau*sqrt(p.k1*p.k2*f*(1.-f));
	
	out = exp(-(p.k1*f+p.k2*(1.-f))*p.tau)
		* exp(-.5*pow((p.d - (p.ep1*f + p.ep2*(1.-f)))/p.sigma,2.))
		* (gsl_sf_bessel_I0(y) + (p.k2*f+p.k1*(1.-f))*p.tau * gsl_sf_bessel_I1(y)/y);
		
	return out;
}

double log_likelihood(int N, double *d, double ep1, double ep2, double sigma, double k1, double k2, double tau) {
	
	int i;
	double lli,ll = 0.;
	double result, error;
	
	struct fxn_params theta;
	theta.ep1 = ep1;
	theta.ep2 = ep2;
	theta.sigma = sigma;
	theta.k1 = k1;
	theta.k2 = k2;
	theta.tau = tau;
	
	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
	gsl_set_error_handler_off();
	
	gsl_function F;
	F.function = &integrand;
	F.params = &theta;
	
	for (i = 0; i < N; i++){
		// Peak for state 1
		lli = k2/(k1+k2) * exp(-k1*tau - .5 * pow((d[i]-ep1)/sigma,2.));
		// Peak for state 2
		lli += k1/(k1+k2) * exp(-k2*tau - .5 * pow((d[i]-ep2)/sigma,2.));
		// Add in the contribution from the numerical integration
		theta.d = d[i];
		gsl_integration_qags(&F,0.,1.,0.,1e-6,1000,w,&result,&error); // compute to relative error
		lli += 2.*k1*k2/(k1+k2)*tau * result;
		// Log and get the prefactor
		lli = log(lli) - .5 * log(2.* M_PI) - log(sigma); 
		ll += lli;
	}
	
	gsl_integration_workspace_free (w);
	return ll;
}
