#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

struct fxn_params {double d; double tau; double ep1; double ep2; double sigma; double k1; double k2;};
struct integration_params {gsl_integration_workspace *w; gsl_function *F; double * result; double * error; double epsilon;};

double integrand(double f, void * param_pointer) {
	struct fxn_params p = *(struct fxn_params *)param_pointer;
	double y, out;

	y = 2.*p.tau*sqrt(p.k1*p.k2*f*(1.-f));

	if (f == 0.){
		out = exp(-p.k2*p.tau) * (1.+p.k1*p.tau / 2.) * exp(-.5 * pow((p.d-p.ep2)/p.sigma,2.)); // Limits
	} else if (f == 1.){
		out = exp(-p.k1 * p.tau) * (1.+ p.k2 * p.tau / 2.) * exp(-.5 * pow((p.d - p.ep1)/p.sigma,2.)); //Limits
	} else {
		out = exp(-(p.k1*f+p.k2*(1.-f))*p.tau)
			* exp(-.5*pow((p.d - (p.ep1*f + p.ep2*(1.-f)))/p.sigma,2.))
			* (gsl_sf_bessel_I0(y) + (p.k2*f+p.k1*(1.-f))*p.tau * gsl_sf_bessel_I1(y)/y);
	}
	return out;
}

double point_ll(struct integration_params * p) {

	double lli;
	struct fxn_params t = *(struct fxn_params *)(p->F->params);

	// Peak for state 1
	lli = t.k2/(t.k1+t.k2) * exp(-t.k1*t.tau - .5 * pow((t.d-t.ep1)/t.sigma,2.));
	// Peak for state 2
	lli += t.k1/(t.k1+t.k2) * exp(-t.k2*t.tau - .5 * pow((t.d-t.ep2)/t.sigma,2.));
	// Add in the contribution from the numerical integration
	gsl_integration_qags(p->F,0.,1.,p->epsilon,5000,0.,p->w,p->result,p->error); // compute to absolute error.. relative gets strange pattern at low ln(L) < -50
	// gsl_integration_qags(p->F,0.,1.,0.,p->epsilon,50000,p->w,p->result,p->error); // compute to relative error
	lli += 2.*t.k1*t.k2/(t.k1+t.k2)*t.tau * *(p->result);
	// Log and get the prefactor
	lli = log(lli) - .5 * log(2.* M_PI) - log(t.sigma);

	return lli;
}

void log_likelihood(int N, double *d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * out) {

	int i;
	double result, error;

	struct fxn_params theta;
	theta.ep1 = ep1;
	theta.ep2 = ep2;
	theta.sigma = sigma;
	theta.k1 = k1;
	theta.k2 = k2;
	theta.tau = tau;

	gsl_integration_workspace * w = gsl_integration_workspace_alloc (5000);
	gsl_set_error_handler_off();

	gsl_function F;
	F.function = &integrand;
	F.params = &theta;

	struct integration_params p;
	p.w = w;
	p.F = &F;
	p.result = &result;
	p.error = &error;
	p.epsilon = epsilon;

	for (i = 0; i < N; i++){
		theta.d = d[i];
		out[i] = point_ll(&p);
	}

	gsl_integration_workspace_free (w);
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
