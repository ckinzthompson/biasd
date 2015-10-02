#include <math.h>

// From GSL's cheb_eval_e.c, bessel_I0.c, and bessel_I1.c <-- GNU license stuff

/* ----------------------------------------- */
/* ----------Chebyshev Polynomials---------- */
/* ----------------------------------------- */

typedef struct
{
  double * c;   /* coefficients  c[0] .. c[order] */
  int order;    /* order of expansion             */
  double a;     /* lower interval point           */
  double b;     /* upper interval point           */
} cheb_series;

static double bi0_data[12] = {
  -.07660547252839144951,
  1.92733795399380827000,
   .22826445869203013390, 
   .01304891466707290428,
   .00043442709008164874,
   .00000942265768600193,
   .00000014340062895106,
   .00000000161384906966,
   .00000000001396650044,
   .00000000000009579451,
   .00000000000000053339,
   .00000000000000000245
};
static cheb_series bi0_cs = {bi0_data, 11, -1, 1};

static double ai0_data[21] = {
   .07575994494023796, 
   .00759138081082334,
   .00041531313389237,
   .00001070076463439,
  -.00000790117997921,
  -.00000078261435014,
   .00000027838499429,
   .00000000825247260,
  -.00000001204463945,
   .00000000155964859,
   .00000000022925563,
  -.00000000011916228,
   .00000000001757854,
   .00000000000112822,
  -.00000000000114684,
   .00000000000027155,
  -.00000000000002415,
  -.00000000000000608,
   .00000000000000314,
  -.00000000000000071,
   .00000000000000007
};
static cheb_series ai0_cs = {ai0_data, 20, -1, 1};

static double ai02_data[22] = {
   .05449041101410882,
   .00336911647825569,
   .00006889758346918,
   .00000289137052082,
   .00000020489185893,
   .00000002266668991,
   .00000000339623203,
   .00000000049406022,
   .00000000001188914,
  -.00000000003149915,
  -.00000000001321580,
  -.00000000000179419,
   .00000000000071801,
   .00000000000038529,
   .00000000000001539,
  -.00000000000004151,
  -.00000000000000954,
   .00000000000000382,
   .00000000000000176,
  -.00000000000000034,
  -.00000000000000027,
   .00000000000000003
};
static cheb_series ai02_cs = {ai02_data, 21, -1, 1};

static double bi1_data[11] = {
  -0.001971713261099859,
   0.407348876675464810,
   0.034838994299959456,
   0.001545394556300123,
   0.000041888521098377,
   0.000000764902676483,
   0.000000010042493924,
   0.000000000099322077,
   0.000000000000766380,
   0.000000000000004741,
   0.000000000000000024
};
static cheb_series bi1_cs = {bi1_data, 10, -1, 1};

static double ai1_data[21] = {
  -0.02846744181881479,
  -0.01922953231443221,
  -0.00061151858579437,
  -0.00002069971253350,
   0.00000858561914581,
   0.00000104949824671,
  -0.00000029183389184,
  -0.00000001559378146,
   0.00000001318012367,
  -0.00000000144842341,
  -0.00000000029085122,
   0.00000000012663889,
  -0.00000000001664947,
  -0.00000000000166665,
   0.00000000000124260,
  -0.00000000000027315,
   0.00000000000002023,
   0.00000000000000730,
  -0.00000000000000333,
   0.00000000000000071,
  -0.00000000000000006
};
static cheb_series ai1_cs = { ai1_data, 20, -1, 1};

static double ai12_data[22] = {
   0.02857623501828014,
  -0.00976109749136147,
  -0.00011058893876263,
  -0.00000388256480887,
  -0.00000025122362377,
  -0.00000002631468847,
  -0.00000000383538039,
  -0.00000000055897433,
  -0.00000000001897495,
   0.00000000003252602,
   0.00000000001412580,
   0.00000000000203564,
  -0.00000000000071985,
  -0.00000000000040836,
  -0.00000000000002101,
   0.00000000000004273,
   0.00000000000001041,
  -0.00000000000000382,
  -0.00000000000000186,
   0.00000000000000033,
   0.00000000000000028,
  -0.00000000000000003
};
static cheb_series ai12_cs = {ai12_data, 21, -1, 1};


/* ----------------------------------------- */
/* ------------ Bessel Functions------------ */
/* ----------------------------------------- */

double cheb_eval(cheb_series * cs, double x) {
	int j;
	double d  = 0.0;
	double dd = 0.0;

	double y  = (2.0*x - cs->a - cs->b) / (cs->b - cs->a);
	double y2 = 2.0 * y;

	double e = 0.0;

	for(j = cs->order; j>=1; j--) {
		double temp = d;
		d = y2*d - dd + cs->c[j];
		e += fabs(y2*temp) + fabs(dd) + fabs(cs->c[j]);
		dd = temp;
	}

	double temp = d;
	d = y*d - dd + 0.5 * cs->c[0];
	e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs->c[0]);
	return d;
}

double i0_scaled(double x) {
	double y = fabs(x);

	if(y < 2.0 * 1.490116e-08) {
		return 1.0 - y;
	}
	else if(y <= 3.0) {
		return exp(-y) * (2.75 + cheb_eval(&bi0_cs, y*y/4.5-1.0));
	}
	else if(y <= 8.0) {
		return (0.375 + cheb_eval(&ai0_cs, (48.0/y-11.0)/5.0))/ sqrt(y);
	}
	else {
		return (0.375 + cheb_eval(&ai02_cs, 16.0/y-1.0))/ sqrt(y);
	}
}

double i0(double x) {
	double y = fabs(x);

	if(y < 2.0 * 1.490116e-08) {
		return 1.0;
	}
	else if(y <= 3.0) {
		return 2.75 + cheb_eval(&bi0_cs, y*y/4.5-1.0);
	}
	else if(y < 7.097827e+02 - 1.0) {
		return exp(y) * i0_scaled(x);
	}
	else {
		return NAN;
	}
}

double i1_scaled(double x) {
	const double xmin    = 4.450148e-308;
	const double x_small =  4.214685e-08;
	double y = fabs(x);

	if(y == 0.0) {
		return 0.0;
	}
	else if(y < x_small) {
		return 0.5*x;
	}
	else if(y <= 3.0) {
		return x * exp(-y) * (0.875 + cheb_eval(&bi1_cs, y*y/4.5-1.0));
	}
	else if(y <= 8.0) {
		double b;
		double s;
		b = (0.375 + cheb_eval(&ai1_cs, (48.0/y-11.0)/5.0)) / sqrt(y);
		s = (x > 0.0 ? 1.0 : -1.0);
		return s * b;
	}
	else {
		double b;
		double s;
		b = (0.375 + cheb_eval(&ai12_cs, 16.0/y-1.0)) / sqrt(y);
		s = (x > 0.0 ? 1.0 : -1.0);
		return s * b;
	}
}

double i1(double x) {
	const double xmin    = 4.450148e-308;
	const double x_small =  4.214685e-08;
	double y = fabs(x);

	if(y == 0.0) {
	    return 0.0;
	}
	else if(y < xmin) {
		return NAN;
	}
	else if(y < x_small) {
		return 0.5*x;
	}
	else if(y <= 3.0) {
		return x * (0.875 + cheb_eval(&bi1_cs, y*y/4.5-1.0));
	}
	else if(y < 7.097827e+02) {
		return exp(y) * i1_scaled(x);
	}
	else {
		return NAN;
	}
}



/*
#########################################################
	Integrand for BIASD
#########################################################
*/


double integrand(int n, double args[n]) {
	double f = args[0];
	double d = args[1];
	double ep1 = args[2];
	double ep2 = args[3];
	double sigma = args[4];
	double k1 = args[5];
	double k2 = args[6];
	double tau = args[7];

	double out;

	double k = k1 + k2;
	double p1 = k2/k;
	double p2 = k1/k;
	double y = 2.*k*tau * pow(p1*p2*f*(1.-f),.5);
	double z = p2*f + p1*(1.-f);

	if (f < 0. || f > 1. || k1 <= 0. || k2 <= 0. || sigma <= 0. || tau <= 0. || ep1 >= ep2) {
		out = 0.;
	}
	else {
		out = 2.*k*tau*p1*p2*(i0(y)+k*tau*(1.-z)*i1(y)/y)*exp(-z*k*tau);
		out *= 1./sigma * M_2_SQRTPI/2. * M_SQRT1_2 * exp(-.5/sigma/sigma*pow(d-(ep1*f+ep2*(1.-f)),2.));
	}
	return out;
}


/* 
How to compile for CTYPES in python:
Note: This doesn't need GSL b/c it is the same fxns
1) $ gcc -L/usr/local/lib -shared -o integrand_gsl.so -fPIC -O3 integrand_gsl.c -lm
*/