#ifndef BIASD_LL_H_
#define BIASD_LL_H_

double * log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau);
//double sum_log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau);

double bessel_i0(double x);
double bessel_i1(double x);

double integrand(double f, double d, double ep1, double ep2, double sigma, double k1, double k2, double tau);
double simpson(int N, double * x, double * y);

#endif 
