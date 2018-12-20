#ifndef BIASD_2S_LL_H_
#define BIASD_2S_LL_H_

// External calls from python
// extern "C" void log_likelihood(int device, int N, double * d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll);
// extern "C" double sum_log_likelihood(int device, int N, double *d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon);

// BIASD related functions
__global__ void kernel_loglikelihood(int N, double * d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll);

extern "C" void load_dataz(int device, int N, double * d, double * ll, void * d_d, void * ll_d);
extern "C" void free_data(void *d_d, void *ll_d);
extern "C" void log_likelihood(int device, int N, void *d_d, void *ll_d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll);
extern "C" double sum_log_likelihood(int device, int N, void *d_d, void *ll_d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon);

#endif
