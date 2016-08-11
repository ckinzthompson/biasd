#ifndef BIASD_LL_H_
#define BIASD_LL_H_

typedef struct integral_params { double d; double ep1; double ep2; double sigma; double k1; double k2; double tau;} ip;
typedef struct fval {double x; double y;} fval_t;
typedef struct node { fval_t * val; struct node * next;} node_t;
typedef struct check_out {int success; double integral; double x_mid; double y_mid;} cs_out;

// #############################################################################
// #############################################################################

node_t * new_list();
fval_t * new_fval(double x, double y);
void push(node_t ** head, fval_t *val);
fval_t * pop(node_t ** head);
void free_list(node_t ** head);

double bessel_i0(double x);
double bessel_i1(double x);
double integrand(double f,ip * p);
cs_out check_simpson(fval_t * va, fval_t * vb, double epsilon, ip * args);
double adaptive_quad(double epsilon, ip * args);
void log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon, double * out);
double sum_log_likelihood(int N, double *d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon);

#endif 
