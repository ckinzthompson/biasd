//Ripped straight from JW vdM's forwardback.cpp for MatLab Mex-file stuff from ebFRET.
//Turned it into C code b/c it makes more sense and also... for use with ctypes in python
//Redid the indexing to make sense...

#include <math.h>
#include <stdlib.h>
// #include <stdio.h> //debug printing
	
void fb_once(int T, int K, double *p_x_z, double *A, double *pi, double *g, double *xi, double *ln_z);

void forward_backward(int M, int T, int K, double *p_x_z, double *A, double *pi, double *g, double *xi, double *ln_z);

void fb_once(int T, int K, double *p_x_z, double *A, double *pi, double *g, double *xi, double *ln_z)	
{

	double *a, *b, *c;
	a = malloc(T*K*sizeof(double));
	b = malloc(T*K*sizeof(double));
	c = malloc(T*sizeof(double));
	
	
	// initialize to zero
	int i;
	for (i=0; i<T*K; i++)
	{
		a[i] = 0;
		b[i] = 0;
	}
	for (i=0; i<T; i++)
	{
		c[i] = 0;
	}

	// Forward Sweep - Calculate
	//
	//   a(t, k)  =  sum_l p_x_z(t,k) A(l, k) alpha(t-1, l)  
	//   c(t)	 =  sum_k a(t, k)
	//
	// and normalize 
	//
	//   a(t, k)  /=  c(t)

	// a(0, k)  =  p_x_z(0, k) pi(k)
	int k;
	for (k = 0; k < K; k++) 
	{
		a[0*K + k] = pi[k] * p_x_z[0*K + k];
		c[0] += a[0*K + k];
	}
	// normalize a(0,k) by c(k)
	for (k = 0; k < K; k++) 
	{
		a[0*K + k] /= c[0];
	}

	int t = 0;
	int l;
	for (t = 1; t < T; t++)
	{
		// a(t, k)  =  sum_l p_x_z(t,k) A(l, k) alpha(t-1, l)  
		for (k = 0; k < K; k++) 
		{
			for (l = 0; l < K; l++) 
			{
				// a(t,k) +=  p_x_z(t,k) A(l, k) alpha(t-1, l)  
				a[t*K + k] += p_x_z[t*K + k] * A[l*K + k] * a[(t-1)*K + l];
			}			
			// c(t) += a(t,k)
			c[t] += a[t*K + k];
		}
		// normalize a(t,k) by c(t)
		for (k = 0; k < K; k++) 
		{
			a[t*K + k] /= c[t];
		}
	}
		
	// Back sweep - calculate
	//
	// b(t,k)  =  1/c(t+1) sum_l p_x_z(t+1, l) A(k, l) beta(t+1, l) 

	// b(T-1,k) = 1
	for (k = 0; k < K; k++) 
	{
		b[(T-1)*K + k] = 1;
	}

	// t = T-2:0
	for (t = T-2; t >= 0; t--)
	{
		// b(t, k)  =  sum_l p_x_z(t+1,l) A(k, l) beta(t+1, l)  
		for (k = 0; k < K; k++) 
		{
			for (l = 0; l < K; l++) 
			{
				// b(t ,k) += p_x_z(t+1, l) A(k, l) betal(t+1, l)  
				b[t*K + k] += p_x_z[(t+1)*K + l] * A[k*K + l] * b[(t+1)*K + l];
			}			
			// normalize b(t,k) by c(t+1)
			b[t*K + k] /= c[t+1];
		}
	}
	
	// g(t,k) = a(t,k) * b(t,k)
	for (i=0; i<T*K; i++)
	{
		g[i] = a[i] * b[i];
	}
	// xi(t, k, l) = alpha(t, k) A(k,l) p_x_z(t+1, l) beta(t+1, l) / c(t+1)
	for (t = 0; t < T-1; t++)
	{
		for (k = 0; k < K; k++) 
		{
			for (l = 0; l < K; l++) 
			{
				xi[t*K*K + k*K +l] = (a[t*K + k] \
					* A[k*K + l] \
					* p_x_z[(t+1)*K + l] \
					* b[(t+1)*K + l]) / c[t+1];
			}			
		}
	}
	// ln_Z = sum_t log(c[t])
	// ln_z[0] = 0;
	for (t=0; t<T; t++)
	{
		*ln_z += log(c[t]);
	}

	// delete memory allocated for a, b and c
	free(a);
	free(b);
	free(c);

	return;
}

void forward_backward(int M, int T, int K, double *p_x_z, double *A, double *pi, double *g, double *xi, double *ln_z){
	// data is assumed to be flat MxTxK -->
	int i = 0;
	for (i=0;i<M;i++){
		fb_once(T, K, &(p_x_z[i*T*K]), A, pi, &(g[i*T*K]), &(xi[i*(T-1)*K*K]), ln_z);
		// printf("%f, %f, %f, %f\n", p_x_z[i*T*K], g[i*T*K], xi[i*(T-1)*K*K], *ln_z);
	}
}

// gcc -shared -o ./forward_backward-linux.so -fPIC -O3 ./forward_backward.c
// gcc -shared -o ./forward_backward-mac.so -fPIC -O3 ./forward_backward.c