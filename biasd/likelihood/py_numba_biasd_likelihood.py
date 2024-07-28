import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="FNV hashing is not implemented in Numba")

import numpy as np
import numba as nb
from math import pow,exp,sqrt,log,fabs

wg = np.array([
	0.066671344308688137593568809893332,
	0.149451349150580593145776339657697,
	0.219086362515982043995534934228163,
	0.269266719309996355091226921569469,
	0.295524224714752870173892994651338
],dtype='double')
wgk = np.array([
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
],dtype='double')
xgk = np.array([
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
],dtype='double')

# Bessels are from numerical recipies in C?
@nb.jit(nb.double(nb.double),nopython=True)
def bessel_i0(x):
	ax = fabs(x)
	if ax < 3.75:
		y = x/3.75
		y *= y
		return 1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
			+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))))
	else:
		y = 3.75/ax
		ans = (exp(ax)/sqrt(ax)) * (0.39894228+y*(0.1328592e-1+y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1+y*0.392377e-2))))))))
	return ans

@nb.jit(nb.double(nb.double),nopython=True)
def bessel_i1(x):
	ax = fabs(x)
	if ax < 3.75:
		y = x/3.75
		y *= y
		ans = ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934+y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))))
	else:
		y = 3.75/ax
		ans = 0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1-y*0.420059e-2))
		ans = 0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))))
		ans *= (exp(ax)/sqrt(ax))
	return fabs(ans)

@nb.jit(nb.double(nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double),nopython=True)
def integrand(f,d,ep1,ep2,sig1,sig2,k1,k2,tau):
	y = 2.*tau*sqrt(k1*k2*f*(1.-f));
	var = sig1 * sig1 * f + sig2 * sig2 * (1.-f);

	return exp(-(k1*f+k2*(1.-f))*tau)/ sqrt(var)* exp(-.5*pow((d - (ep1 * f + ep2 * (1.-f))),2.)/var)* (bessel_i0(y) + (k2*f+k1*(1.-f))*tau* bessel_i1(y)/y)


@nb.jit(nb.double[:](nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double),nopython=True)
def adaptive_integrate(a0,b0,epsilon,d,ep1,ep2,sig1,sig2,k1,k2,tau):

	# // a is the lowerbound of the integral
	# // b is the upperbound of the integral
	# // out[0] is the value of the integral
	# // out[1] is the error of the integral
	# // epsilon is the error limit we must reach in each sub-interval

	a = a0
	b = b0
	out = np.array((0.,0.))

	# // This loops from the bottom to the top, subdividing and trying again on the lower half, then moving up once converged to epsilon
	while (a < b0):
		# // Evaluate quadrature on current interval

		# // Perform the quadrature
		# // Do quadrature with the Gauss-10, Kronrod-21 rule
		# // translated from Fortran from Quadpack's dqk21.f...

		center = 0.5*(b+a)
		halflength = 0.5*(b-a)

		result10 = 0.
		result21 = 0.

		fval1 = np.array((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.))
		fval2 = np.array((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.))
		# // Pre-calculate function values

		for i in range(10):
			fval1[i] = integrand(center+halflength*xgk[i],d,ep1,ep2,sig1,sig2,k1,k2,tau)
			fval2[i] = integrand(center-halflength*xgk[i],d,ep1,ep2,sig1,sig2,k1,k2,tau)

		for i in range(5):
			# // Evaluate Gauss-10
			result10 += wg[i]*(fval1[2*i+1] + fval2[2*i+1])

			# //Evaluate Kronrod-21
			result21 += wgk[2*i]*(fval1[2*i] + fval2[2*i])
			result21 += wgk[2*i+1]*(fval1[2*i+1] + fval2[2*i+1])

		# // Add 0 point to Kronrod-21
		fc = integrand(center,d,ep1,ep2,sig1,sig2,k1,k2,tau)
		result21 += wgk[10]*fc

		# // Scale results to the interval
		result10 *= fabs(halflength)
		result21 *= fabs(halflength)

		# // Error calculation
		avgg = result21/fabs(b-a)
		errors = epsilon ## set to epsilon instead of 0 b/c stops weird divide by zero errors

		for i in range(5):
			errors += wgk[2*i]*(fabs(fval1[2*i]-avgg)+fabs(fval2[2*i]-avgg))
			errors += wgk[2*i+1]*(fabs(fval1[2*i+1]-avgg)+fabs(fval2[2*i+1]-avgg))

		errors += wgk[10]*(fabs(fc-avgg))
		qq = pow(200.*fabs(result21-result10)/errors,1.5)
		if qq < 1:
			errors *= qq

		# // Check convergence on this interval
		if (errors < epsilon ):
			# // keep the results
			out[0] += result21
			out[1] += errors
			# // step the interval
			a = b
			b = b0
		else:
			# // sub-divide the interval
			b -= 0.5*(b-a)

	return out

@nb.jit([nb.double[:](nb.double[:],nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double,nb.double)],nopython=True,parallel=True)
def _log_likelihood(d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon):
	out = np.zeros_like(d)

	if sigma1 <= 0 or sigma2 <= 0 or k1 <= 0 or k2 <= 0:
		return out-np.inf

	for i in nb.prange(d.size):
		# Calculate the state contributions
		out[i]  = k2/(k1+k2) / sigma1 * np.exp(-k1*tau - .5 * ((d[i]-ep1)/sigma1)**2.) # state 1
		out[i] += k1/(k1+k2) / sigma2 * np.exp(-k2*tau - .5 * ((d[i]-ep2)/sigma2)**2.) # state 2

		# Perform the blurring integral
		intval = adaptive_integrate(0.,1.,epsilon,d[i],ep1,ep2,sigma1,sigma2,k1,k2,tau)

		out[i] += 2.*k1*k2/(k1+k2)*tau *intval[0] # the blurring contribution
		out[i] = np.log(out[i]) - .5 * np.log(2.* np.pi) # Add prefactor
	return out

def log_likelihood_numba(theta,data,tau,device=None):
	epsilon=1e-16
	ep1,ep2,sigma1,k1,k2 = theta
	sigma2 = sigma1
	return np.nansum(_log_likelihood(data,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon))

def nosum_log_likelihood_numba(theta,data,tau,device=None):
	epsilon=1e-16
	ep1,ep2,sigma1,k1,k2 = theta
	sigma2 = sigma1
	return _log_likelihood(data,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon)
