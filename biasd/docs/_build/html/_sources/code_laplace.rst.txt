.. _code_laplace:

Laplace
=============

This page gives the details about the code in biasd.laplace.


Laplace Approximation
+++++++++++++++++++++

In order to calculate the Laplace approximation to the posterior probability distribution, you must calculate the second derivative of the log-posterior function at the maximum a posteriori (MAP) estimate. This module contains code to calculate the finite difference Hessian, find the MAP estimate of the BIASD log-posterior using numerical maximization (Nelder-Mead), and apply this analysis to a time series. You should probably only need to use the `biasd.laplace.laplace_approximation()` function.


.. automodule:: laplace
	:members:


