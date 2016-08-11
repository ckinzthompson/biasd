#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "biasd_adaptive.h"


// #############################################################################
// #############################################################################

node_t * new_list(){
	node_t * head;
	head = malloc(sizeof(node_t));
	(*head).next = NULL;
	(*head).val = NULL;
	return head;
}

fval_t * new_fval(double x, double y){
	fval_t * newf;
	newf = malloc(sizeof(fval_t));
	(*newf).x = x;
	(*newf).y = y;
	return newf;
}

void push(node_t ** head, fval_t * val){
	if (val != NULL) {
		node_t * new_node;
		new_node = malloc(sizeof(node_t));
	
		(*new_node).val = val;
		(*new_node).next = *head;
	
		*head = new_node;
	}
}

fval_t * pop(node_t ** head){
	if (*head != NULL){
		node_t * next = (*head)->next;
		fval_t * popped = (*head)->val;
		free(*head);
		*head = next;
		return popped;
	}
	else{
		return NULL;
	}
	
}

void free_list(node_t ** head){
	fval_t * empty;
	
	while (1){
		empty = pop(head);
		if (empty == NULL){
			break;
		}
	}
}

// #############################################################################
// #############################################################################

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

double integrand(double f, ip * p) {
	double y, out;
	
	y = 2.*p->tau*sqrt(p->k1*p->k2*f*(1.-f));
	
	if (f == 0.){
		out = exp(-1. * p->k2 * p->tau) * (1.+ p->k1 * p->tau / 2.) * exp(-.5 * pow((p->d - p->ep2)/p->sigma,2.)); // Limits
	} else if (f == 1.){
		out = exp(-1. * p->k1 * p->tau) * (1.+ p->k2 * p->tau / 2.) * exp(-.5 * pow((p->d - p->ep1)/p->sigma,2.)); //Limits
	} else if ((f > 0.) && (f < 1.)){
		out = exp(-(p->k1*f+p->k2*(1.-f))*p->tau)
			* exp(-.5*pow((p->d - (p->ep1 * f +
				p->ep2 * (1.-f)))/p->sigma,2.))
			* (bessel_i0(y) + (p->k2*f+p->k1*(1.-f))*p->tau
				* bessel_i1(y)/y); // Full Expression
	} else {
		out = NAN;
	}
	return out;
}

// #############################################################################
// #############################################################################

cs_out check_simpson(fval_t * va, fval_t * vb, double epsilon, ip * args) {
	
	double a = va->x;
	double b = vb->x;
	
	cs_out out = {0,0.,0.,0.};
	double h = (b-a)/2.;
	double xmid = a + h;
	double ymid = integrand(xmid,args);
	
	// Calc Simpson 1 division
	double s1 = h/3. * (va->y + 4. * ymid + vb->y);

	// Calc Simpson 2 divisions
	double s2_left = h/6. * (va->y + 4.*integrand(a + h/2.,args) + ymid);
	double s2_right = h/6. * (ymid + 4.*integrand(a + h*1.5,args) + vb->y);
	
	double err = fabs(s1 - s2_left - s2_right);

	if (err < 15.*epsilon) {
		out.success = 1;
		out.integral = s2_left+s2_right;
	} else {
		out.x_mid = xmid;
		out.y_mid = ymid;
	}
	return out;
}

double adaptive_quad(double epsilon, ip * args){
	double integral = 0.;
	cs_out check;
	node_t * head = new_list();
	push(&head,new_fval(1.,integrand(1.,args)));
	push(&head,new_fval(0.,integrand(0.,args)));
	
	fval_t * v1, * v2;
	
	int i;
	for (i = 0; i < 2000; i++) { // max iterations = 2000
		v1 = pop(&head);
		v2 = pop(&head);
	
		if ((v1 != NULL) && (v2 != NULL)){
			check = check_simpson(v1,v2,epsilon,args);
			if (check.success){
				// printf("%e, %e   =   %e\n",v1->x,v2->x,check.integral );
				integral += check.integral;
				push(&head,v2);
			} else {
				push(&head,v2);
				push(&head,new_fval(check.x_mid,check.y_mid));
				push(&head,v1);
			}
		}
		else {
			break;
		}
	}
	free_list(&head);
	return integral;
}

// #############################################################################
// #############################################################################

double * log_likelihood(int N, double * d, double ep1, double ep2, double sigma, double k1, double k2, double tau, double epsilon) {
	
	double * out;
	out = (double *) malloc(N*sizeof(double));
	
	ip p = {0.,ep1,ep2,sigma,k1,k2,tau};
	
	double lli;

	int i;
	for (i=0;i<N;i++){
		
		// Peak for state 1
		lli = k2/(k1+k2) * exp(-1. * k1 * tau - .5 * pow((d[i] - ep1) / sigma,2.));
		// Peak for state 2
		lli += k1/(k1+k2) * exp(-1.* k2 * tau - .5 * pow((d[i] - ep2) / sigma,2.));

		// Add in the contribution from the numerical integration
		p.d = d[i];
		lli += 2.*k1 * k2/(k1 + k2) * tau * adaptive_quad(epsilon,&p);

		// Log and get the prefactor
		lli = log(lli) - .5 * log(2.* M_PI) - log(sigma); 
		out[i] = lli;
	}
	return out;
}
