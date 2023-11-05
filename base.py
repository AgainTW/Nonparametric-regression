import numpy as np

# minkowski
from math import *
from decimal import Decimal

def p_root(value, root):
	root_value = 1 / float(root)
	return round ( Decimal(value)**Decimal(root_value), 3 )

def minkowski_distance(x, y, p_value):
	temp = []
	for i in range(len(x)):
		a = x[i]
		b = y[i]
		temp.append( pow(abs(a-b), p_value) )
	return ( p_root(sum(temp), p_value) )

# Weight Matrix
def weight_matrix(point, X, tau): 
	m = X.shape[0] 
	w = np.mat(np.eye(m)) 

	for i in range(m):
		xi = X[i] 
		d = (-2 * tau * tau)
		w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d) 
	return w