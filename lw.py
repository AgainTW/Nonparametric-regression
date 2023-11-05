import numpy as np
import math
from base import weight_matrix

# Locally weighted regression
class lw(object):
	def __init__(self, tau=20):
		self.data = None
		self.x = None
		self.y = None
		self.m = None
		self.tau = tau

	def wm(self, point, X, tau): 
		return weight_matrix(point, X, tau)

	def fit(self, X, y):
		self.m = X.shape[0]
		self.y = y	
		one_array = np.ones(len(X))
		one_array = one_array.reshape(len(one_array),1)
		self.x = np.column_stack((X, one_array))
			

	def predict(self, point): 
		point_ = np.concatenate((point, np.ones(1)))
		self.data = self.wm(point_, self.x, self.tau)
		theta = np.linalg.pinv(self.x.T*(self.data * self.x))*(self.x.T*(self.data * self.y))
		pred = np.dot(point_, theta)

		return pred
    
