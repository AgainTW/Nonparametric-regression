import numpy as np
import math
from base import minkowski_distance
from collections import defaultdict

# KNN
class KNN(object):
	def __init__(self, k=20, p=2, *, mod='f'):
		self.data = None
		self.k = k
		self.p = p
		self.mod = mod

	def distance(self, p1, p2):
		return minkowski_distance(p1, p2, self.p)

	def fit(self, X, y):
		self.data = dict(zip(X, y))

	def predict(self, point):
		distances = {}

		for p, _ in self.data.items():
			distances[p] = self.distance(p, point)

		sort_distances = dict(sorted(distances.items(), key=lambda x: x[1]))

		if(self.mod=='f'):
			topk = defaultdict(int)
			for idx, (p, v) in enumerate(sort_distances.items()):
				if idx == self.k - 1:
					break
				topk[self.data[p]] += 1

			topk = sorted(topk.items(), key=lambda x: -x[1])
			return topk[0][0]
		elif(self.mod=='m'):
			mean = 0
			for idx, (p, v) in enumerate(sort_distances.items()):
				if idx == self.k - 1:
					break
				mean += self.data[p]

			return mean/(self.k-1)		