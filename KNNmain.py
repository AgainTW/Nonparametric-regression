import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from KNN import KNN

# main
if __name__ == '__main__':
	# load & pre-processing
	npz_data = np.load('data/data2.npz')
	point = list(npz_data['X'])
	label = list(npz_data['y'])
	# convert x to [(), ()] type
	if npz_data['X'].ndim==1:
		x = []
		for i in point:
			temp = []
			temp.append(i)
			x.append(tuple(temp))		
	else:
		x = []
		for i in point:
			x.append(tuple(i))

	# main	
	knn = KNN(k=5, p=2, mod='m')
	knn.fit(x, label)

	# 2D-plot
	if npz_data['X'].ndim==1:
		predict_label = []
		predict_x = []
		for i in range(0,100):
			predict_x.append(i/10)
			predict_label.append( knn.predict((i/10,)) )
		plt.scatter(x,label)
		plt.plot(predict_x, predict_label, color='r', marker="o")
		plt.show()

	# 3D-plot
	if npz_data['X'].ndim==2:
		predict_label = []
		predict_x = []
		for i in range(0,60):
			predict_x.append( (i-30)/10 )
			predict_label.append( knn.predict(((i-30)/10, (i-30)/10)) )
		predict_label = np.array(predict_label).reshape(len(range(0,60)),1)

		ax = plt.subplot(111, projection='3d')
		x = npz_data['X'].reshape(len(npz_data['X']),len(npz_data['X'].T))
		ax.scatter(x[:,0], x[:,1], label, color='b')
		ax.scatter(predict_x, predict_x, predict_label, color='r')
		plt.show()