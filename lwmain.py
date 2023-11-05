import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lw import lw

# main
if __name__ == '__main__':
	# load & pre-processing
	npz_data = np.load('data/data2.npz')
	label = npz_data['y'].reshape(len(npz_data['y']),1)
	if npz_data['X'].ndim==1:
		x = npz_data['X'].reshape(len(npz_data['X']),1)
	else:	
		x = npz_data['X'].reshape(len(npz_data['X']),len(npz_data['X'].T))	

	# main	
	lw = lw(tau=10)
	lw.fit(x, label)

	# 2D-plot
	if npz_data['X'].ndim==1:
		predict_label = []
		predict_x = []
		for i in range(0,100):
			predict_x.append(i/10)
			predict_label.append(lw.predict([i/10]))
		predict_label = np.array(predict_label).reshape(len(range(0,100)),1)
		plt.scatter(x,label)
		plt.plot(predict_x, predict_label, color='r', marker="o")
		plt.show()

	# 3D-plot
	if npz_data['X'].ndim==2:
		predict_label = []
		predict_x = []
		for i in range(0,60):
			predict_x.append( (i-30)/10 )
			a = [(i-30)/10, (i-30)/10]
			predict_label.append(lw.predict(a))
		predict_label = np.array(predict_label).reshape(len(range(0,60)),1)

		ax = plt.subplot(111, projection='3d')
		ax.scatter(x[:,0], x[:,1], label, color='b')
		ax.scatter(predict_x, predict_x, predict_label, color='r')
		plt.show()
