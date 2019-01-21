from sklearn.datasets import make_swiss_roll
import numpy as np
X, y = make_swiss_roll(n_samples=1000)
X = (X - X.mean(axis=0)) / (X.std(axis=0))
data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
np.savetxt('data.csv', data, header='x,y,z,c', delimiter=',', comments='')
print(X)
