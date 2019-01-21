import numpy as np
from sklearn.datasets import load_digits
data = load_digits()
X = data['images']
X = 2 * (X.reshape((X.shape[0], 64)) / 16.0) - 1
y = data['target']
data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
np.savetxt('data.csv', data, header=','.join([str(i) for i in range(X.shape[1] + 1)]), delimiter=',', comments='')
print(X)
