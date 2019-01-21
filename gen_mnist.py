import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train /= 255.0
x_train = x_train.reshape((x_train.shape[0], -1))
x_train = x_train * 2 - 1
X = x_train[0:10000]
y = y_train[0:10000]
data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
np.savetxt('data.csv', data, header=','.join([str(i) for i in range(X.shape[1] + 1)]), delimiter=',', comments='')
print(X)
