# nnxor/nnxor.py

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

np.random.seed(444)     # seed random function

# XOR Truth Table
X = np.array([[0, 0],                   # inputs
             [0, 1],
             [1, 0],
             [1, 1]])
y = np.array([[0], [1], [1], [0]])      # outputs

# define the neural network
model = Sequential()
model.add(Dense(2, input_dim=2))    # first layer, two neurons
model.add(Activation('sigmoid'))    # sigmoid activation function (input & outpu)
model.add(Dense(1))                 # output layer, one neuron
model.add(Activation('sigmoid'))

# train the network
sgd = SGD(lr=0.1)                   # Stochastic Gradient Descent, learning rate = 0.1
model.compile(loss='mean_squared_error', optimizer=sgd)     # mean squared error as loss function to be minimized

model.fit(X, y, batch_size=1, epochs=5000)      # update weights after every training example

if __name__ == '__main__':
    print(model.predict(X))
