from Layer.Dense import Dense
from Model.Model import Model  #import layer and model class

import numpy as np 

x = [
    [0, 0],
    [1, 0],
    [1, 1],   
    [0, 0],
    [0, 0],
    [0, 1]
]

y = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
]

x = np.array(x).astype(np.float32)
y = np.array(y).astype(np.float32)     #data we're trying to train

layers = [
    Dense(nods=10, activation='relu', input_shape=(2, )),  #10 neuron dense layer
    Dense(nods=10, activation='relu'),   #same
    Dense(nods=3, activation='softmax')  #3 neuron dense layer. 
]

model = Model(layers=layers, optimizer='gd', loss='cce', learning_rate=0.1)
model.fit(X=x, Y=y, epochs=10000)
model.accuracy(x, y)